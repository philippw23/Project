"""BiomedCLIP image+text downstream classifier — concatenated 512+512→1024-dim features.

L2-normalized post-projection image and text embeddings are concatenated and fed into a
linear or MLP probing head. The full encoder (including projection heads) is always frozen.

Head variants (--head):
    linear       — single nn.Linear(1024, 3), no metadata
    mlp          — MLP with age/sex late fusion (default)
    mlp_no_meta  — same MLP capacity, no clinical metadata

No BTXRD evaluation: BTXRD samples carry no report text, so the text-encoder
pathway this baseline depends on has nothing to embed for that dataset. There
is no --btxrd_manifest flag here.

Usage:
    python src/biomedclip_img_text_downstream.py \\
        --checkpoint results/biomedclip_pretrain/.../best_r1_checkpoint.pt \\
        --splits     data/internal_dataset/split.json \\
        --head       mlp --early_stopping_metric val_loss
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.utils.misc import ROOT_DIR, MODEL_TAG, DEFAULT_OUT_DIR, DEFAULT_SPLITS
from biomedclip.data.datasets import DownstreamDatasetWithText, EmbeddingDataset
from biomedclip.data.transforms import build_train_transform
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.lora import inject_lora
from biomedclip.models.classifier import LinearHead, MalignancyMLP
from biomedclip.utils.downstream_eval import resolve_label_maps, report_eval, compute_auroc, safe_wandb_log

DEFAULT_CHECKPOINT = ROOT_DIR / "results" / "biomedclip_pretrain" / "best_r1_checkpoint.pt"
EMBED_DIM = 1024  # 512-dim image + 512-dim text


def build_head(args: argparse.Namespace, embed_dim: int, device: torch.device,
               num_classes: int) -> nn.Module:
    if args.head == "linear":
        return LinearHead(embed_dim, num_classes=num_classes).to(device)
    elif args.head == "mlp_no_meta":
        return MalignancyMLP(embed_dim, args.hidden_dims, args.dropout,
                             args.meta_embed_dim, use_meta=False,
                             num_classes=num_classes).to(device)
    else:
        return MalignancyMLP(embed_dim, args.hidden_dims, args.dropout,
                             args.meta_embed_dim, use_meta=True,
                             num_classes=num_classes).to(device)


@torch.no_grad()
def extract_img_text_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-compute concatenated L2-normalised image+text embeddings for one split."""
    model.eval()
    all_emb, all_age, all_sex, all_lbl = [], [], [], []
    for batch in tqdm(loader, desc="  embedding", leave=False):
        images  = batch["image"].to(device)
        texts   = batch["text"].to(device)
        img_emb = F.normalize(model.encode_image(images), dim=-1)
        txt_emb = F.normalize(model.encode_text(texts),   dim=-1)
        all_emb.append(torch.cat([img_emb, txt_emb], dim=-1).cpu())
        all_age.append(batch["age"])
        all_sex.append(batch["sex"])
        all_lbl.append(batch["label"])
    return (
        torch.cat(all_emb),
        torch.cat(all_age),
        torch.cat(all_sex),
        torch.cat(all_lbl),
    )

def train_one_epoch_aug(
    head:      nn.Module,
    encoder:   nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """Training loop with on-the-fly augmentation.

    Passes augmented images through the frozen encoder each batch so that
    random augmentations produce different embeddings every epoch.
    """
    head.train()
    encoder.eval()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        texts  = batch["text"].to(device)
        age    = batch["age"].to(device)
        sex    = batch["sex"].to(device)
        lbl    = batch["label"].to(device)
        with torch.no_grad():
            img_emb = F.normalize(encoder.encode_image(images), dim=-1)
            txt_emb = F.normalize(encoder.encode_text(texts),   dim=-1)
            emb     = torch.cat([img_emb, txt_emb], dim=-1)
        optimizer.zero_grad()
        loss = criterion(head(emb, age, sex), lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    head.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    for emb, age, sex, lbl in loader:
        emb, age, sex, lbl = emb.to(device), age.to(device), sex.to(device), lbl.to(device)
        logits = head(emb, age, sex)
        total_loss += criterion(logits, lbl).item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(lbl.cpu())
        all_probs.append(F.softmax(logits, dim=-1).cpu())
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = torch.cat(all_probs).numpy()
    return total_loss / len(loader), float((preds == labels).mean()), preds, labels, probs


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BiomedCLIP img+text downstream — concatenated post-projection embeddings."
    )
    # ── Checkpoint / splits ──────────────────────────────────────────────────
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--freezed_biomedclip", action="store_true",
                        help="Use vanilla BiomedCLIP weights (no checkpoint, no LoRA).")
    parser.add_argument("--splits",   default=str(DEFAULT_SPLITS))
    parser.add_argument("--out_dir",  default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--binary", type=lambda x: str(x).lower() in ("true", "1", "yes"),
                        default=False,
                        help="Binary classification (benign vs malignant). "
                             "Use with split_binary.json — intermediate cases must already be excluded.")
    parser.add_argument("--eval_test", action="store_true",
                        help="Evaluate the split's held-out test set after training. Off by "
                             "default so hyperparameter sweeps never touch test. No BTXRD "
                             "evaluation — BTXRD carries no report text for the text encoder.")

    # ── Head variant ─────────────────────────────────────────────────────────
    parser.add_argument("--head", default="mlp",
                        choices=["linear", "mlp", "mlp_no_meta"])

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--patience",       type=int,   default=10)
    parser.add_argument("--early_stopping_metric", default="val_bal_acc",
                        choices=["val_loss", "val_bal_acc"])
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--dropout",        type=float, default=0.3)
    parser.add_argument("--meta_embed_dim", type=int,   default=16)
    parser.add_argument("--hidden_dims",    type=str,   nargs="+", default=[256, 128])
    parser.add_argument("--weight_decay",   type=float, default=0.01)

    # ── Loss function ─────────────────────────────────────────────────────────
    parser.add_argument("--loss", default="ce",
                        choices=["ce", "wce", "ce_smooth", "focal", "cb_focal", "ldam", "balanced_softmax"])
    parser.add_argument("--class_weighting", default="sqrt",
                        choices=["none", "inverse", "sqrt", "effective"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--focal_gamma",     type=float, default=2.0)
    parser.add_argument("--cb_beta",         type=float, default=0.99)
    parser.add_argument("--ldam_max_margin", type=float, default=0.5)
    parser.add_argument("--ldam_scale",      type=float, default=30.0)

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--run_name",      default=None,
                        help="Output-dir name under <out_dir>/biomedclip_img_text_downstream/ "
                             "(default: auto-generated from head/loss/timestamp).")
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="biomedclip-img-text-downstream")
    parser.add_argument("--wandb_run",     default=None)
    parser.add_argument("--wandb_entity",  default=None)
    parser.add_argument("--sweep",         action="store_true")

    args = parser.parse_args(argv)
    raw = " ".join(str(x) for x in args.hidden_dims)
    args.hidden_dims = [int(x) for x in raw.strip("[]").replace(",", " ").split()]
    return args


def _apply_sweep_config(args: argparse.Namespace) -> None:
    cfg = wandb.config
    for key in (
        "lr", "dropout", "meta_embed_dim", "weight_decay", "loss",
        "class_weighting", "label_smoothing", "focal_gamma", "cb_beta",
        "ldam_max_margin", "ldam_scale", "batch_size", "head",
        "early_stopping_metric",
    ):
        if key in cfg:
            setattr(args, key, cfg[key])
    if "hidden_dims" in cfg:
        args.hidden_dims = list(cfg["hidden_dims"])


def main(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Head: {args.head}  |  Early-stopping: {args.early_stopping_metric}")

    use_wandb = (args.wandb or args.sweep) and WANDB_AVAILABLE
    if (args.wandb or args.sweep) and not WANDB_AVAILABLE:
        warnings.warn("--wandb/--sweep set but wandb is not installed. Skipping.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            config=dict(vars(args).items()),
        )
        if args.sweep:
            _apply_sweep_config(args)

    # ── Label mapping ─────────────────────────────────────────────────────────
    label_to_idx, idx_to_label, num_classes = resolve_label_maps(args.binary)
    if args.binary:
        print("Mode: binary (benign vs malignant)")

    # ── Load BiomedCLIP ───────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_TAG}")
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)
    tokenizer = open_clip.get_tokenizer(MODEL_TAG)

    if args.freezed_biomedclip:
        print("Using vanilla BiomedCLIP (no checkpoint, no LoRA).")
    else:
        ckpt     = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        lora_cfg = ckpt.get("lora_config") or {}
        if not lora_cfg:
            raise RuntimeError(
                f"Checkpoint '{args.checkpoint}' has no 'lora_config'. "
                "Re-run pretraining with the current biomedclip_pretrain.py."
            )
        if lora_cfg.get("no_lora"):
            print(f"Partial fine-tune checkpoint (unfreeze_blocks={lora_cfg['unfreeze_blocks']}, no LoRA).")
        else:
            inject_lora(model, lora_cfg["lora_layers"], lora_cfg["lora_r"], lora_cfg["lora_alpha"])
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    model = model.to(device)
    model.requires_grad_(False)
    model.eval()
    print(f"Encoder frozen. Embedding dim: {EMBED_DIM} (512 image + 512 text)")

    # ── Data ──────────────────────────────────────────────────────────────────
    with open(args.splits, encoding="utf-8") as fh:
        raw = json.load(fh)
    splits = {"train": raw["train"], "val": raw["val"], "test": raw["test"]}

    all_samples    = splits["train"] + splits["val"] + splits["test"]
    age_sex_lookup = {Path(s["image"]).stem: (float(s["age"]), float(s["sex"])) for s in all_samples}
    train_ages     = [s["age"] for s in splits["train"]]
    age_mean = float(np.mean(train_ages))
    age_std  = float(np.std(train_ages))
    print(f"Age stats (train): mean={age_mean:.1f}, std={age_std:.1f}")

    preprocess_train = build_train_transform(preprocess_val)
    val_kwargs   = dict(age_sex_lookup=age_sex_lookup, age_mean=age_mean, age_std=age_std,
                        preprocess=preprocess_val,   tokenizer=tokenizer, use_mask=args.use_mask,
                        label_to_idx=label_to_idx)
    train_kwargs = dict(age_sex_lookup=age_sex_lookup, age_mean=age_mean, age_std=age_std,
                        preprocess=preprocess_train, tokenizer=tokenizer, use_mask=args.use_mask,
                        label_to_idx=label_to_idx)

    train_ds = DownstreamDatasetWithText(splits["train"], **train_kwargs)
    val_ds   = DownstreamDatasetWithText(splits["val"],   **val_kwargs)
    test_ds  = DownstreamDatasetWithText(splits["test"],  **val_kwargs)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    use_pin       = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader_raw   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── Pre-compute val (and test if --eval_test) embeddings (train is embedded
    # on-the-fly for augmentation) ──────────────────────────────────────────────
    print("Pre-computing val embeddings...")
    val_emb, val_age, val_sex, val_lbl = extract_img_text_embeddings(model, val_loader_raw, device)
    print(f"Embedding dim: {val_emb.shape[1]}")
    val_loader = DataLoader(EmbeddingDataset(val_emb, val_age, val_sex, val_lbl),
                            batch_size=args.batch_size, shuffle=False)

    if args.eval_test:
        test_loader_raw = DataLoader(test_ds, shuffle=False, **loader_kwargs)
        print("Pre-computing test embeddings...")
        test_emb, test_age, test_sex, test_lbl = extract_img_text_embeddings(
            model, test_loader_raw, device)
        test_loader = DataLoader(EmbeddingDataset(test_emb, test_age, test_sex, test_lbl),
                                 batch_size=args.batch_size, shuffle=False)

    # ── Class weighting ───────────────────────────────────────────────────────
    train_labels_all = torch.tensor(
        [label_to_idx[s["label"]] for s in train_ds.samples], dtype=torch.long
    )
    label_counts  = torch.bincount(train_labels_all, minlength=num_classes).float()
    class_weights = compute_class_weights(
        label_counts, num_classes=num_classes, mode=args.class_weighting,
        beta=args.cb_beta, device=device,
    )
    print(f"Class counts (train): { {idx_to_label[i]: int(label_counts[i]) for i in range(num_classes)} }")
    print(f"Loss: {args.loss} | class_weighting={args.class_weighting}")

    # ── Head, loss, optimiser ─────────────────────────────────────────────────
    head      = build_head(args, EMBED_DIM, device, num_classes)
    criterion = build_classification_loss(args, label_counts, num_classes, class_weights, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = args.run_name or (
        f"run_{args.head}_{args.loss}_{args.class_weighting}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir  = Path(args.out_dir) / "biomedclip_img_text_downstream" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best_head.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    maximize_metric  = args.early_stopping_metric == "val_bal_acc"
    best_metric      = -float("inf") if maximize_metric else float("inf")
    patience_counter = 0
    # Val metrics of the checkpoint actually saved/restored below (set inside
    # `if improved:`), not a running max/last-epoch value.
    best_val_bal_acc = best_val_acc = best_val_f1_macro = best_val_auroc = None
    print(f"\nTraining {args.head} head for {args.epochs} epochs "
          f"(patience={args.patience}, monitor={args.early_stopping_metric})\n")

    for epoch in range(1, args.epochs + 1):
        train_loss                                              = train_one_epoch_aug(head, model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(head, val_loader, criterion, device)
        val_bal_acc       = balanced_accuracy_score(val_labels, val_preds)
        val_f1_macro      = f1_score(val_labels, val_preds, average="macro")
        val_auroc         = compute_auroc(val_labels, val_probs, num_classes)
        val_weighted_prec, val_weighted_rec, _, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="weighted", zero_division=0
        )
        val_combined_acc = 0.5 * val_acc + 0.5 * val_bal_acc

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | val_bal_acc={val_bal_acc:.3f} | val_combined_acc={val_combined_acc:.3f} | "
            f"val_f1_macro={val_f1_macro:.3f} | val_auroc={val_auroc:.3f}"
        )
        if use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "val/loss":   val_loss,
                "val/acc":    val_acc,
                "val/balanced_acc":        val_bal_acc,
                "val/precision_weighted":  val_weighted_prec,
                "val/recall_weighted":     val_weighted_rec,
                "val/combined_acc":        val_combined_acc,
                "val/f1_macro":            val_f1_macro,
                "val/auroc":               val_auroc,
            }, step=epoch)

        current_metric = val_bal_acc if maximize_metric else val_loss
        improved = current_metric > best_metric if maximize_metric else current_metric < best_metric
        if improved:
            best_metric      = current_metric
            patience_counter = 0
            best_val_bal_acc  = val_bal_acc
            best_val_acc      = val_acc
            best_val_f1_macro = val_f1_macro
            best_val_auroc    = val_auroc
            if use_wandb:
                wandb.run.summary["val/best_balanced_acc"] = val_bal_acc
                wandb.run.summary["val/best_acc"]          = val_acc
                wandb.run.summary["val/best_f1_macro"]     = val_f1_macro
                wandb.run.summary["val/best_auroc"]        = val_auroc
            torch.save({
                "epoch": epoch,
                "head_state_dict": head.state_dict(),
                "val_loss":    val_loss,
                "val_bal_acc": val_bal_acc,
                "head":        args.head,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    # ── Restore best head ─────────────────────────────────────────────────────
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    head.load_state_dict(best_ckpt["head_state_dict"])

    # ── Held-out test evaluation (opt-in via --eval_test; no BTXRD — see module
    # docstring) ─────────────────────────────────────────────────────────────
    eval_metrics: dict = {
        "val/balanced_acc": best_val_bal_acc,
        "val/acc":          best_val_acc,
        "val/f1_macro":     best_val_f1_macro,
        "val/auroc":        best_val_auroc,
    }
    if args.eval_test:
        test_loss, _, test_preds, test_labels, test_probs = evaluate(head, test_loader, criterion, device)
        eval_metrics.update(report_eval(
            "TEST", test_preds, test_labels, test_loss, idx_to_label, num_classes,
            prefix="test", probs=test_probs, use_wandb=use_wandb))
        eval_metrics["test/_preds"]  = test_preds.tolist()
        eval_metrics["test/_labels"] = test_labels.tolist()
        eval_metrics["test/_probs"]  = test_probs.tolist()
    else:
        print("\nSkipping held-out test evaluation (--eval_test not set).")

    if use_wandb:
        if eval_metrics:
            safe_wandb_log(eval_metrics)
        wandb.finish()

    print(f"\nBest {args.early_stopping_metric}: {best_metric:.4f}")
    print(f"Checkpoints saved to: {run_dir}")
    return eval_metrics


if __name__ == "__main__":
    main(parse_args())
