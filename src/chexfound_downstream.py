"""CheXFound supervised downstream malignancy classifier.

Loads the CheXFound ViT-L/16 encoder (continued-pretrain or frozen original
weights) and trains a classification head on top.

Head variants (--head):
    linear       — single nn.Linear(1024, 3), no metadata
    mlp          — MLP with age/sex late fusion  (default)
    mlp_no_meta  — same MLP capacity, no clinical metadata

Usage (continued-pretrain checkpoint):
    python src/chexfound_downstream.py \\
        --checkpoint results/chexfound_pretrain/.../checkpoint_last.pth \\
        --splits     results/biomedclip_pretrain/.../splits.json \\
        --excel      data/internal_dataset/metadata.xlsx

Usage (frozen original weights):
    python src/chexfound_downstream.py \\
        --checkpoint none \\
        --chexfound_weights /path/to/chexfound_vitl16.pth \\
        --splits ... --excel ...
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, classification_report, confusion_matrix,
                              f1_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.utils.misc import ROOT_DIR, DEFAULT_OUT_DIR, DEFAULT_SPLITS
from biomedclip.data.datasets import (
    DownstreamDataset, EmbeddingDataset, LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES,
    LABEL_TO_IDX_BINARY, IDX_TO_LABEL_BINARY, NUM_CLASSES_BINARY,
)
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.classifier import LinearHead, MalignancyMLP, extract_embeddings
from biomedclip.utils.downstream_eval import (
    require_binary_for_btxrd, load_btxrd_samples, build_downstream_loader, report_eval,
)

_CHEXFOUND_DATA = ROOT_DIR / "src" / "chexfound" / "data"
DEFAULT_CHEXFOUND_CONFIG     = _CHEXFOUND_DATA / "config.yaml"
DEFAULT_CHEXFOUND_CHECKPOINT = _CHEXFOUND_DATA / "teacher_checkpoint.pth"

EMBED_DIM = 1024  # CheXFound ViT-L/16


def build_encoder(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, object, object]:
    from chexfound.data.transforms import (
        build_preprocess_val_chexfound,
        build_train_transform_chexfound,
    )
    from chexfound.models.encoders import CheXFoundViT, load_continued_pretrain_weights

    preprocess_val   = build_preprocess_val_chexfound(image_size=args.image_size)
    preprocess_train = build_train_transform_chexfound(preprocess_val)

    checkpoint = None if args.checkpoint.lower() == "none" else args.checkpoint

    if checkpoint is None:
        if not args.chexfound_weights:
            raise ValueError(
                "--chexfound_weights is required when --checkpoint none. "
                "Provide the path to the original CheXFound .pth file."
            )
        print("Loading frozen CheXFound baseline (original pretrained weights).")
        encoder = CheXFoundViT(
            args.chexfound_config, args.chexfound_weights,
            lora_layers=0, r=8, alpha=16.0, load_pretrained=True,
        )
    else:
        print(f"Loading CheXFound from continued-pretrain checkpoint: {checkpoint}")
        encoder = CheXFoundViT(
            args.chexfound_config, weights_path=None,
            lora_layers=0, r=8, alpha=16.0, load_pretrained=False,
        )
        load_continued_pretrain_weights(encoder, checkpoint)

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder, preprocess_train, preprocess_val


def build_head(args: argparse.Namespace, embed_dim: int, device: torch.device,
               num_classes: int = NUM_CLASSES) -> nn.Module:
    if args.head == "linear":
        return LinearHead(embed_dim, num_classes=num_classes).to(device)
    elif args.head == "mlp_no_meta":
        return MalignancyMLP(embed_dim, args.hidden_dims, args.dropout,
                             args.meta_embed_dim, use_meta=False, num_classes=num_classes).to(device)
    else:  # mlp
        return MalignancyMLP(embed_dim, args.hidden_dims, args.dropout,
                             args.meta_embed_dim, use_meta=True, num_classes=num_classes).to(device)


def train_one_epoch(
    head: nn.Module,
    encoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    head.train()
    encoder.eval()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        age    = batch["age"].to(device)
        sex    = batch["sex"].to(device)
        lbl    = batch["label"].to(device)

        with torch.no_grad():
            emb = F.normalize(encoder(images), dim=-1)

        optimizer.zero_grad()
        logits = head(emb, age, sex)
        loss   = criterion(logits, lbl)
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
) -> tuple[float, float, np.ndarray, np.ndarray]:
    head.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for emb, age, sex, lbl in loader:
        emb, age, sex, lbl = emb.to(device), age.to(device), sex.to(device), lbl.to(device)
        logits = head(emb, age, sex)
        total_loss += criterion(logits, lbl).item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(lbl.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / len(loader), float((preds == labels).mean()), preds, labels


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CheXFound downstream malignancy classifier."
    )
    # ── CheXFound encoder ────────────────────────────────────────────────────
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHEXFOUND_CHECKPOINT),
                        help="Continued-pretrain checkpoint. Pass 'none' for frozen baseline mode.")
    parser.add_argument("--chexfound_config",  default=str(DEFAULT_CHEXFOUND_CONFIG))
    parser.add_argument("--chexfound_weights", default=None,
                        help="Original CheXFound .pth checkpoint (frozen baseline mode only).")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--image_size", type=int, default=512,
                        help="Input image resolution. 512 = native CheXFound resolution; "
                             "224 = same as BiomedCLIP (uses pos-embedding interpolation).")
    parser.add_argument("--binary", type=lambda x: str(x).lower() in ("true", "1", "yes"),
                        default=False,
                        help="Binary mode: benign vs malignant only (intermediate cases skipped).")
    parser.add_argument("--eval_test", action="store_true",
                        help="Evaluate the split's held-out test set after training. Off by "
                             "default so hyperparameter sweeps never touch test/BTXRD.")
    parser.add_argument("--btxrd_manifest", default=None,
                        help="Path to a BTXRD downstream manifest to also score as an external "
                             "test set. Requires --eval_test and --binary.")
    parser.add_argument("--splits",   default=str(DEFAULT_SPLITS))
    parser.add_argument("--out_dir",  default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--use_mask", action="store_true")

    # ── Head variant ─────────────────────────────────────────────────────────
    parser.add_argument("--head", default="mlp",
                        choices=["linear", "mlp", "mlp_no_meta"])

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--patience",       type=int,   default=10)
    parser.add_argument("--early_stopping_metric", default="val_loss",
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
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="chexfound-downstream")
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


def main(args: argparse.Namespace) -> None:
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
    require_binary_for_btxrd(args.binary, args.btxrd_manifest)
    if args.binary:
        label_to_idx = LABEL_TO_IDX_BINARY
        idx_to_label = IDX_TO_LABEL_BINARY
        num_classes  = NUM_CLASSES_BINARY
        print("Mode: binary (benign vs malignant)")
    else:
        label_to_idx = LABEL_TO_IDX
        idx_to_label = IDX_TO_LABEL
        num_classes  = NUM_CLASSES

    # ── Encoder ───────────────────────────────────────────────────────────────
    encoder, preprocess_train, preprocess_val = build_encoder(args, device)
    print(f"CheXFound encoder loaded ({EMBED_DIM}-dim), frozen")

    # ── Data ──────────────────────────────────────────────────────────────────
    with open(args.splits, encoding="utf-8") as fh:
        raw = json.load(fh)
    splits = {
        "train": raw["train"],
        "val":   raw["val"],
        "test":  raw["test"],
    }

    all_samples    = splits["train"] + splits["val"] + splits["test"]
    age_sex_lookup = {Path(s["image"]).stem: (float(s["age"]), float(s["sex"])) for s in all_samples}
    train_ages     = [s["age"] for s in splits["train"]]
    age_mean = float(np.mean(train_ages))
    age_std  = float(np.std(train_ages))
    print(f"Age stats (train): mean={age_mean:.1f}, std={age_std:.1f}")

    train_ds = DownstreamDataset(splits["train"], age_sex_lookup, age_mean, age_std,
                                  preprocess_train, args.use_mask, label_to_idx=label_to_idx)
    val_ds   = DownstreamDataset(splits["val"],   age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   args.use_mask, label_to_idx=label_to_idx)
    test_ds  = DownstreamDataset(splits["test"],  age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   args.use_mask, label_to_idx=label_to_idx)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    use_pin       = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}

    train_loader    = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader_raw  = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader_raw = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print("Pre-computing val embeddings...")
    val_emb,  val_age,  val_sex,  val_lbl  = extract_embeddings(encoder, val_loader_raw,  device)

    embed_dim = val_emb.shape[1]
    print(f"Embedding dim: {embed_dim}")

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

    val_emb_ds  = EmbeddingDataset(val_emb,  val_age,  val_sex,  val_lbl)
    val_loader  = DataLoader(val_emb_ds,  batch_size=args.batch_size, shuffle=False)
    if args.eval_test:
        print("Pre-computing test embeddings...")
        test_emb, test_age, test_sex, test_lbl = extract_embeddings(encoder, test_loader_raw, device)
        test_emb_ds = EmbeddingDataset(test_emb, test_age, test_sex, test_lbl)
        test_loader = DataLoader(test_emb_ds, batch_size=args.batch_size, shuffle=False)

    # ── Head, loss, optimiser ─────────────────────────────────────────────────
    head      = build_head(args, embed_dim, device, num_classes=num_classes)
    criterion = build_classification_loss(args, label_counts, num_classes, class_weights, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir) / "chexfound_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id    = (wandb.run.id if use_wandb and wandb.run else None) or "local"
    ckpt_path = out_dir / f"best_head_{run_id}.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    maximize_metric  = args.early_stopping_metric == "val_bal_acc"
    best_metric      = -float("inf") if maximize_metric else float("inf")
    patience_counter = 0
    print(f"\nTraining {args.head} head for {args.epochs} epochs "
          f"(patience={args.patience}, monitor={args.early_stopping_metric})\n")

    for epoch in range(1, args.epochs + 1):
        train_loss                               = train_one_epoch(head, encoder, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(head, val_loader, criterion, device)
        val_bal_acc       = balanced_accuracy_score(val_labels, val_preds)
        val_weighted_prec, val_weighted_rec, _, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="weighted", zero_division=0
        )
        val_combined_acc = 0.5 * val_acc + 0.5 * val_bal_acc

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | val_bal_acc={val_bal_acc:.3f} | val_combined_acc={val_combined_acc:.3f}"
        )
        if use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/balanced_acc":        val_bal_acc,
                "val/precision_weighted":  val_weighted_prec,
                "val/recall_weighted":     val_weighted_rec,
                "val/combined_acc":        val_combined_acc,
            }, step=epoch)

        current_metric = val_bal_acc if maximize_metric else val_loss
        improved = current_metric > best_metric if maximize_metric else current_metric < best_metric
        if improved:
            best_metric      = current_metric
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "head_state_dict": head.state_dict(),
                "val_loss": val_loss,
                "val_bal_acc": val_bal_acc,
                "head": args.head,
                "args": vars(args),
                "age_mean": age_mean,
                "age_std": age_std,
                "embed_dim": embed_dim,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    # ── Restore best head ─────────────────────────────────────────────────────
    head.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["head_state_dict"]
    )

    # ── Held-out test / external BTXRD evaluation (opt-in via --eval_test) ─────
    eval_metrics: dict = {}
    if args.eval_test:
        test_loss, _, test_preds, test_labels = evaluate(head, test_loader, criterion, device)
        eval_metrics.update(report_eval(
            "TEST", test_preds, test_labels, test_loss, idx_to_label, num_classes, prefix="test"))

        if args.btxrd_manifest:
            print(f"Pre-computing BTXRD embeddings from {args.btxrd_manifest}...")
            btxrd_samples = load_btxrd_samples(args.btxrd_manifest)
            btxrd_raw, n_btxrd = build_downstream_loader(
                btxrd_samples, age_mean, age_std, preprocess_val, args.use_mask,
                args.batch_size, label_to_idx, device)
            print(f"BTXRD samples: {n_btxrd}")
            b_emb, b_age, b_sex, b_lbl = extract_embeddings(encoder, btxrd_raw, device)
            btxrd_loader = DataLoader(
                EmbeddingDataset(b_emb, b_age, b_sex, b_lbl),
                batch_size=args.batch_size, shuffle=False)
            btxrd_loss, _, btxrd_preds, btxrd_labels = evaluate(head, btxrd_loader, criterion, device)
            eval_metrics.update(report_eval(
                "BTXRD (external)", btxrd_preds, btxrd_labels, btxrd_loss,
                idx_to_label, num_classes, prefix="btxrd"))
    else:
        print("\nSkipping held-out test / BTXRD evaluation (--eval_test not set).")

    if use_wandb:
        if eval_metrics:
            wandb.log(eval_metrics)
        wandb.finish()

    print(f"\nBest {args.early_stopping_metric}: {best_metric:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main(parse_args())
