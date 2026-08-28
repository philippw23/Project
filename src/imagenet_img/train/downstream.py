"""End-to-end supervised downstream classifier — ImageNet pretrained ViT-B/16, frozen.

The encoder (ViT-B/16, ImageNet-1k weights) is kept frozen throughout training.
Only the MalignancyMLP head is trained (linear probing).

Val and test embeddings are pre-computed once before the training loop.
Train embeddings are computed each batch because random augmentations differ per epoch.

Usage:
    python src/imagenet_img_downstream.py \\
        --splits     data/internal_dataset/split.json \\
        --out_dir    results/imagenet_img \\
        --epochs     100 \\
        --lr_mlp     3e-4
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.data.datasets import (DownstreamDataset, EmbeddingDataset)
from biomedclip.data.splits import build_stratified_splits
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.classifier import MalignancyMLP, extract_embeddings
from biomedclip.utils.misc import DEFAULT_DATASET_JSON, DEFAULT_OUT_DIR
from biomedclip.utils.downstream_eval import (
    resolve_label_maps, require_binary_for_btxrd, load_btxrd_samples,
    build_downstream_loader, report_eval,
)
from imagenet_img.data.transforms import build_train_transform, build_val_transform
from imagenet_img.models.encoders import build_encoder


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_one_epoch(
    encoder: nn.Module,
    mlp: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    mlp.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        age    = batch["age"].to(device)
        sex    = batch["sex"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            emb = F.normalize(encoder(images), dim=-1)

        logits = mlp(emb, age, sex)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_cached(
    mlp: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate on pre-computed embeddings (no encoder forward pass)."""
    mlp.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for emb, age, sex, labels in loader:
        emb    = emb.to(device)
        age    = age.to(device)
        sex    = sex.to(device)
        labels = labels.to(device)

        logits = mlp(emb, age, sex)
        total_loss += criterion(logits, labels).item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc    = float((preds == labels).mean())
    return total_loss / len(loader), acc, preds, labels


# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageNet pretrained ViT-B/16 downstream classifier (linear probing)."
    )

    parser.add_argument("--head", default="mlp", choices=["mlp", "mlp_no_meta"],
                        help="mlp: image + age/sex fusion (default);  mlp_no_meta: image only")

    parser.add_argument("--splits",  default=None,
                        help="Path to a pre-existing split.json. If omitted, splits are generated.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_JSON),
                        help="Path to dataset_full.json (default: %(default)s)")
    parser.add_argument("--downstream_train_frac", type=float, default=0.8)
    parser.add_argument("--downstream_val_frac",   type=float, default=0.1)
    parser.add_argument("--test_frac",             type=float, default=0.1)
    parser.add_argument("--out_dir",  default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--use_mask", action="store_true",
                        help="Crop images around the lesion mask before feeding to the encoder.")
    parser.add_argument("--binary", type=lambda x: str(x).lower() in ("true", "1", "yes"),
                        default=False,
                        help="Binary classification (benign vs malignant). "
                             "Use with split_binary.json — intermediate cases must already be excluded.")
    parser.add_argument("--eval_test", action="store_true",
                        help="Evaluate the split's held-out test set after training. Off by "
                             "default so hyperparameter sweeps never touch test/BTXRD.")
    parser.add_argument("--btxrd_manifest", default=None,
                        help="Path to a BTXRD downstream manifest to also score as an external "
                             "test set. Requires --eval_test and --binary.")

    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--early_stopping_metric", default="val_bal_acc",
                        choices=["val_loss", "val_bal_acc"],
                        help="Metric to monitor for early stopping and best-checkpoint saving.")
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr_mlp",        type=float, default=3e-4)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--meta_embed_dim", type=int,  default=16)
    parser.add_argument("--hidden_dims",   type=str,   nargs="+", default=["128"])

    parser.add_argument("--loss", default="ce",
                        choices=["ce", "wce", "focal", "cb_focal", "balanced_softmax", "ldam"])
    parser.add_argument("--class_weighting", default="sqrt",
                        choices=["none", "inverse", "sqrt", "effective"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--cb_beta",     type=float, default=0.99)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--ldam_max_margin", type=float, default=0.5)
    parser.add_argument("--ldam_scale",      type=float, default=30.0)

    parser.add_argument("--overfit_n",     type=int, default=0,
                        help="If > 0, truncate all splits to N samples (overfit sanity check).")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--run_name",      default=None,
                        help="Output-dir name under --out_dir "
                             "(default: auto-generated from timestamp/head/loss).")
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="imagenet-img-downstream")
    parser.add_argument("--wandb_run",     default=None)
    parser.add_argument("--wandb_entity",  default=None)
    parser.add_argument("--sweep",         action="store_true")

    args = parser.parse_args(argv)

    raw = " ".join(str(x) for x in args.hidden_dims)
    args.hidden_dims = [int(x) for x in raw.strip("[]").replace(",", " ").split()]

    return args


def _apply_sweep_config(args: argparse.Namespace) -> None:
    cfg = wandb.config
    sweep_keys = (
        "head", "lr_mlp", "weight_decay", "dropout", "meta_embed_dim",
        "batch_size", "loss", "class_weighting", "focal_gamma", "cb_beta",
    )
    for key in sweep_keys:
        if key in cfg:
            setattr(args, key, cfg[key])
    if "hidden_dims" in cfg:
        args.hidden_dims = list(cfg["hidden_dims"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Encoder: ViT-B/16 (ImageNet pretrained, frozen)")

    use_wandb = (args.wandb or args.sweep) and WANDB_AVAILABLE
    if (args.wandb or args.sweep) and not WANDB_AVAILABLE:
        warnings.warn("--wandb/--sweep set but wandb is not installed.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            config=vars(args),
        )
        if args.sweep:
            _apply_sweep_config(args)

    # ── Label mapping ─────────────────────────────────────────────────────────
    require_binary_for_btxrd(args.binary, args.btxrd_manifest)
    label_to_idx, idx_to_label, num_classes = resolve_label_maps(args.binary)
    if args.binary:
        print("Mode: binary (benign vs malignant)")

    # ── Encoder + MLP ─────────────────────────────────────────────────────────
    encoder, embed_dim = build_encoder()
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    use_meta = args.head != "mlp_no_meta"
    mlp = MalignancyMLP(embed_dim, args.hidden_dims, args.dropout, args.meta_embed_dim,
                        use_meta=use_meta, num_classes=num_classes).to(device)
    print(f"Encoder: ViT-B/16 ({embed_dim}-dim, frozen) | Head: {args.head} | "
          f"MLP params: {sum(p.numel() for p in mlp.parameters()):,}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.splits is not None:
        with open(args.splits, encoding="utf-8") as fh:
            raw = json.load(fh)
        splits = {"train": raw["train"], "val": raw["val"], "test": raw["test"]}
    else:
        train, val, test = build_stratified_splits(args, run_dir=Path(args.out_dir))
        splits = {"train": train, "val": val, "test": test}

    if args.overfit_n > 0:
        subset = splits["train"][:args.overfit_n]
        splits["train"] = subset
        splits["val"]   = subset
        splits["test"]  = subset

    all_samples = splits["train"] + splits["val"] + splits["test"]
    age_sex_lookup = {
        Path(s["image"]).stem: (float(s["age"]), float(s["sex"]))
        for s in all_samples
    }

    train_ages = [s["age"] for s in splits["train"]]
    age_mean = float(np.mean(train_ages))
    age_std  = float(np.std(train_ages))

    preprocess_train = build_train_transform()
    preprocess_val   = build_val_transform()

    train_ds = DownstreamDataset(splits["train"], age_sex_lookup, age_mean, age_std,
                                  preprocess_train, use_mask=args.use_mask, label_to_idx=label_to_idx)
    val_ds   = DownstreamDataset(splits["val"],   age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   use_mask=args.use_mask, label_to_idx=label_to_idx)
    test_ds  = DownstreamDataset(splits["test"],  age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   use_mask=args.use_mask, label_to_idx=label_to_idx)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    use_pin       = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    emb_loader_kwargs = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": use_pin}

    # ── Pre-compute val (and test if --eval_test) embeddings ──────────────────
    print("Pre-computing val embeddings...")
    val_emb,  val_age,  val_sex,  val_lbl  = extract_embeddings(encoder, val_loader,  device)
    val_emb_loader  = DataLoader(EmbeddingDataset(val_emb,  val_age,  val_sex,  val_lbl),
                                 shuffle=False, **emb_loader_kwargs)
    if args.eval_test:
        print("Pre-computing test embeddings...")
        test_emb, test_age, test_sex, test_lbl = extract_embeddings(encoder, test_loader, device)
        test_emb_loader = DataLoader(EmbeddingDataset(test_emb, test_age, test_sex, test_lbl),
                                     shuffle=False, **emb_loader_kwargs)

    # ── Class weighting ───────────────────────────────────────────────────────
    train_labels_all = torch.tensor(
        [label_to_idx[s["label"]] for s in train_ds.samples], dtype=torch.long
    )
    label_counts  = torch.bincount(train_labels_all, minlength=num_classes).float()
    class_weights = compute_class_weights(label_counts, num_classes, args.class_weighting,
                                           args.cb_beta, device)
    criterion = build_classification_loss(args, label_counts, num_classes, class_weights, device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr_mlp,
                                   weight_decay=args.weight_decay)

    # ── Output dir ────────────────────────────────────────────────────────────
    if args.run_name:
        run_tag = args.run_name
    else:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = f"{ts}_{args.head}_{args.loss}"
        if use_wandb and wandb.run:
            run_tag = f"{run_tag}_{wandb.run.id}"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_checkpoint.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    maximize_metric  = args.early_stopping_metric == "val_bal_acc"
    best_metric      = -float("inf") if maximize_metric else float("inf")
    patience_counter = 0
    print(f"\nTraining for {args.epochs} epochs "
          f"(patience={args.patience}, monitor={args.early_stopping_metric})\n")

    for epoch in range(1, args.epochs + 1):
        train_loss                               = train_one_epoch(encoder, mlp, train_loader,
                                                                    optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate_cached(mlp, val_emb_loader,
                                                                     criterion, device)

        val_bal_acc       = balanced_accuracy_score(val_labels, val_preds)
        val_weighted_prec, val_weighted_rec, _, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="weighted", zero_division=0
        )
        val_combined_acc = 0.5 * val_acc + 0.5 * val_bal_acc

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | val_bal_acc={val_bal_acc:.3f} | "
            f"val_combined_acc={val_combined_acc:.3f}"
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
                "mlp_state_dict": mlp.state_dict(),
                "val_loss": val_loss,
                "val_bal_acc": val_bal_acc,
                "args": vars(args),
                "age_mean": age_mean,
                "age_std": age_std,
                "embed_dim": embed_dim,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ── Restore best head ─────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mlp.load_state_dict(ckpt["mlp_state_dict"])

    # ── Held-out test / external BTXRD evaluation (opt-in via --eval_test) ─────
    eval_metrics: dict = {}
    if args.eval_test:
        test_loss, _, test_preds, test_labels = evaluate_cached(mlp, test_emb_loader,
                                                                 criterion, device)
        eval_metrics.update(report_eval(
            "TEST", test_preds, test_labels, test_loss, idx_to_label, num_classes, prefix="test"))
        eval_metrics["test/_preds"]  = test_preds.tolist()
        eval_metrics["test/_labels"] = test_labels.tolist()

        if args.btxrd_manifest:
            print(f"Pre-computing BTXRD embeddings from {args.btxrd_manifest}...")
            btxrd_samples = load_btxrd_samples(args.btxrd_manifest)
            btxrd_raw, n_btxrd = build_downstream_loader(
                btxrd_samples, age_mean, age_std, preprocess_val, args.use_mask,
                args.batch_size, label_to_idx, device)
            print(f"BTXRD samples: {n_btxrd}")
            b_emb, b_age, b_sex, b_lbl = extract_embeddings(encoder, btxrd_raw, device)
            btxrd_emb_loader = DataLoader(EmbeddingDataset(b_emb, b_age, b_sex, b_lbl),
                                          shuffle=False, **emb_loader_kwargs)
            btxrd_loss, _, btxrd_preds, btxrd_labels = evaluate_cached(
                mlp, btxrd_emb_loader, criterion, device)
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
    print(f"Checkpoint saved to: {ckpt_path}")
    return eval_metrics
