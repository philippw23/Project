"""End-to-end supervised downstream classifier — image encoder trained from scratch.

Two encoder variants (--encoder resnet18 | vit_tiny) are both initialised with
random weights and trained jointly with the MalignancyMLP head via cross-entropy.

LR schedule (two param groups):
  encoder  — cosine decay with linear warmup
  MLP head — flat LR throughout

Usage:
    python src/scratch_img_downstream.py \\
        --encoder    resnet18 \\
        --splits     results/.../splits.json \\
        --excel      data/internal_dataset/metadata.xlsx \\
        --out_dir    results \\
        --epochs     100 \\
        --lr_encoder 1e-3 \\
        --lr_mlp     1e-3
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              precision_recall_fscore_support)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.data.datasets import (DownstreamDataset, IDX_TO_LABEL,
                                       LABEL_TO_IDX, NUM_CLASSES)
from biomedclip.data.splits import load_age_sex_lookup
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.classifier import MalignancyMLP
from biomedclip.utils.misc import DEFAULT_EXCEL, DEFAULT_OUT_DIR
from scratch_img.data.transforms import build_train_transform, build_val_transform
from scratch_img.models.encoders import build_encoder


# ── LR schedule ──────────────────────────────────────────────────────────────

def _lr_factor(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_one_epoch(
    encoder: nn.Module,
    mlp: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    encoder.train()
    mlp.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        age    = batch["age"].to(device)
        sex    = batch["sex"].to(device)
        labels = batch["label"].to(device)

        emb    = F.normalize(encoder(images), dim=-1)
        logits = mlp(emb, age, sex)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    mlp: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    encoder.eval()
    mlp.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        age    = batch["age"].to(device)
        sex    = batch["sex"].to(device)
        labels = batch["label"].to(device)

        emb    = F.normalize(encoder(images), dim=-1)
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
        description="Scratch image-only downstream classifier (ResNet-18 / ViT-Tiny)."
    )

    parser.add_argument("--encoder", required=True, choices=["resnet18", "vit_tiny"])

    parser.add_argument("--splits",   required=True)
    parser.add_argument("--excel",    default=str(DEFAULT_EXCEL))
    parser.add_argument("--out_dir",  default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--use_mask", action="store_true",
                        help="Crop images around the lesion mask before feeding to the encoder.")

    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr_encoder",    type=float, default=1e-3)
    parser.add_argument("--lr_mlp",        type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int,   default=10)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--meta_embed_dim", type=int,  default=16)
    parser.add_argument("--hidden_dims",   type=str,   nargs="+", default=["128"])

    parser.add_argument("--loss", default="ce",
                        choices=["ce", "wce", "focal", "cb_focal", "balanced_softmax"])
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
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="scratch-img-downstream")
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
        "encoder", "lr_encoder", "lr_mlp", "warmup_epochs", "weight_decay",
        "dropout", "meta_embed_dim", "batch_size", "loss", "class_weighting",
        "focal_gamma", "cb_beta",
    )
    for key in sweep_keys:
        if key in cfg:
            setattr(args, key, cfg[key])
    if "hidden_dims" in cfg:
        args.hidden_dims = list(cfg["hidden_dims"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Encoder: {args.encoder}")

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

    # ── Encoder + MLP ─────────────────────────────────────────────────────────
    encoder, embed_dim = build_encoder(args.encoder)
    encoder = encoder.to(device)
    mlp = MalignancyMLP(embed_dim, args.hidden_dims, args.dropout, args.meta_embed_dim).to(device)
    print(f"Encoder: {args.encoder} ({embed_dim}-dim) | "
          f"Params: {sum(p.numel() for p in encoder.parameters()):,}")

    # ── Data ──────────────────────────────────────────────────────────────────
    with open(args.splits, encoding="utf-8") as fh:
        splits = json.load(fh)

    if args.overfit_n > 0:
        subset = splits["downstream_train"][:args.overfit_n]
        splits["downstream_train"] = subset
        splits["downstream_val"]   = subset
        splits["test"]             = subset

    age_sex_lookup = load_age_sex_lookup(Path(args.excel))

    train_ages = [
        age_sex_lookup[Path(s["image"]).stem][0]
        for s in splits["downstream_train"]
        if Path(s["image"]).stem in age_sex_lookup
    ]
    age_mean = float(np.mean(train_ages))
    age_std  = float(np.std(train_ages))

    preprocess_train = build_train_transform()
    preprocess_val   = build_val_transform()

    train_ds = DownstreamDataset(splits["downstream_train"], age_sex_lookup, age_mean, age_std,
                                  preprocess_train, use_mask=args.use_mask)
    val_ds   = DownstreamDataset(splits["downstream_val"],   age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   use_mask=args.use_mask)
    test_ds  = DownstreamDataset(splits["test"],             age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   use_mask=args.use_mask)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    use_pin       = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Class weighting ───────────────────────────────────────────────────────
    train_labels_all = torch.tensor(
        [LABEL_TO_IDX[s["label"]] for s in train_ds.samples], dtype=torch.long
    )
    label_counts  = torch.bincount(train_labels_all, minlength=NUM_CLASSES).float()
    class_weights = compute_class_weights(label_counts, NUM_CLASSES, args.class_weighting,
                                           args.cb_beta, device)
    criterion = build_classification_loss(args, label_counts, NUM_CLASSES, class_weights, device)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.lr_encoder},
        {"params": mlp.parameters(),     "lr": args.lr_mlp},
    ], weight_decay=args.weight_decay)

    scheduler = LambdaLR(optimizer, lr_lambda=[
        lambda epoch: _lr_factor(epoch, args.warmup_epochs, args.epochs),
        lambda epoch: 1.0,
    ])

    # ── Output dir ────────────────────────────────────────────────────────────
    run_id  = (wandb.run.id if use_wandb and wandb.run else None) or \
              datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"scratch_img_{args.encoder}" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_checkpoint.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})\n")

    for epoch in range(1, args.epochs + 1):
        train_loss                               = train_one_epoch(encoder, mlp, train_loader,
                                                                    optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(encoder, mlp, val_loader,
                                                             criterion, device)
        scheduler.step()

        val_bal_acc      = balanced_accuracy_score(val_labels, val_preds)
        val_combined_acc = 0.5 * val_acc + 0.5 * val_bal_acc

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | val_bal_acc={val_bal_acc:.3f} | "
            f"val_combined_acc={val_combined_acc:.3f} | "
            f"enc_lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/balanced_acc": val_bal_acc,
                "val/combined_acc": val_combined_acc,
                "lr/encoder": optimizer.param_groups[0]["lr"],
            }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "mlp_state_dict": mlp.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    mlp.load_state_dict(ckpt["mlp_state_dict"])

    test_loss, test_acc, test_preds, test_labels = evaluate(encoder, mlp, test_loader,
                                                             criterion, device)
    label_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Loss: {test_loss:.4f}  |  Accuracy: {test_acc:.3f}")
    test_bal_acc = balanced_accuracy_score(test_labels, test_preds)
    print(f"Balanced accuracy: {test_bal_acc:.3f}")
    print(f"Macro F1: {f1_score(test_labels, test_preds, average='macro'):.3f}")
    print()
    print(classification_report(test_labels, test_preds,
                                  labels=list(range(NUM_CLASSES)),
                                  target_names=label_names, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(
        confusion_matrix(test_labels, test_preds, labels=list(range(NUM_CLASSES))),
        index=label_names, columns=label_names,
    ).to_string())

    if use_wandb:
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average="macro", zero_division=0
        )
        per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, labels=list(range(NUM_CLASSES)), zero_division=0
        )
        log_dict = {
            "test/loss": test_loss, "test/acc": test_acc,
            "test/balanced_acc": test_bal_acc,
            "test/precision_macro": test_prec,
            "test/recall_macro": test_rec,
            "test/f1_macro": test_f1,
        }
        for i, name in enumerate(label_names):
            log_dict[f"test/precision_{name}"] = per_class_prec[i]
            log_dict[f"test/recall_{name}"]    = per_class_rec[i]
            log_dict[f"test/f1_{name}"]        = per_class_f1[i]
        wandb.log(log_dict)
        wandb.finish()

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")
