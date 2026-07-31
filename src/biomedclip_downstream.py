"""BiomedCLIP supervised downstream classifier — all head variants.

Always uses 768-dim pre-projection ViT features (raw CLS token, before head.proj).

Head variants (--head):
    linear       — single nn.Linear(768, 3), no metadata
    mlp          — MLP with age/sex late fusion  (default)
    mlp_no_meta  — same MLP capacity, no clinical metadata

Usage:
    python src/biomedclip_downstream.py \\
        --checkpoint results/biomedclip_pretrain/.../best_r1_checkpoint.pt \\
        --splits     results/biomedclip_pretrain/.../splits.json \\
        --excel      data/internal_dataset/metadata.xlsx \\
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

from biomedclip.utils.misc import ROOT_DIR, MODEL_TAG, DEFAULT_OUT_DIR, DEFAULT_SPLITS
from biomedclip.data.transforms import build_train_transform, build_preprocess_val
from LACE.models.encoders import enable_dynamic_img_size
from biomedclip.data.datasets import (
    DownstreamDataset, EmbeddingDataset,
    LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES,
    LABEL_TO_IDX_BINARY, IDX_TO_LABEL_BINARY, NUM_CLASSES_BINARY,
)
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.lora import inject_lora, unfreeze_or_inject_downstream_lora
from biomedclip.models.classifier import LinearHead, MalignancyMLP, extract_embeddings
from biomedclip.utils.downstream_eval import (
    require_binary_for_btxrd, load_btxrd_samples, build_downstream_loader, report_eval,
)
DEFAULT_CHECKPOINT = ROOT_DIR / "results" / "biomedclip_pretrain" / "best_r1_checkpoint.pt"

EMBED_DIM = 768  # always pre-projection ViT CLS token


def build_head(args: argparse.Namespace, embed_dim: int, device: torch.device,
               num_classes: int = NUM_CLASSES) -> nn.Module:
    if args.head == "linear":
        return LinearHead(embed_dim, num_classes=num_classes).to(device)
    elif args.head == "mlp_no_meta":
        return MalignancyMLP(embed_dim, args.hidden_dims, args.dropout,
                             args.meta_embed_dim, use_meta=False,
                             num_classes=num_classes).to(device)
    else:  # mlp (default)
        return MalignancyMLP(embed_dim, args.hidden_dims, args.dropout,
                             args.meta_embed_dim, use_meta=True,
                             num_classes=num_classes).to(device)


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


def train_one_epoch_finetune(
    head: nn.Module,
    encoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    head.train()
    encoder.eval()  # keeps dropout/BN frozen; LoRA params still receive gradients
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        age    = batch["age"].to(device)
        sex    = batch["sex"].to(device)
        lbl    = batch["label"].to(device)
        optimizer.zero_grad()
        emb    = F.normalize(encoder(images), dim=-1)
        logits = head(emb, age, sex)
        loss   = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_finetune(
    head: nn.Module,
    encoder: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    head.eval()
    encoder.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        age    = batch["age"].to(device)
        sex    = batch["sex"].to(device)
        lbl    = batch["label"].to(device)
        emb    = F.normalize(encoder(images), dim=-1)
        logits = head(emb, age, sex)
        total_loss += criterion(logits, lbl).item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(lbl.cpu())
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / len(loader), float((preds == labels).mean()), preds, labels


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
        description="BiomedCLIP downstream malignancy classifier — linear / MLP / MLP-no-meta heads."
    )
    # ── Checkpoint / splits ──────────────────────────────────────────────────
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--freezed_biomedclip", action="store_true",
                        help="Use vanilla BiomedCLIP weights (no fine-tuned checkpoint, no LoRA).")
    parser.add_argument("--splits",   default=str(DEFAULT_SPLITS))
    parser.add_argument("--out_dir",  default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--use_mask", action="store_true")

    # ── Head variant ─────────────────────────────────────────────────────────
    parser.add_argument("--head", default="mlp",
                        choices=["linear", "mlp", "mlp_no_meta"],
                        help="Classification head: linear probe, MLP+meta (default), or MLP without metadata.")

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--patience",       type=int,   default=10)
    parser.add_argument("--early_stopping_metric", default="val_loss",
                        choices=["val_loss", "val_bal_acc"],
                        help="Metric to monitor for early stopping and best-checkpoint saving.")
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

    # ── Encoder fine-tuning (LoRA) ────────────────────────────────────────────
    parser.add_argument("--finetune_lora_layers", type=int, default=0,
                        help="Unfreeze/inject LoRA into the last N encoder blocks for downstream fine-tuning "
                             "(0 = frozen encoder / linear probing, default: %(default)s)")
    parser.add_argument("--lr_encoder",    type=float, default=1e-5,
                        help="LR for encoder LoRA params when --finetune_lora_layers > 0 (default: %(default)s)")
    parser.add_argument("--finetune_lora_r", type=int, default=8,
                        help="LoRA rank for freshly injected downstream LoRA "
                             "(only used when the loaded block has no pretrained LoRA, default: %(default)s)")

    # ── Binary classification ─────────────────────────────────────────────────
    parser.add_argument("--binary", type=lambda x: str(x).lower() in ("true", "1", "yes"),
                        default=False,
                        help="Binary classification (benign vs malignant). "
                             "Use with split_binary.json — intermediate cases must already be excluded.")

    # ── Held-out test / external BTXRD evaluation ─────────────────────────────
    parser.add_argument("--eval_test", action="store_true",
                        help="Evaluate the split's held-out test set after training. Off by "
                             "default so hyperparameter sweeps never touch test/BTXRD.")
    parser.add_argument("--btxrd_manifest", default=None,
                        help="Path to a BTXRD downstream manifest (built by "
                             "build_btxrd_downstream.py) to also score as an external test set. "
                             "Requires --eval_test and --binary.")

    # ── Image resolution ─────────────────────────────────────────────────────
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input resolution (default 224). BiomedCLIP ViT-B/16 supports "
                             "arbitrary sizes via pos-embedding interpolation.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--run_name",      default=None,
                        help="Override the auto-generated run name (used for the checkpoint "
                             "dir under --out_dir/biomedclip_downstream/; e.g. per-fold naming "
                             "in CV mode). Default: run_<head>_<loss>_<class_weighting>_<timestamp>.")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="biomedclip-downstream")
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
        "early_stopping_metric", "finetune_lora_layers", "lr_encoder",
        "finetune_lora_r",
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

    # ── Load BiomedCLIP and build encoder ─────────────────────────────────────
    print(f"Loading model: {MODEL_TAG}")
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)
    preprocess_val = build_preprocess_val(preprocess_val, args.image_size)
    if args.image_size != 224:
        print(f"Image size: {args.image_size}×{args.image_size} (pos-embedding interpolated)")

    if args.freezed_biomedclip:
        print("Using vanilla BiomedCLIP encoder (no checkpoint, no LoRA).")
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

    # Always use pre-projection 768-dim ViT CLS token.
    encoder = model.visual.trunk.to(device)
    if args.image_size != 224:
        enable_dynamic_img_size(encoder)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    finetune = args.finetune_lora_layers > 0
    if finetune:
        lora_params = unfreeze_or_inject_downstream_lora(
            encoder, args.finetune_lora_layers,
            r=args.finetune_lora_r, alpha=args.finetune_lora_r * 2,
        )
        print(f"Encoder: last {args.finetune_lora_layers} blocks LoRA fine-tuned "
              f"({len(lora_params)} trainable params, r={args.finetune_lora_r})")
    else:
        lora_params = []
        print(f"Encoder: pre-projection ViT features ({EMBED_DIM}-dim), frozen")

    preprocess_train = build_train_transform(preprocess_val)

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

    if finetune:
        # Val is re-evaluated each epoch with on-the-fly embeddings (encoder changes).
        # Test embeddings are computed once after training from the best checkpoint.
        val_loader = val_loader_raw
        embed_dim  = EMBED_DIM
    else:
        print("Pre-computing val embeddings...")
        val_emb,  val_age,  val_sex,  val_lbl  = extract_embeddings(encoder, val_loader_raw,  device)
        embed_dim = val_emb.shape[1]
        print(f"Embedding dim: {embed_dim}")
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
    if finetune:
        optimizer = torch.optim.AdamW([
            {"params": head.parameters(),  "lr": args.lr,         "weight_decay": args.weight_decay},
            {"params": lora_params,         "lr": args.lr_encoder, "weight_decay": args.weight_decay},
        ], betas=(0.9, 0.98), eps=1e-6)
    else:
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name  = args.run_name or (
        f"run_{args.head}_{args.loss}_{args.class_weighting}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir   = Path(args.out_dir) / "biomedclip_downstream" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best_head.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    maximize_metric = args.early_stopping_metric == "val_bal_acc"
    best_metric      = -float("inf") if maximize_metric else float("inf")
    patience_counter = 0
    print(f"\nTraining {args.head} head for {args.epochs} epochs "
          f"(patience={args.patience}, monitor={args.early_stopping_metric})\n")

    for epoch in range(1, args.epochs + 1):
        if finetune:
            train_loss                               = train_one_epoch_finetune(head, encoder, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_preds, val_labels = evaluate_finetune(head, encoder, val_loader, criterion, device)
        else:
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
            save_dict = {
                "epoch": epoch,
                "head_state_dict": head.state_dict(),
                "val_loss": val_loss,
                "val_bal_acc": val_bal_acc,
                "head": args.head,
                "args": vars(args),
                "age_mean": age_mean,
                "age_std": age_std,
                "embed_dim": embed_dim,
            }
            if finetune:
                save_dict["encoder_lora_state_dict"] = {
                    name: p.detach().cpu()
                    for name, p in encoder.named_parameters() if p.requires_grad
                }
            torch.save(save_dict, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    # ── Restore best head (and encoder for fine-tuning) ───────────────────────
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    head.load_state_dict(best_ckpt["head_state_dict"])
    if finetune:
        enc_state = encoder.state_dict()
        enc_state.update(best_ckpt["encoder_lora_state_dict"])
        encoder.load_state_dict(enc_state)

    # ── Held-out test / external BTXRD evaluation (opt-in via --eval_test) ─────
    # Metrics returned to callers (e.g. the k-fold CV orchestrator).
    results: dict = {
        f"val/best_{args.early_stopping_metric}": best_metric,
    }
    eval_metrics: dict = {}
    if args.eval_test:
        if finetune:
            print("Pre-computing test embeddings from best checkpoint...")
            test_emb, test_age, test_sex, test_lbl = extract_embeddings(encoder, test_loader_raw, device)
            test_emb_ds = EmbeddingDataset(test_emb, test_age, test_sex, test_lbl)
            test_loader = DataLoader(test_emb_ds, batch_size=args.batch_size, shuffle=False)
        test_loss, _, test_preds, test_labels = evaluate(head, test_loader, criterion, device)
        test_metrics = report_eval(
            "TEST", test_preds, test_labels, test_loss, idx_to_label, num_classes, prefix="test")
        eval_metrics.update(test_metrics)
        results.update(test_metrics)
        # raw per-sample predictions for pooled out-of-fold CV scoring
        # (consumed by downstream_cv.py; ignored by single-run callers)
        results["test/_preds"]  = test_preds.tolist()
        results["test/_labels"] = test_labels.tolist()

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
            btxrd_metrics = report_eval(
                "BTXRD (external)", btxrd_preds, btxrd_labels, btxrd_loss,
                idx_to_label, num_classes, prefix="btxrd")
            eval_metrics.update(btxrd_metrics)
            results.update(btxrd_metrics)
    else:
        print("\nSkipping held-out test / BTXRD evaluation (--eval_test not set).")

    if use_wandb:
        if eval_metrics:
            wandb.log(eval_metrics)
        wandb.finish()

    print(f"\nBest {args.early_stopping_metric}: {best_metric:.4f}")
    print(f"Checkpoints saved to: {run_dir}")
    return results


if __name__ == "__main__":
    main(parse_args())
