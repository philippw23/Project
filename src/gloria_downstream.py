"""GLoRIA supervised downstream classifier — non-linear probing.

Loads the GLoRIA image encoder (ResNet-50, pretrained on CheXPert) from a
lightning checkpoint and trains a frozen-encoder classification head.

Encoder features (--use_projection):
    False (default) — 2048-dim pre-projection ResNet-50 global features
    True            — 768-dim post-projection (via img_encoder.global_embedder)

Head variants (--head):
    linear       — single nn.Linear(embed_dim, 3), no metadata
    mlp          — MLP with age/sex late fusion (default)
    mlp_no_meta  — same MLP capacity, no clinical metadata

Usage:
    python src/gloria_downstream.py \\
        --checkpoint src/gloria/pretrained/chexpert_resnet50.ckpt \\
        --splits     data/internal_dataset/split.json \\
        --head       mlp
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              precision_recall_fscore_support)
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# GLoRIA lives in its own older conda environment; register stub modules for
# dependencies (pytorch_lightning, skimage, nltk, cv2) that are absent in the
# main environment but are only needed for text-processing / visualisation code
# that we never invoke during downstream probing.
GLORIA_DIR = Path(__file__).resolve().parent / "gloria"
if str(GLORIA_DIR) not in sys.path:
    sys.path.insert(0, str(GLORIA_DIR))

import types as _types
import importlib.util as _ilu


class _AutoStub(_types.ModuleType):
    """Module stub that returns a no-op object for any attribute access."""
    class _NoOp:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return self
        def __getattr__(self, n): return self.__class__()

    def __getattr__(self, name: str):
        return self._NoOp


class _StubFinder:
    _STUB_PREFIXES = ("pytorch_lightning", "skimage", "nltk", "cv2")

    def find_module(self, name, path=None):  # noqa: D102
        if any(name == p or name.startswith(p + ".") for p in self._STUB_PREFIXES):
            return self

    def load_module(self, name):  # noqa: D102
        if name not in sys.modules:
            sys.modules[name] = _AutoStub(name)
        return sys.modules[name]


sys.meta_path.insert(0, _StubFinder())

from biomedclip.utils.misc import ROOT_DIR, DEFAULT_SPLITS, DEFAULT_OUT_DIR
from biomedclip.data.datasets import (
    DownstreamDataset, EmbeddingDataset, LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES,
)
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.classifier import LinearHead, MalignancyMLP, extract_embeddings

DEFAULT_GLORIA_CKPT = ROOT_DIR / "src" / "gloria" / "pretrained" / "chexpert_resnet50.ckpt"
GLORIA_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 'half' normalisation used in GLoRIA pretraining


# ── Transforms ────────────────────────────────────────────────────────────────

def build_gloria_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*GLORIA_NORM),
    ])


def build_gloria_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4)),
        transforms.ToTensor(),
        transforms.Normalize(*GLORIA_NORM),
    ])


# ── Encoder loading ───────────────────────────────────────────────────────────

class GLoRIAEncoder(nn.Module):
    """Wraps GLoRIA img_encoder to expose a simple forward(images) -> embedding API."""

    def __init__(self, img_encoder: nn.Module, use_projection: bool = False) -> None:
        super().__init__()
        self.img_encoder = img_encoder
        self.use_projection = use_projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_ft = self.img_encoder(x)  # 2048-dim pre-projection
        if self.use_projection:
            return self.img_encoder.global_embedder(global_ft)  # 768-dim
        return global_ft


def _load_gloria_submodule(pkg_name: str, rel_path: str):
    """Load a gloria submodule directly from its file, bypassing the package __init__."""
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    path = GLORIA_DIR / "gloria" / rel_path
    spec = _ilu.spec_from_file_location(pkg_name, path)
    mod  = _ilu.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_gloria_vision_modules() -> tuple:
    """Import only the vision parts of gloria, skipping the text / PL chain."""
    # Register a dummy gloria package so relative imports inside the files work.
    if "gloria" not in sys.modules:
        _pkg = _types.ModuleType("gloria")
        _pkg.__path__ = [str(GLORIA_DIR / "gloria")]
        _pkg.__package__ = "gloria"
        sys.modules["gloria"] = _pkg
    if "gloria.models" not in sys.modules:
        _mpkg = _types.ModuleType("gloria.models")
        _mpkg.__path__ = [str(GLORIA_DIR / "gloria" / "models")]
        _mpkg.__package__ = "gloria.models"
        sys.modules["gloria.models"] = _mpkg

    _load_gloria_submodule("gloria.models.cnn_backbones", "models/cnn_backbones.py")
    vision_mod = _load_gloria_submodule("gloria.models.vision_model", "models/vision_model.py")
    lora_mod   = _load_gloria_submodule("gloria.models.lora", "models/lora.py")
    return vision_mod, lora_mod


def load_gloria_encoder(
    ckpt_path: str, use_projection: bool = False
) -> tuple[GLoRIAEncoder, int]:
    """Load GLoRIA img_encoder from a checkpoint.

    Handles three checkpoint formats:
      - Original CheXPert pytorch-lightning checkpoint
      - Bone-tumor fine-tuned lightning checkpoint (LoRA injected post-load)
      - gloria_pretrain_v1 format saved by gloria_pretrain.py

    Returns (encoder_module, embed_dim).
    """
    from omegaconf import OmegaConf

    vision_mod, lora_mod = _ensure_gloria_vision_modules()
    ImageEncoder       = vision_mod.ImageEncoder
    inject_lora_gloria = lora_mod.inject_lora_gloria
    unfreeze_gloria    = lora_mod.unfreeze_gloria

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if ckpt.get("format") == "gloria_pretrain_v1":
        # New format saved by gloria_pretrain.py
        cfg         = OmegaConf.create(ckpt["original_gloria_cfg"])
        img_encoder = ImageEncoder(cfg)

        adapter_mode = ckpt.get("adapter_mode", "lora")
        n_layers     = ckpt.get("n_layers", 0)
        if adapter_mode == "lora":
            inject_lora_gloria(
                img_encoder,
                lora_layers=n_layers,
                r=ckpt.get("lora_r", 8),
                alpha=ckpt.get("lora_alpha", 16.0),
            )
        elif adapter_mode == "unfreeze":
            unfreeze_gloria(img_encoder, n_layers=n_layers)

        img_encoder.load_state_dict(ckpt["img_encoder_state_dict"], strict=True)
        print(f"Loaded gloria_pretrain_v1 checkpoint ({adapter_mode}, n_layers={n_layers}): {ckpt_path}")

    else:
        # Original pytorch-lightning format
        cfg         = ckpt["hyper_parameters"]
        img_encoder = ImageEncoder(cfg)

        lora_cfg = cfg.model.vision.get("lora")
        if lora_cfg:
            inject_lora_gloria(
                img_encoder,
                lora_layers=lora_cfg.lora_layers,
                r=lora_cfg.r,
                alpha=lora_cfg.alpha,
            )

        state = {
            k.replace("gloria.img_encoder.", "", 1): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("gloria.img_encoder.")
        }
        img_encoder.load_state_dict(state, strict=True)
        print(f"Loaded GLoRIA img_encoder from: {ckpt_path}")

    feature_dim = img_encoder.feature_dim    # 2048 for ResNet-50
    embed_dim   = img_encoder.output_dim if use_projection else feature_dim

    return GLoRIAEncoder(img_encoder, use_projection=use_projection), embed_dim


# ── Head construction ─────────────────────────────────────────────────────────

def build_head(args: argparse.Namespace, embed_dim: int, device: torch.device) -> nn.Module:
    if args.head == "linear":
        return LinearHead(embed_dim).to(device)
    if args.head == "mlp_no_meta":
        return MalignancyMLP(embed_dim, args.hidden_dims, args.dropout,
                             args.meta_embed_dim, use_meta=False).to(device)
    return MalignancyMLP(embed_dim, args.hidden_dims, args.dropout,
                         args.meta_embed_dim, use_meta=True).to(device)


# ── Training / evaluation loops ───────────────────────────────────────────────

def train_one_epoch(
    head: nn.Module,
    encoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    head.train()
    total_loss = 0.0
    for batch in loader:
        image = batch["image"].to(device)
        age   = batch["age"].to(device)
        sex   = batch["sex"].to(device)
        lbl   = batch["label"].to(device)
        with torch.no_grad():
            emb = encoder(image)
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLoRIA downstream malignancy classifier — frozen encoder + head."
    )
    # ── Checkpoint / splits ──────────────────────────────────────────────────
    parser.add_argument("--checkpoint", default=str(DEFAULT_GLORIA_CKPT))
    parser.add_argument("--splits",     default=str(DEFAULT_SPLITS))
    parser.add_argument("--out_dir",    default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--use_mask",   action="store_true")
    parser.add_argument("--use_projection", action="store_true",
                        help="Use 768-dim post-projection embedding instead of "
                             "2048-dim pre-projection ResNet features.")

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
    parser.add_argument(
        "--loss", default="ce",
        choices=["ce", "wce", "ce_smooth", "focal", "cb_focal", "ldam", "balanced_softmax"],
    )
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
    parser.add_argument("--wandb_project", default="gloria-downstream")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Head: {args.head}  |  "
          f"Early-stopping: {args.early_stopping_metric}")

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

    # ── Load GLoRIA encoder ───────────────────────────────────────────────────
    encoder, embed_dim = load_gloria_encoder(args.checkpoint, args.use_projection)
    proj_label = (
        "post-projection 768-dim" if args.use_projection
        else f"pre-projection {embed_dim}-dim"
    )
    print(f"Encoder: GLoRIA img_encoder ({proj_label}), frozen")

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    encoder = encoder.to(device)

    # ── Transforms ───────────────────────────────────────────────────────────
    preprocess_train = build_gloria_train_transform()
    preprocess_val   = build_gloria_val_transform()

    # ── Data ──────────────────────────────────────────────────────────────────
    with open(args.splits, encoding="utf-8") as fh:
        raw_splits = json.load(fh)
    splits = {
        "train": raw_splits["train"],
        "val":   raw_splits["val"],
        "test":  raw_splits["test"],
    }

    all_samples    = splits["train"] + splits["val"] + splits["test"]
    age_sex_lookup = {
        Path(s["image"]).stem: (float(s["age"]), float(s["sex"])) for s in all_samples
    }
    train_ages     = [s["age"] for s in splits["train"]]
    age_mean = float(np.mean(train_ages))
    age_std  = float(np.std(train_ages))
    print(f"Age stats (train): mean={age_mean:.1f}, std={age_std:.1f}")

    train_ds = DownstreamDataset(splits["train"], age_sex_lookup, age_mean, age_std,
                                  preprocess_train, args.use_mask)
    val_ds   = DownstreamDataset(splits["val"],   age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   args.use_mask)
    test_ds  = DownstreamDataset(splits["test"],  age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   args.use_mask)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    use_pin       = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}
    train_loader_raw = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader_raw   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader_raw  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Class weighting ───────────────────────────────────────────────────────
    train_labels_all = torch.tensor(
        [LABEL_TO_IDX[s["label"]] for s in train_ds.samples], dtype=torch.long
    )
    label_counts  = torch.bincount(train_labels_all, minlength=NUM_CLASSES).float()
    class_weights = compute_class_weights(
        label_counts, num_classes=NUM_CLASSES, mode=args.class_weighting,
        beta=args.cb_beta, device=device,
    )
    counts_str = {IDX_TO_LABEL[i]: int(label_counts[i]) for i in range(NUM_CLASSES)}
    print(f"Class counts (train): {counts_str}")
    print(f"Loss: {args.loss} | class_weighting={args.class_weighting}")

    # ── Pre-compute embeddings for val/test (deterministic) ───────────────────
    # Training embeddings are NOT pre-computed: the encoder runs live each batch
    # so that random augmentations (random crop, affine, colour jitter) are
    # re-sampled every epoch rather than frozen to a single pass.
    print("Pre-computing val/test embeddings...")
    val_emb, val_age, val_sex, val_lbl = extract_embeddings(
        encoder, val_loader_raw, device)
    test_emb, test_age, test_sex, test_lbl = extract_embeddings(
        encoder, test_loader_raw, device)
    print(f"Embedding dim: {val_emb.shape[1]}")

    val_emb_ds  = EmbeddingDataset(val_emb,  val_age,  val_sex,  val_lbl)
    test_emb_ds = EmbeddingDataset(test_emb, test_age, test_sex, test_lbl)
    train_loader = train_loader_raw   # raw images — encoder runs per batch
    val_loader   = DataLoader(val_emb_ds,  batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_emb_ds, batch_size=args.batch_size, shuffle=False)

    # ── Head, loss, optimiser ─────────────────────────────────────────────────
    head      = build_head(args, embed_dim, device)
    criterion = build_classification_loss(args, label_counts, NUM_CLASSES, class_weights, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = (
        f"run_{args.head}_{args.loss}_{args.class_weighting}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir   = Path(args.out_dir) / "gloria_downstream" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best_head.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    maximize_metric  = args.early_stopping_metric == "val_bal_acc"
    best_metric      = -float("inf") if maximize_metric else float("inf")
    patience_counter = 0
    print(f"\nTraining {args.head} head for {args.epochs} epochs "
          f"(patience={args.patience}, monitor={args.early_stopping_metric})\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            head, encoder, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(
            head, val_loader, criterion, device)
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
                "head_state_dict": head.state_dict(),
                "val_loss": val_loss,
                "val_bal_acc": val_bal_acc,
                "head": args.head,
                "embed_dim": embed_dim,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    head.load_state_dict(best_ckpt["head_state_dict"])
    test_loss, test_acc, test_preds, test_labels = evaluate(head, test_loader, criterion, device)

    label_names    = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    present_labels = sorted(set(test_labels.tolist()) | set(test_preds.tolist()))
    present_names  = [label_names[i] for i in present_labels]
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Loss: {test_loss:.4f}  |  Accuracy: {test_acc:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(test_labels, test_preds):.3f}")
    print(f"Macro F1: {f1_score(test_labels, test_preds, average='macro'):.3f}")
    test_weighted_prec, test_weighted_rec, _, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="weighted", zero_division=0
    )
    print(f"Weighted Precision: {test_weighted_prec:.3f}  |  Weighted Recall: {test_weighted_rec:.3f}")
    print()
    print(classification_report(
        test_labels, test_preds,
        labels=present_labels, target_names=present_names,
        digits=3, zero_division=0,
    ))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(
        confusion_matrix(test_labels, test_preds, labels=present_labels),
        index=present_names, columns=present_names,
    ).to_string())

    if use_wandb:
        test_bal_acc = balanced_accuracy_score(test_labels, test_preds)
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average="macro", zero_division=0
        )
        per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, labels=present_labels, zero_division=0
        )
        log_dict = {
            "test/loss": test_loss, "test/acc": test_acc,
            "test/balanced_acc": test_bal_acc,
            "test/precision_macro":    test_prec,    "test/recall_macro":    test_rec,
            "test/precision_weighted": test_weighted_prec, "test/recall_weighted": test_weighted_rec,
            "test/f1_macro": test_f1,
        }
        for i, name in enumerate(present_names):
            log_dict[f"test/precision_{name}"] = per_class_prec[i]
            log_dict[f"test/recall_{name}"]    = per_class_rec[i]
            log_dict[f"test/f1_{name}"]        = per_class_f1[i]
        wandb.log(log_dict)
        wandb.finish()

    print(f"\nBest {args.early_stopping_metric}: {best_metric:.4f}")
    print(f"Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main(parse_args())
