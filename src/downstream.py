"""Unified downstream malignancy classification script for all baselines.

Supports three baselines via --baseline:
  biomedclip — BiomedCLIP ViT-B/16 (512 or 768-dim)
  chexfound  — CheXFound ViT-L/16 (1024-dim)

MLP input : [image_embedding | age (1, z-scored) | sex (1, binary)]
MLP output: 3-class logits  (benign=0 / intermediate=1 / malignant=2)

Usage (BiomedCLIP, pretrained checkpoint):
    python src/downstream.py \\
        --baseline biomedclip \\
        --biomedclip_checkpoint results/biomedclip_pretrain/.../best_val_checkpoint.pt \\
        --splits results/biomedclip_pretrain/.../splits.json \\
        --excel  data/metadata.xlsx

Usage (BiomedCLIP, frozen vanilla):
    python src/downstream.py \\
        --baseline biomedclip --freezed_biomedclip \\
        --splits ... --excel ...

Usage (CheXFound, continued-pretrain checkpoint):
    python src/downstream.py \\
        --baseline chexfound \\
        --chexfound_checkpoint results/chexfound_pretrain/.../checkpoint_last.pth \\
        --splits ... --excel ...

Usage (CheXFound, frozen original weights):
    python src/downstream.py \\
        --baseline chexfound \\
        --chexfound_checkpoint none \\
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

from biomedclip.utils.misc import ROOT_DIR, DEFAULT_EXCEL, DEFAULT_OUT_DIR
from biomedclip.data.datasets import (
    DownstreamDataset, EmbeddingDataset, LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES,
)
from biomedclip.data.splits import load_age_sex_lookup
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.classifier import MalignancyMLP, extract_embeddings

DEFAULT_SPLITS = ROOT_DIR / "results" / "biomedclip_pretrain" / "splits.json"

_CHEXFOUND_DATA = ROOT_DIR / "src" / "chexfound" / "data"
DEFAULT_CHEXFOUND_CONFIG     = _CHEXFOUND_DATA / "config.yaml"
DEFAULT_CHEXFOUND_CHECKPOINT = _CHEXFOUND_DATA / "teacher_checkpoint.pth"


# ── Encoder building ─────────────────────────────────────────────────────────

def build_encoder(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, object, object, int]:
    """Return (encoder, train_transform, val_transform, embed_dim) for the selected baseline.

    The returned encoder is frozen (all requires_grad=False) and placed on device.
    """
    if args.baseline == "biomedclip":
        return _build_biomedclip_encoder(args, device)
    elif args.baseline == "chexfound":
        return _build_chexfound_encoder(args, device)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline!r}")


def _build_biomedclip_encoder(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, object, object, int]:
    import open_clip
    from biomedclip.utils.misc import MODEL_TAG
    from biomedclip.data.transforms import build_train_transform
    from biomedclip.models.lora import inject_lora

    print(f"Loading BiomedCLIP model: {MODEL_TAG}")
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)

    if args.freezed_biomedclip:
        print("Using vanilla BiomedCLIP encoder (no checkpoint, no LoRA).")
    else:
        ckpt = torch.load(args.biomedclip_checkpoint, map_location="cpu", weights_only=False)
        lora_cfg = ckpt.get("lora_config") or {}
        if not lora_cfg:
            raise RuntimeError(
                f"Checkpoint '{args.biomedclip_checkpoint}' has no 'lora_config'. "
                "Re-run pretraining with the current biomedclip_pretrain.py."
            )
        inject_lora(model, lora_cfg["lora_layers"], lora_cfg["lora_r"], lora_cfg["lora_alpha"])
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"Loaded BiomedCLIP checkpoint: {args.biomedclip_checkpoint} "
            f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})"
        )

    if args.use_projected_features:
        encoder = model.visual.to(device)
        embed_dim = 512
        print("BiomedCLIP encoder: projected features (512-dim)")
    else:
        encoder = model.visual.trunk.to(device)
        embed_dim = 768
        print("BiomedCLIP encoder: pre-projection ViT features (768-dim)")

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    preprocess_train = build_train_transform(preprocess_val)
    return encoder, preprocess_train, preprocess_val, embed_dim


def _build_chexfound_encoder(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, object, object, int]:
    from chexfound.data.transforms import (
        build_preprocess_val_chexfound,
        build_train_transform_chexfound,
    )
    from chexfound.models.encoders import CheXFoundViT, load_continued_pretrain_weights

    preprocess_val   = build_preprocess_val_chexfound()
    preprocess_train = build_train_transform_chexfound(preprocess_val)

    checkpoint = (
        None if args.chexfound_checkpoint.lower() == "none" else args.chexfound_checkpoint
    )

    if checkpoint is None:
        if not args.chexfound_weights:
            raise ValueError(
                "--chexfound_weights is required for frozen baseline mode. "
                "Provide the path to the original CheXFound .pth file, "
                "or omit --chexfound_checkpoint to use the bundled checkpoint."
            )
        print("Loading frozen CheXFound baseline (original pretrained weights).")
        encoder = CheXFoundViT(
            args.chexfound_config, args.chexfound_weights,
            lora_layers=0, r=8, alpha=16.0, load_pretrained=True,
        )
    else:
        print(f"Loading CheXFound encoder from continued-pretrain checkpoint: {checkpoint}")
        encoder = CheXFoundViT(
            args.chexfound_config, weights_path=None,
            lora_layers=0, r=8, alpha=16.0, load_pretrained=False,
        )
        load_continued_pretrain_weights(encoder, checkpoint)

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder = encoder.to(device)
    encoder.eval()

    return encoder, preprocess_train, preprocess_val, 1024


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_one_epoch(
    mlp: nn.Module,
    encoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    mlp.train()
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
        logits = mlp(emb, age, sex)
        loss   = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    mlp: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    mlp.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for emb, age, sex, lbl in loader:
        emb, age, sex, lbl = emb.to(device), age.to(device), sex.to(device), lbl.to(device)
        logits = mlp(emb, age, sex)
        total_loss += criterion(logits, lbl).item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(lbl.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc    = (preds == labels).mean()
    return total_loss / len(loader), float(acc), preds, labels


# ── Argparse ─────────────────────────────────────────────────────────────────

def _str_to_bool(v: str) -> bool:
    return str(v).lower() not in ("false", "0", "no", "none")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified downstream malignancy classifier for BiomedCLIP / CheXFound."
    )

    # ── Baseline selector ────────────────────────────────────────────────────
    parser.add_argument("--baseline", required=True,
                        choices=["biomedclip", "chexfound"],
                        help="Which pretrained encoder to use.")

    # ── BiomedCLIP-specific args ─────────────────────────────────────────────
    parser.add_argument("--biomedclip_checkpoint",
                        default=str(ROOT_DIR / "results" / "biomedclip_pretrain" / "best_r1_checkpoint.pt"),
                        help="Path to BiomedCLIP pretrain checkpoint. Ignored when --baseline chexfound.")
    parser.add_argument("--freezed_biomedclip", action="store_true",
                        help="Use vanilla BiomedCLIP weights without a fine-tuned checkpoint.")
    parser.add_argument("--use_projected_features", nargs="?", const=True,
                        type=_str_to_bool, default=False,
                        help="512-dim projected CLIP embedding instead of 768-dim ViT features.")

    # ── CheXFound-specific args ───────────────────────────────────────────────
    parser.add_argument("--chexfound_config",  default=str(DEFAULT_CHEXFOUND_CONFIG),
                        help="CheXFound model config YAML.")
    parser.add_argument("--chexfound_weights", default=None,
                        help="Original CheXFound .pth checkpoint (frozen baseline mode only).")
    parser.add_argument("--chexfound_checkpoint", default=str(DEFAULT_CHEXFOUND_CHECKPOINT),
                        help="Continued-pretrain checkpoint. Pass 'none' for frozen baseline mode.")

    # ── Data / splits ─────────────────────────────────────────────────────────
    parser.add_argument("--splits",   default=str(DEFAULT_SPLITS))
    parser.add_argument("--excel",    default=str(DEFAULT_EXCEL))
    parser.add_argument("--out_dir",  default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--use_mask", action="store_true")

    # ── Training hyperparameters ──────────────────────────────────────────────
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--meta_embed_dim", type=int,  default=16)
    parser.add_argument("--hidden_dims",   type=str,   nargs="+", default=[256, 128])
    parser.add_argument("--weight_decay",  type=float, default=0.01)

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
    parser.add_argument("--wandb_project", default=None,
                        help="W&B project name. Defaults to '{baseline}-downstream'.")
    parser.add_argument("--wandb_run",     default=None)
    parser.add_argument("--wandb_entity",  default=None)
    parser.add_argument("--sweep",         action="store_true")

    args = parser.parse_args(argv)

    raw = " ".join(str(x) for x in args.hidden_dims)
    args.hidden_dims = [int(x) for x in raw.strip("[]").replace(",", " ").split()]

    if args.wandb_project is None:
        args.wandb_project = f"{args.baseline}-downstream"

    return args


def _apply_sweep_config(args: argparse.Namespace) -> None:
    cfg = wandb.config
    sweep_keys = (
        "lr", "dropout", "meta_embed_dim", "weight_decay", "loss",
        "class_weighting", "label_smoothing", "focal_gamma", "cb_beta",
        "ldam_max_margin", "ldam_scale", "batch_size", "use_projected_features",
    )
    for key in sweep_keys:
        if key in cfg:
            setattr(args, key, cfg[key])
    if "hidden_dims" in cfg:
        args.hidden_dims = list(cfg["hidden_dims"])


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Baseline: {args.baseline}")

    # ── W&B ──────────────────────────────────────────────────────────────────
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

    # ── Encoder ───────────────────────────────────────────────────────────────
    encoder, preprocess_train, preprocess_val, embed_dim = build_encoder(args, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    with open(args.splits, encoding="utf-8") as fh:
        splits = json.load(fh)

    age_sex_lookup = load_age_sex_lookup(Path(args.excel))

    lookup_keys = list(age_sex_lookup.keys())
    split_stems = [Path(s["image"]).stem for s in splits["downstream_train"][:5]]
    print(f"Lookup sample keys : {lookup_keys[:5]}")
    print(f"Split sample stems : {split_stems}")
    n_matches = sum(1 for s in splits["downstream_train"] if Path(s["image"]).stem in age_sex_lookup)
    print(f"Matching train samples: {n_matches} / {len(splits['downstream_train'])}")

    if n_matches == 0:
        raise RuntimeError(
            "No age/sex entries matched any split sample. "
            "Check that Excel column A filenames match the image stems in splits.json.\n"
            f"  Lookup example: {lookup_keys[:3]}\n"
            f"  Split example:  {split_stems[:3]}"
        )

    train_ages = [
        age_sex_lookup[Path(s["image"]).stem][0]
        for s in splits["downstream_train"]
        if Path(s["image"]).stem in age_sex_lookup
    ]
    age_mean = float(np.mean(train_ages))
    age_std  = float(np.std(train_ages))
    print(f"Age stats (train): mean={age_mean:.1f}, std={age_std:.1f}")

    train_ds = DownstreamDataset(splits["downstream_train"], age_sex_lookup, age_mean, age_std,
                                  preprocess_train, args.use_mask)
    val_ds   = DownstreamDataset(splits["downstream_val"],   age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   args.use_mask)
    test_ds  = DownstreamDataset(splits["test"],             age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   args.use_mask)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    use_pin       = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}

    train_loader    = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader_raw  = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader_raw = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Pre-compute val/test embeddings ───────────────────────────────────────
    print("Pre-computing val/test embeddings...")
    val_emb,  val_age,  val_sex,  val_lbl  = extract_embeddings(encoder, val_loader_raw,  device)
    test_emb, test_age, test_sex, test_lbl = extract_embeddings(encoder, test_loader_raw, device)

    embed_dim = val_emb.shape[1]
    print(f"Embedding dim: {embed_dim}")

    # ── Class weighting ───────────────────────────────────────────────────────
    train_labels_all = torch.tensor(
        [LABEL_TO_IDX[s["label"]] for s in train_ds.samples], dtype=torch.long
    )
    label_counts  = torch.bincount(train_labels_all, minlength=NUM_CLASSES).float()
    class_weights = compute_class_weights(
        label_counts,
        num_classes=NUM_CLASSES,
        mode=args.class_weighting,
        beta=args.cb_beta,
        device=device,
    )
    print(f"Class counts (train): { {IDX_TO_LABEL[i]: int(label_counts[i]) for i in range(NUM_CLASSES)} }")
    if class_weights is None:
        print("Class weights (train): none")
    else:
        print(f"Class weights (train): { {IDX_TO_LABEL[i]: float(class_weights[i]) for i in range(NUM_CLASSES)} }")
    print(f"Loss: {args.loss} | class_weighting={args.class_weighting}")

    val_emb_ds  = EmbeddingDataset(val_emb,  val_age,  val_sex,  val_lbl)
    test_emb_ds = EmbeddingDataset(test_emb, test_age, test_sex, test_lbl)

    val_loader  = DataLoader(val_emb_ds,  batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_emb_ds, batch_size=args.batch_size, shuffle=False)

    # ── MLP, loss, optimiser ──────────────────────────────────────────────────
    mlp       = MalignancyMLP(embed_dim, args.hidden_dims, args.dropout, args.meta_embed_dim).to(device)
    criterion = build_classification_loss(args, label_counts, NUM_CLASSES, class_weights, device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir) / f"{args.baseline}_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id    = (wandb.run.id if use_wandb and wandb.run else None) or "local"
    ckpt_path = out_dir / f"best_mlp_{run_id}.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"\nTraining MLP for {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        train_loss                               = train_one_epoch(mlp, encoder, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(mlp, val_loader, criterion, device)
        val_bal_acc      = balanced_accuracy_score(val_labels, val_preds)
        val_combined_acc = 0.5 * val_acc + 0.5 * val_bal_acc

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | "
            f"val_bal_acc={val_bal_acc:.3f} | val_combined_acc={val_combined_acc:.3f}"
        )
        if use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/balanced_acc": val_bal_acc,
                "val/combined_acc": val_combined_acc,
            }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "mlp_state_dict": mlp.state_dict(), "val_loss": val_loss},
                       ckpt_path)

    # ── Test evaluation ───────────────────────────────────────────────────────
    mlp.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["mlp_state_dict"]
    )
    test_loss, test_acc, test_preds, test_labels = evaluate(mlp, test_loader, criterion, device)

    label_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Loss: {test_loss:.4f}  |  Accuracy: {test_acc:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(test_labels, test_preds):.3f}")
    print(f"Macro F1: {f1_score(test_labels, test_preds, average='macro'):.3f}")
    print()
    print(classification_report(
        test_labels, test_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=label_names,
        digits=3, zero_division=0,
    ))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(
        confusion_matrix(test_labels, test_preds, labels=list(range(NUM_CLASSES))),
        index=label_names, columns=label_names,
    ).to_string())

    if use_wandb:
        test_bal_acc = balanced_accuracy_score(test_labels, test_preds)
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
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main(parse_args())
