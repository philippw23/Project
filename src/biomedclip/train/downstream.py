"""Downstream malignancy classification using the pretrained BiomedCLIP image encoder.

Loads the LoRA-adapted image encoder from a checkpoint produced by
biomedclip_pretrain.py, freezes it, and trains a small MLP classifier on top.

MLP input : [image_embedding (512 projected | 768 pre-projection) | age (1, z-scored) | sex (1, binary)]
MLP output: 3-class logits  (benign=0 / intermediate=1 / malignant=2)

Usage:
    python src/biomedclip_downstream.py \\
        --checkpoint results/biomedclip_pretrain/best_r1_checkpoint.pt \\
        --splits     results/biomedclip_pretrain/splits.json \\
        --excel      data/metadata.xlsx \\
        --use_mask   --epochs 50
"""

from __future__ import annotations

import argparse
import json
import warnings
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

from biomedclip.utils.misc import ROOT_DIR, MODEL_TAG, DEFAULT_EXCEL, DEFAULT_OUT_DIR
from biomedclip.data.transforms import build_train_transform
from biomedclip.data.datasets import (
    DownstreamDataset, EmbeddingDataset, LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES,
)
from biomedclip.data.splits import load_age_sex_lookup
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.lora import inject_lora
from biomedclip.models.classifier import MalignancyMLP, extract_embeddings

DEFAULT_SPLITS     = ROOT_DIR / "results" / "biomedclip_pretrain" / "splits.json"
DEFAULT_CHECKPOINT = ROOT_DIR / "results" / "biomedclip_pretrain" / "best_r1_checkpoint.pt"


def train_one_epoch(
    mlp: nn.Module,
    encoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train the MLP for one pass over the training split.

    The encoder stays in eval mode so batch-norm statistics and dropout are
    fixed, but it re-encodes each batch from raw images so that the random
    augmentations in preprocess_train are applied fresh on every epoch.
    """
    mlp.train()
    encoder.eval()
    total_loss = 0.0
    for batch in loader:
        # Unpack the batch dict from DownstreamDataset.
        images = batch["image"].to(device)
        age    = batch["age"].to(device)
        sex    = batch["sex"].to(device)
        lbl    = batch["label"].to(device)

        # Encode images without tracking gradients — encoder is frozen.
        with torch.no_grad():
            emb = F.normalize(encoder(images), dim=-1)  # (B, embed_dim), unit norm

        optimizer.zero_grad()
        logits = mlp(emb, age, sex)   # (B, 3) raw class logits
        loss   = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Return mean loss over all batches.
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    mlp: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate the MLP on a pre-computed embedding split.

    The loader yields (emb, age, sex, label) tuples from an EmbeddingDataset,
    so the encoder is not called here.  Returns loss, accuracy, and the raw
    prediction/label arrays for downstream metric computation.
    """
    mlp.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for emb, age, sex, lbl in loader:
        emb, age, sex, lbl = emb.to(device), age.to(device), sex.to(device), lbl.to(device)
        logits = mlp(emb, age, sex)
        total_loss += criterion(logits, lbl).item()
        # argmax over the 3 class logits gives the predicted class index.
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(lbl.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc    = (preds == labels).mean()
    return total_loss / len(loader), float(acc), preds, labels


def _str_to_bool(v: str) -> bool:
    """Accept explicit True/False strings from wandb sweep agents alongside bare flags."""
    return str(v).lower() not in ("false", "0", "no", "none")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream malignancy classifier on top of BiomedCLIP image encoder."
    )
    # ── Checkpoint / splits ──────────────────────────────────────────────────
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--freezed_biomedclip", action="store_true",
                        help="Use the vanilla BiomedCLIP encoder without loading a "
                             "fine-tuned checkpoint (no LoRA injection).")
    parser.add_argument("--splits",      default=str(DEFAULT_SPLITS))
    parser.add_argument("--excel",       default=str(DEFAULT_EXCEL))
    parser.add_argument("--out_dir",     default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--use_mask",    action="store_true")

    # ── Encoder feature choice ───────────────────────────────────────────────
    # nargs='?' with const=True lets this work both as a bare flag (--use_projected_features)
    # and with an explicit value (--use_projected_features=False) as wandb sweep agents pass it.
    parser.add_argument("--use_projected_features", nargs="?", const=True,
                        type=_str_to_bool, default=False,
                        help="Use the 512-dim projected CLIP embedding instead of the "
                             "768-dim pre-projection ViT features (default: pre-projection).")

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--meta_embed_dim", type=int, default=16,
                        help="Projection dim for age/sex before fusion (default: %(default)s)")
    parser.add_argument("--hidden_dims", type=str, nargs="+", default=[256, 128])
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # ── Loss function ─────────────────────────────────────────────────────────
    parser.add_argument("--loss", default="ce",
                        choices=["ce", "wce", "ce_smooth", "focal", "cb_focal", "ldam", "balanced_softmax"],
                        help="Classification loss for the downstream head.")
    parser.add_argument("--class_weighting", default="sqrt",
                        choices=["none", "inverse", "sqrt", "effective"],
                        help="Class weighting scheme used by CE/Focal/LDAM losses.")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for ce_smooth/focal/cb_focal.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for focal losses.")
    parser.add_argument("--cb_beta", type=float, default=0.99,
                        help="Beta for effective-number class weights.")
    parser.add_argument("--ldam_max_margin", type=float, default=0.5,
                        help="Maximum class margin for LDAM loss.")
    parser.add_argument("--ldam_scale", type=float, default=30.0,
                        help="Logit scale for LDAM loss.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--wandb",       action="store_true")
    parser.add_argument("--wandb_project", default="biomedclip-downstream")
    parser.add_argument("--wandb_run",   default=None)
    parser.add_argument("--wandb_entity", default=None,
                        help="W&B team/entity name (default: personal account)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run as wandb sweep agent (hyperparams come from wandb.config)")
    args = parser.parse_args(argv)

    # Normalise hidden_dims: accept both '--hidden_dims 256 128' and '--hidden_dims [256,128]'.
    raw = " ".join(str(x) for x in args.hidden_dims)
    args.hidden_dims = [int(x) for x in raw.strip("[]").replace(",", " ").split()]
    return args


def _apply_sweep_config(args: argparse.Namespace) -> None:
    """Overwrite args with values from wandb.config when running as sweep agent."""
    cfg = wandb.config
    # Each key in cfg overrides the matching argparse attribute.
    for key in (
        "lr", "dropout", "meta_embed_dim", "weight_decay", "loss",
        "class_weighting", "label_smoothing", "focal_gamma", "cb_beta",
        "ldam_max_margin", "ldam_scale", "batch_size", "use_projected_features",
    ):
        if key in cfg:
            setattr(args, key, cfg[key])
    if "hidden_dims" in cfg:
        args.hidden_dims = list(cfg["hidden_dims"])


def main(args: argparse.Namespace) -> None:
    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── W&B initialisation ────────────────────────────────────────────────────
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
        # Sweep agents receive hyperparams from the sweep controller via wandb.config.
        if args.sweep:
            _apply_sweep_config(args)

    # ── Model loading ─────────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_TAG}")
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)

    if args.freezed_biomedclip:
        # Use vanilla BiomedCLIP weights without any fine-tuning (baseline).
        print("Using vanilla BiomedCLIP encoder (no checkpoint, no LoRA).")
    else:
        # Load the LoRA-adapted checkpoint produced by pretrain.py.
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        # Recover the LoRA rank/alpha/layers used during pretraining so the
        # model architecture matches the saved state dict exactly.
        lora_cfg = ckpt.get("lora_config") or {}
        if not lora_cfg:
            raise RuntimeError(
                f"Checkpoint '{args.checkpoint}' has no 'lora_config'. "
                "Re-run pretraining with the current biomedclip_pretrain.py which saves this field."
            )
        lora_layers = lora_cfg["lora_layers"]
        lora_r      = lora_cfg["lora_r"]
        lora_alpha  = lora_cfg["lora_alpha"]
        print(f"LoRA config from checkpoint: layers={lora_layers}, r={lora_r}, alpha={lora_alpha}")

        # inject_lora freezes all params and adds trainable LoRA branches,
        # then load_state_dict fills in both frozen and LoRA weights.
        inject_lora(model, lora_layers, lora_r, lora_alpha)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # ── Encoder selection ─────────────────────────────────────────────────────
    # model.visual       → full BiomedCLIP visual tower including the 768→512 projection head
    # model.visual.trunk → ViT backbone only, output dimension 768
    if args.use_projected_features:
        encoder = model.visual.to(device)         # 512-dim CLIP embedding space
        print("Encoder: projected features (512-dim)")
    else:
        encoder = model.visual.trunk.to(device)   # 768-dim ViT CLS token
        print("Encoder: pre-projection ViT features (768-dim)")

    # Freeze all encoder parameters — the downstream MLP is the only trainable part.
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    # ── Data preparation ──────────────────────────────────────────────────────
    # preprocess_train adds random flips/crops on top of preprocess_val for augmentation.
    preprocess_train = build_train_transform(preprocess_val)

    with open(args.splits, encoding="utf-8") as fh:
        splits = json.load(fh)

    # age_sex_lookup maps image stem → (age_raw, sex_binary).
    age_sex_lookup = load_age_sex_lookup(Path(args.excel))

    # Verify that the Excel lookup keys match the image stems in splits.json.
    lookup_keys  = list(age_sex_lookup.keys())
    split_stems  = [Path(s["image"]).stem for s in splits["downstream_train"][:5]]
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

    # Compute z-score normalisation statistics from the training split only
    # so that val/test ages are normalised with the same scale.
    train_ages = [
        age_sex_lookup[Path(s["image"]).stem][0]
        for s in splits["downstream_train"]
        if Path(s["image"]).stem in age_sex_lookup
    ]
    age_mean = float(np.mean(train_ages))
    age_std  = float(np.std(train_ages))
    print(f"Age stats (train): mean={age_mean:.1f}, std={age_std:.1f}")

    # Train split uses augmented transforms; val/test use clean val transforms.
    train_ds = DownstreamDataset(splits["downstream_train"], age_sex_lookup, age_mean, age_std,
                                  preprocess_train, args.use_mask)
    val_ds   = DownstreamDataset(splits["downstream_val"],   age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   args.use_mask)
    test_ds  = DownstreamDataset(splits["test"],             age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   args.use_mask)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    use_pin       = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}

    # train_loader yields raw images (encoder re-encodes each epoch for augmentation).
    # val_loader_raw / test_loader_raw are only used once to pre-compute embeddings.
    train_loader    = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader_raw  = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader_raw = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Pre-compute val/test embeddings ───────────────────────────────────────
    # Val and test sets use no augmentation, so embeddings are fixed across epochs.
    # Pre-computing them once avoids running the encoder on every eval call.
    print("Pre-computing val/test embeddings...")
    val_emb,  val_age,  val_sex,  val_lbl  = extract_embeddings(encoder, val_loader_raw,  device)
    test_emb, test_age, test_sex, test_lbl = extract_embeddings(encoder, test_loader_raw, device)

    # Infer embed_dim from the actual embedding shape so the MLP adapts automatically
    # to either 512 (projected) or 768 (pre-projection) without any hard-coded value.
    embed_dim = val_emb.shape[1]
    print(f"Embedding dim: {embed_dim}")

    # ── Class weighting ───────────────────────────────────────────────────────
    # Count labels in the training split to compute per-class weights for the loss.
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

    # Wrap the pre-computed embedding arrays in EmbeddingDataset so the eval
    # loop can use a standard DataLoader without touching the encoder.
    val_emb_ds  = EmbeddingDataset(val_emb,  val_age,  val_sex,  val_lbl)
    test_emb_ds = EmbeddingDataset(test_emb, test_age, test_sex, test_lbl)

    val_loader  = DataLoader(val_emb_ds,  batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_emb_ds, batch_size=args.batch_size, shuffle=False)

    # ── MLP, loss, optimiser ──────────────────────────────────────────────────
    mlp       = MalignancyMLP(embed_dim, args.hidden_dims, args.dropout, args.meta_embed_dim).to(device)
    criterion = build_classification_loss(args, label_counts, NUM_CLASSES, class_weights, device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir) / "biomedclip_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the W&B run ID in the checkpoint filename to prevent collisions across sweep runs.
    run_id    = (wandb.run.id if use_wandb and wandb.run else None) or "local"
    ckpt_path = out_dir / f"best_mlp_{run_id}.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"\nTraining MLP for {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        # Train one epoch on raw images (encoder re-encodes with augmentation each time).
        train_loss              = train_one_epoch(mlp, encoder, train_loader, optimizer, criterion, device)
        # Evaluate on pre-computed fixed embeddings.
        val_loss, val_acc, val_preds, val_labels = evaluate(mlp, val_loader, criterion, device)
        val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
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

        # Save the MLP weights whenever validation loss improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            mlp_state = mlp.state_dict()
            torch.save({"epoch": epoch, "mlp_state_dict": mlp_state, "val_loss": val_loss},
                       ckpt_path)

    # ── Test evaluation ───────────────────────────────────────────────────────
    # Reload the best validation checkpoint before running the final test evaluation.
    mlp_state = torch.load(ckpt_path, map_location=device, weights_only=False)["mlp_state_dict"]
    mlp.load_state_dict(mlp_state)
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
        test_labels,
        test_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=label_names,
        digits=3,
        zero_division=0,
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
            "test/loss": test_loss,
            "test/acc": test_acc,
            "test/balanced_acc": test_bal_acc,
            "test/precision_macro": test_prec,
            "test/recall_macro": test_rec,
            "test/f1_macro": test_f1,
        }
        for i, name in enumerate(label_names):
            log_dict[f"test/precision_{name}"] = per_class_prec[i]
            log_dict[f"test/recall_{name}"] = per_class_rec[i]
            log_dict[f"test/f1_{name}"] = per_class_f1[i]
        wandb.log(log_dict)
        wandb.finish()

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main(parse_args())
