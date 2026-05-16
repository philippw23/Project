"""LACE downstream malignancy classification (v1 and v2).

--version v1: Frozen SharedViT CLS token (768-dim, L2-normalised) → MalignancyMLP
--version v2: Frozen SharedViT + MaskTokenModule → LACEv2Classifier head

Splits are loaded from a splits.json produced during LACE pretraining; age and sex
are read directly from the split samples (no separate Excel lookup required).

Usage (v1):
    python src/lace_downstream.py \\
        --version v1 \\
        --checkpoint results/lace_pretrain/.../best_checkpoint.pt \\
        --splits    results/lace_pretrain/.../splits.json

Usage (v2):
    python src/lace_downstream.py \\
        --version v2 \\
        --checkpoint results/lace_v2_pretrain/.../best_checkpoint.pt \\
        --splits    results/lace_v2_pretrain/.../splits.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.data.datasets import (DownstreamDataset, EmbeddingDataset,
                                       IDX_TO_LABEL, LABEL_TO_IDX, NUM_CLASSES)
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.classifier import MalignancyMLP
from biomedclip.utils.misc import DEFAULT_OUT_DIR
from LACE.data.transforms import build_train_transform_lace
from LACE.models.downstream import LACEv2Classifier
from LACE.models.encoders import SharedViT
from LACE.models.mask_tokens import MaskTokenModule


class _V2HeadWrapper(nn.Module):
    """Wraps LACEv2Classifier's trainable components for eval on pre-computed representations.

    Shares parameters with the classifier — training this wrapper is equivalent to
    training the classifier head directly.
    """

    def __init__(self, classifier: LACEv2Classifier) -> None:
        super().__init__()
        self.age_emb = classifier.age_emb
        self.sex_emb = classifier.sex_emb
        self.head    = classifier.head

    def forward(
        self,
        lesion_repr: torch.Tensor,
        age: torch.Tensor,
        sex: torch.Tensor,
    ) -> torch.Tensor:
        if age.dim() == 1:
            age = age.unsqueeze(-1)
        age_feat   = self.age_emb(age.float())
        sex_feat   = self.sex_emb(sex.long())
        metric_emb = torch.cat([age_feat, sex_feat], dim=-1)
        return self.head(torch.cat([lesion_repr, metric_emb], dim=-1))


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_vit(checkpoint: dict, device: torch.device) -> SharedViT:
    lora_cfg = checkpoint.get("lora_config", {})
    if not lora_cfg:
        raise RuntimeError("Checkpoint has no 'lora_config'. Was it produced by LACE pretraining?")
    vit = SharedViT(
        lora_layers=lora_cfg["lora_layers"],
        r=lora_cfg["lora_r"],
        alpha=lora_cfg["lora_alpha"],
        embed_dim=lora_cfg.get("embed_dim", 256),
    )
    vit.load_state_dict(checkpoint["vit_state"])
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit.to(device)


def build_v1_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SharedViT, MalignancyMLP, object, object]:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vit  = _load_vit(ckpt, device)
    vit.eval()

    preprocess_val   = vit.preprocess_val
    preprocess_train = build_train_transform_lace(preprocess_val)

    mlp = MalignancyMLP(
        embed_dim=768,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        meta_embed_dim=args.meta_embed_dim,
    ).to(device)

    print(f"Loaded v1 checkpoint (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', float('nan')):.4f})")
    return vit, mlp, preprocess_train, preprocess_val


def build_v2_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[LACEv2Classifier, _V2HeadWrapper, object, object]:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    vit = _load_vit(ckpt, device)
    preprocess_val   = vit.preprocess_val
    preprocess_train = build_train_transform_lace(preprocess_val)

    n_mask_tokens = ckpt.get("n_mask_tokens", 16)
    mask_module   = MaskTokenModule(n_tokens=n_mask_tokens)
    mask_module.load_state_dict(ckpt["mask_module_state"])

    classifier = LACEv2Classifier(
        mask_module=mask_module,
        vit=vit,
        n_meta_dim=args.meta_embed_dim,
        n_classes=NUM_CLASSES,
    ).to(device)

    head_wrapper = _V2HeadWrapper(classifier)

    print(
        f"Loaded v2 checkpoint (epoch {ckpt.get('epoch', '?')}, "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f}, "
        f"n_mask_tokens={n_mask_tokens})"
    )
    return classifier, head_wrapper, preprocess_train, preprocess_val


# ── Representation extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_v1_embeddings(
    vit: SharedViT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract L2-normalised 768-dim CLS features for the frozen v1 ViT."""
    vit.eval()
    all_emb, all_age, all_sex, all_lbl = [], [], [], []
    for batch in loader:
        cls_feat, _ = vit.forward_all(batch["image"].to(device))  # [B, 768]
        all_emb.append(F.normalize(cls_feat, dim=-1).cpu())
        all_age.append(batch["age"])
        all_sex.append(batch["sex"])
        all_lbl.append(batch["label"])
    return torch.cat(all_emb), torch.cat(all_age), torch.cat(all_sex), torch.cat(all_lbl)


@torch.no_grad()
def extract_v2_representations(
    vit: SharedViT,
    mask_module: MaskTokenModule,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract 768-dim lesion representations (mean-pooled mask features) for v2."""
    vit.eval()
    mask_module.eval()
    all_repr, all_age, all_sex, all_lbl = [], [], [], []
    for batch in loader:
        _, patch_feat = vit.forward_all(batch["image"].to(device))  # [B, 196, 768]
        _, _, mask_feats = mask_module(patch_feat)                   # [B, N, 768]
        all_repr.append(mask_feats.mean(dim=1).cpu())                # [B, 768]
        all_age.append(batch["age"])
        all_sex.append(batch["sex"])
        all_lbl.append(batch["label"])
    return torch.cat(all_repr), torch.cat(all_age), torch.cat(all_sex), torch.cat(all_lbl)


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_one_epoch_v1(
    mlp: MalignancyMLP,
    vit: SharedViT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    mlp.train()
    vit.eval()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        age    = batch["age"].to(device)
        sex    = batch["sex"].to(device)
        lbl    = batch["label"].to(device)

        with torch.no_grad():
            cls_feat, _ = vit.forward_all(images)
            cls_feat    = F.normalize(cls_feat, dim=-1)

        optimizer.zero_grad()
        loss = criterion(mlp(cls_feat, age, sex), lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_one_epoch_v2(
    classifier: LACEv2Classifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    classifier.train()
    classifier.vit.eval()
    classifier.mask_module.eval()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        age    = batch["age"].unsqueeze(-1).to(device)  # [B, 1]
        sex    = batch["sex"].long().to(device)
        lbl    = batch["label"].to(device)

        optimizer.zero_grad()
        loss = criterion(classifier(images, age, sex), lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate on a pre-computed EmbeddingDataset yielding (emb, age, sex, lbl) tuples."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for emb, age, sex, lbl in loader:
        emb, age, sex, lbl = emb.to(device), age.to(device), sex.to(device), lbl.to(device)
        logits = model(emb, age, sex)
        total_loss += criterion(logits, lbl).item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(lbl.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / len(loader), float((preds == labels).mean()), preds, labels


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  LACE version: {args.version}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        warnings.warn("--wandb set but wandb is not installed. Skipping.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            config=vars(args),
        )

    # ── Build model ───────────────────────────────────────────────────────────
    if args.version == "v1":
        vit, mlp, preprocess_train, preprocess_val = build_v1_model(args, device)
        trainable_model = mlp
    else:
        classifier, head_wrapper, preprocess_train, preprocess_val = build_v2_model(args, device)
        trainable_model = head_wrapper

    # ── Load splits ───────────────────────────────────────────────────────────
    with open(args.splits, encoding="utf-8") as fh:
        splits = json.load(fh)

    all_samples    = splits["train"] + splits["val"] + splits["test"]
    age_sex_lookup = {Path(s["image"]).stem: (s["age"], s["sex"]) for s in all_samples}

    train_ages = [s["age"] for s in splits["train"]]
    age_mean   = float(np.mean(train_ages))
    age_std    = float(np.std(train_ages))
    print(f"Age stats (train): mean={age_mean:.1f}, std={age_std:.1f}")

    train_ds = DownstreamDataset(splits["train"], age_sex_lookup, age_mean, age_std,
                                  preprocess_train, use_mask=False)
    val_ds   = DownstreamDataset(splits["val"],   age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   use_mask=False)
    test_ds  = DownstreamDataset(splits["test"],  age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   use_mask=False)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    use_pin       = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}
    train_loader    = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader_raw  = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader_raw = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Pre-compute val/test representations ──────────────────────────────────
    print("Pre-computing val/test representations...")
    if args.version == "v1":
        val_emb,  val_age,  val_sex,  val_lbl  = extract_v1_embeddings(vit, val_loader_raw,  device)
        test_emb, test_age, test_sex, test_lbl = extract_v1_embeddings(vit, test_loader_raw, device)
    else:
        val_emb,  val_age,  val_sex,  val_lbl  = extract_v2_representations(
            classifier.vit, classifier.mask_module, val_loader_raw,  device)
        test_emb, test_age, test_sex, test_lbl = extract_v2_representations(
            classifier.vit, classifier.mask_module, test_loader_raw, device)

    val_loader  = DataLoader(EmbeddingDataset(val_emb,  val_age,  val_sex,  val_lbl),
                             batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EmbeddingDataset(test_emb, test_age, test_sex, test_lbl),
                             batch_size=args.batch_size, shuffle=False)

    # ── Class weights and loss ────────────────────────────────────────────────
    train_labels_all = torch.tensor(
        [LABEL_TO_IDX[s["label"]] for s in train_ds.samples], dtype=torch.long
    )
    label_counts  = torch.bincount(train_labels_all, minlength=NUM_CLASSES).float()
    class_weights = compute_class_weights(
        label_counts, num_classes=NUM_CLASSES, mode=args.class_weighting,
        beta=args.cb_beta, device=device,
    )
    criterion = build_classification_loss(args, label_counts, NUM_CLASSES, class_weights, device)
    print(f"Class counts: { {IDX_TO_LABEL[i]: int(label_counts[i]) for i in range(NUM_CLASSES)} }")
    print(f"Loss: {args.loss} | weighting: {args.class_weighting}")

    optimizer = torch.optim.AdamW(
        trainable_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    out_dir = Path(args.out_dir) / f"lace_{args.version}_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id    = (wandb.run.id if use_wandb and wandb.run else None) or "local"
    ckpt_path = out_dir / f"best_{run_id}.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})\n")

    for epoch in range(1, args.epochs + 1):
        if args.version == "v1":
            train_loss = train_one_epoch_v1(mlp, vit, train_loader, optimizer, criterion, device)
        else:
            train_loss = train_one_epoch_v2(classifier, train_loader, optimizer, criterion, device)

        val_loss, val_acc, val_preds, val_labels = evaluate(trainable_model, val_loader, criterion, device)
        val_bal_acc      = balanced_accuracy_score(val_labels, val_preds)
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
                "val/balanced_acc": val_bal_acc,
                "val/combined_acc": val_combined_acc,
            }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "version": args.version,
                "val_loss": val_loss,
                "model_state_dict": trainable_model.state_dict(),
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    trainable_model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    )
    test_loss, test_acc, test_preds, test_labels = evaluate(trainable_model, test_loader, criterion, device)
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
    import pandas as pd
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


# ── Argparse ─────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LACE downstream malignancy classification (v1 and v2)."
    )

    parser.add_argument("--version",    required=True, choices=["v1", "v2"],
                        help="LACE version: v1 uses ViT CLS token, v2 uses MaskTokenModule lesion repr.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a LACE pretrain checkpoint (.pt).")
    parser.add_argument("--splits",     required=True,
                        help="Path to splits.json produced by LACE pretraining.")
    parser.add_argument("--out_dir",    default=str(DEFAULT_OUT_DIR))

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--patience",       type=int,   default=10)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--dropout",        type=float, default=0.3,
                        help="Dropout for the v1 MalignancyMLP.")
    parser.add_argument("--meta_embed_dim", type=int,   default=32)
    parser.add_argument("--hidden_dims",    type=int,   nargs="+", default=[256, 128],
                        help="Hidden layer widths for the v1 MLP.")
    parser.add_argument("--weight_decay",   type=float, default=0.01)

    # ── Loss ──────────────────────────────────────────────────────────────────
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
    parser.add_argument("--wandb_project", default="lace-downstream")
    parser.add_argument("--wandb_run",     default=None)
    parser.add_argument("--wandb_entity",  default=None)

    return parser.parse_args(argv)
