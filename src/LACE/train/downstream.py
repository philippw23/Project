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
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.data.datasets import (DownstreamDataset, EmbeddingDataset,
                                       IDX_TO_LABEL, LABEL_TO_IDX, NUM_CLASSES,
                                       IDX_TO_LABEL_BINARY, LABEL_TO_IDX_BINARY, NUM_CLASSES_BINARY)
from biomedclip.data.transforms import build_preprocess_val
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.classifier import LinearHead, MalignancyMLP
from biomedclip.utils.downstream_eval import require_binary_for_btxrd, report_eval
from biomedclip.utils.misc import DEFAULT_OUT_DIR
from LACE.data.transforms import build_train_transform_lace
from LACE.models.downstream import LACEv2Classifier
from LACE.models.encoders import SharedViT
from LACE.models.mask_tokens import MaskTokenDecoder, MaskPredictionHead


class _V2HeadWrapper(nn.Module):
    """Wraps LACEv2Classifier's trainable components for eval on pre-computed representations.

    Shares parameters with the classifier — training this wrapper is equivalent to
    training the classifier head directly.
    """

    def __init__(self, classifier: LACEv2Classifier) -> None:
        super().__init__()
        self.use_meta = classifier.use_meta
        self.age_emb  = classifier.age_emb  # None when use_meta=False
        self.sex_emb  = classifier.sex_emb  # None when use_meta=False
        self.head     = classifier.head

    def forward(
        self,
        lesion_repr: torch.Tensor,
        age: torch.Tensor,
        sex: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_meta:
            if age.dim() == 1:
                age = age.unsqueeze(-1)
            age_feat   = self.age_emb(age.float())
            sex_feat   = self.sex_emb(sex.long())
            metric_emb = torch.cat([age_feat, sex_feat], dim=-1)
            return self.head(torch.cat([lesion_repr, metric_emb], dim=-1))
        return self.head(lesion_repr)


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_vit(checkpoint: dict, device: torch.device) -> SharedViT:
    lora_cfg = checkpoint.get("lora_config", {})
    if not lora_cfg:
        raise RuntimeError("Checkpoint has no 'lora_config'. Was it produced by LACE pretraining?")
    # lora_cfg describes how the ViT was built during *pretraining*; it must be
    # replayed here so load_state_dict matches, but it says nothing about what
    # trains downstream. Stay quiet until after the freeze, then report the truth.
    vit = SharedViT(
        lora_layers=lora_cfg["lora_layers"],
        r=lora_cfg["lora_r"],
        alpha=lora_cfg["lora_alpha"],
        embed_dim=lora_cfg.get("embed_dim", 512),
        unfreeze_layers=lora_cfg.get("unfreeze_layers", 0),
        verbose=False,
    )
    vit.load_state_dict(checkpoint["vit_state"])
    for p in vit.parameters():
        p.requires_grad_(False)
    print(
        f"SharedViT: loaded pretrained weights — {vit.adapter_mode} — "
        f"now frozen for downstream | {vit.param_summary()}"
    )
    return vit.to(device)


def build_v1_model(
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> tuple[SharedViT, MalignancyMLP, object, object]:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vit  = _load_vit(ckpt, device)
    vit.eval()

    preprocess_val   = build_preprocess_val(vit.preprocess_val, getattr(args, "image_size", 224))
    preprocess_train = build_train_transform_lace(preprocess_val)

    if args.head == "linear":
        mlp = LinearHead(embed_dim=768, num_classes=num_classes).to(device)
    elif args.head == "mlp_no_meta":
        mlp = MalignancyMLP(
            embed_dim=768, hidden_dims=args.hidden_dims,
            dropout=args.dropout, use_meta=False, num_classes=num_classes,
        ).to(device)
    else:
        mlp = MalignancyMLP(
            embed_dim=768, hidden_dims=args.hidden_dims,
            dropout=args.dropout, meta_embed_dim=args.meta_embed_dim,
            num_classes=num_classes,
        ).to(device)

    print(f"Loaded v1 checkpoint (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', float('nan')):.4f})")
    return vit, mlp, preprocess_train, preprocess_val


def build_v2_model(
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> tuple[LACEv2Classifier, _V2HeadWrapper, object, object]:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    vit = _load_vit(ckpt, device)
    preprocess_val   = build_preprocess_val(vit.preprocess_val, getattr(args, "image_size", 224))
    preprocess_train = build_train_transform_lace(preprocess_val)

    n_mask_tokens = ckpt.get("n_mask_tokens", 4)
    mask_decoder = MaskTokenDecoder(
        n_tokens=n_mask_tokens,
        n_heads=ckpt.get("n_mask_heads", 8),
        sigma=ckpt.get("gauss_sigma", 1.5),
    )
    mask_decoder.load_state_dict(ckpt["mask_decoder_state"])
    mask_decoder.to(device)

    mask_head = MaskPredictionHead()
    mask_head.load_state_dict(ckpt["mask_head_state"])
    mask_head.to(device)

    use_meta     = args.head == "mlp"
    linear_head  = args.head == "linear"
    visual_mode  = getattr(args, "downstream_visual_mode", "cls_fg")
    sim_attn_tau = getattr(args, "sim_attn_tau", 0.07)

    classifier = LACEv2Classifier(
        vit=vit,
        mask_decoder=mask_decoder,
        mask_head=mask_head,
        n_meta_dim=args.meta_embed_dim,
        n_classes=num_classes,
        use_meta=use_meta,
        linear_head=linear_head,
        visual_mode=visual_mode,
        sim_attn_tau=sim_attn_tau,
    ).to(device)

    head_wrapper = _V2HeadWrapper(classifier)

    print(
        f"Loaded v2 checkpoint (epoch {ckpt.get('epoch', '?')}, "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f}, "
        f"n_mask_tokens={n_mask_tokens}, visual_mode={visual_mode})"
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
    classifier: LACEv2Classifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract visual representations for v2 using the classifier's _get_visual."""
    classifier.eval()
    all_repr, all_age, all_sex, all_lbl = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        repr_  = classifier._get_visual(images)
        all_repr.append(repr_.cpu())
        all_age.append(batch["age"])
        all_sex.append(batch["sex"])
        all_lbl.append(batch["label"])
    return torch.cat(all_repr), torch.cat(all_age), torch.cat(all_sex), torch.cat(all_lbl)


def _precompute_repr_loader(
    samples: list[dict],
    age_mean: float,
    age_std: float,
    preprocess_val,
    args: argparse.Namespace,
    extractor,
    device: torch.device,
    label_to_idx: dict[str, int],
) -> tuple[DataLoader, int]:
    """Build a DownstreamDataset for `samples`, extract frozen representations,
    and wrap them in an EmbeddingDataset DataLoader.

    age/sex are read from each sample dict; ages are normalised with the given
    (train-derived) `age_mean`/`age_std` so external sets like BTXRD use the same
    statistics as the fold's training data. `extractor(raw_loader)` returns
    (repr, age, sex, lbl).
    """
    lookup = {Path(s["image"]).stem: (s["age"], s["sex"]) for s in samples}
    ds = DownstreamDataset(
        samples, lookup, age_mean, age_std,
        preprocess_val, use_mask=args.use_mask, label_to_idx=label_to_idx,
    )
    raw_loader = DataLoader(
        ds, shuffle=False, batch_size=args.batch_size,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )
    emb, age, sex, lbl = extractor(raw_loader)
    loader = DataLoader(
        EmbeddingDataset(emb, age, sex, lbl),
        batch_size=args.batch_size, shuffle=False,
    )
    return loader, len(ds)


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
    classifier.mask_decoder.eval()
    classifier.mask_head.eval()
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


def _apply_sweep_config(args: argparse.Namespace) -> None:
    cfg = wandb.config
    for key in (
        "lr", "dropout", "meta_embed_dim", "weight_decay", "loss",
        "class_weighting", "label_smoothing", "focal_gamma", "cb_beta",
        "ldam_max_margin", "ldam_scale", "batch_size", "head",
    ):
        if key in cfg:
            setattr(args, key, cfg[key])
    if "hidden_dims" in cfg:
        args.hidden_dims = list(cfg["hidden_dims"])


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  LACE version: {args.version}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = (args.wandb or args.sweep) and WANDB_AVAILABLE
    if (args.wandb or args.sweep) and not WANDB_AVAILABLE:
        warnings.warn("--wandb/--sweep set but wandb is not installed. Skipping.")
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
    if args.binary:
        label_to_idx = LABEL_TO_IDX_BINARY
        idx_to_label = IDX_TO_LABEL_BINARY
        num_classes  = NUM_CLASSES_BINARY
        print("Mode: binary (benign vs malignant)")
    else:
        label_to_idx = LABEL_TO_IDX
        idx_to_label = IDX_TO_LABEL
        num_classes  = NUM_CLASSES

    # ── Build model ───────────────────────────────────────────────────────────
    if args.version == "v1":
        vit, mlp, preprocess_train, preprocess_val = build_v1_model(args, device, num_classes)
        trainable_model = mlp
    else:
        classifier, head_wrapper, preprocess_train, preprocess_val = build_v2_model(args, device, num_classes)
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
                                  preprocess_train, use_mask=args.use_mask, label_to_idx=label_to_idx)
    val_ds   = DownstreamDataset(splits["val"],   age_sex_lookup, age_mean, age_std,
                                  preprocess_val,   use_mask=args.use_mask, label_to_idx=label_to_idx)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}")

    use_pin       = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}
    train_loader    = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader_raw  = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # extractor closure over the frozen backbone (v1 ViT CLS or v2 classifier)
    if args.version == "v1":
        extractor = lambda ldr: extract_v1_embeddings(vit, ldr, device)
    else:
        extractor = lambda ldr: extract_v2_representations(classifier, ldr, device)

    # ── Pre-compute val (always) + test/BTXRD (only when requested) reprs ──────
    print("Pre-computing val representations...")
    val_emb, val_age, val_sex, val_lbl = extractor(val_loader_raw)
    val_loader = DataLoader(EmbeddingDataset(val_emb, val_age, val_sex, val_lbl),
                            batch_size=args.batch_size, shuffle=False)

    # Test/BTXRD are held out during sweeps: only built when --eval_test is set.
    test_loader = None
    btxrd_loader = None
    if args.eval_test:
        print("Pre-computing test representations...")
        test_loader, n_test = _precompute_repr_loader(
            splits["test"], age_mean, age_std, preprocess_val,
            args, extractor, device, label_to_idx,
        )
        print(f"Test samples: {n_test}")
        if args.btxrd_manifest:
            print(f"Pre-computing BTXRD representations from {args.btxrd_manifest}...")
            with open(args.btxrd_manifest, encoding="utf-8") as fh:
                btxrd_samples = json.load(fh)
            btxrd_loader, n_btxrd = _precompute_repr_loader(
                btxrd_samples, age_mean, age_std, preprocess_val,
                args, extractor, device, label_to_idx,
            )
            print(f"BTXRD samples: {n_btxrd}")

    # ── Class weights and loss ────────────────────────────────────────────────
    train_labels_all = torch.tensor(
        [label_to_idx[s["label"]] for s in train_ds.samples], dtype=torch.long
    )
    label_counts  = torch.bincount(train_labels_all, minlength=num_classes).float()
    class_weights = compute_class_weights(
        label_counts, num_classes=num_classes, mode=args.class_weighting,
        beta=args.cb_beta, device=device,
    )
    criterion = build_classification_loss(args, label_counts, num_classes, class_weights, device)
    print(f"Class counts: { {idx_to_label[i]: int(label_counts[i]) for i in range(num_classes)} }")
    print(f"Loss: {args.loss} | weighting: {args.class_weighting}")

    optimizer = torch.optim.AdamW(
        trainable_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    out_dir = Path(args.out_dir) / f"lace_{args.version}_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Identifiable, non-clobbering checkpoint name: timestamp (+ wandb id when
    # present). --run_name overrides for a fixed, human-chosen name.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_id  = wandb.run.id if use_wandb and wandb.run else None
    tag       = args.run_name or (f"{run_stamp}_{wandb_id}" if wandb_id else run_stamp)
    ckpt_path = out_dir / f"best_{tag}.pt"
    print(f"Head checkpoint → {ckpt_path}")

    # ── Training loop ─────────────────────────────────────────────────────────
    maximize_metric   = args.early_stopping_metric == "val_bal_acc"
    best_metric       = -float("inf") if maximize_metric else float("inf")
    best_val_f1_macro = 0.0
    patience_counter  = 0
    print(f"\nTraining for {args.epochs} epochs "
          f"(patience={args.patience}, monitor={args.early_stopping_metric})\n")

    for epoch in range(1, args.epochs + 1):
        if args.version == "v1":
            train_loss = train_one_epoch_v1(mlp, vit, train_loader, optimizer, criterion, device)
        else:
            train_loss = train_one_epoch_v2(classifier, train_loader, optimizer, criterion, device)

        val_loss, val_acc, val_preds, val_labels = evaluate(trainable_model, val_loader, criterion, device)
        val_bal_acc       = balanced_accuracy_score(val_labels, val_preds)
        val_f1_macro      = f1_score(val_labels, val_preds, average="macro")
        val_weighted_prec, val_weighted_rec, _, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="weighted", zero_division=0
        )
        val_combined_acc = 0.5 * val_acc + 0.5 * val_bal_acc
        best_val_f1_macro = max(best_val_f1_macro, val_f1_macro)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | val_bal_acc={val_bal_acc:.3f} | "
            f"val_f1_macro={val_f1_macro:.3f} | val_combined_acc={val_combined_acc:.3f}"
        )
        if use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/balanced_acc":        val_bal_acc,
                "val/f1_macro":            val_f1_macro,
                "val/precision_weighted":  val_weighted_prec,
                "val/recall_weighted":     val_weighted_rec,
                "val/combined_acc":        val_combined_acc,
            }, step=epoch)
            # summary metric the sweep ranks configs by (best epoch's macro-F1)
            wandb.run.summary["val/best_f1_macro"] = best_val_f1_macro

        current_metric = val_bal_acc if maximize_metric else val_loss
        improved = current_metric > best_metric if maximize_metric else current_metric < best_metric
        if improved:
            best_metric      = current_metric
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "version": args.version,
                "val_loss": val_loss,
                "val_bal_acc": val_bal_acc,
                "run_stamp": run_stamp,
                "wandb_id": wandb_id,
                "model_state_dict": trainable_model.state_dict(),
                # everything needed to rebuild + evaluate this head standalone
                # (see lace_downstream_eval.py): architecture/loss hyperparameters,
                # the backbone checkpoint path, and the train age-normalization.
                "args": vars(args),
                "age_mean": age_mean,
                "age_std":  age_std,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    # ── Restore best checkpoint ────────────────────────────────────────────────
    trainable_model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    )

    # Metrics returned to callers (e.g. the k-fold CV orchestrator).
    results: dict = {
        f"val/best_{args.early_stopping_metric}": best_metric,
        "val/best_f1_macro": best_val_f1_macro,
    }

    # ── Held-out evaluation (only when --eval_test) ───────────────────────────
    log_dict: dict = {}
    if args.eval_test and test_loader is not None:
        test_loss, _, test_preds, test_labels = evaluate(
            trainable_model, test_loader, criterion, device
        )
        test_metrics = report_eval(
            "TEST", test_preds, test_labels, test_loss,
            idx_to_label, num_classes, prefix="test",
        )
        results.update(test_metrics)
        log_dict.update(test_metrics)
        # raw per-sample predictions for pooled out-of-fold CV scoring
        # (consumed by lace_downstream_cv.py; ignored by single-run callers)
        results["test/_preds"]  = test_preds.tolist()
        results["test/_labels"] = test_labels.tolist()

    if args.eval_test and btxrd_loader is not None:
        btxrd_loss, _, btxrd_preds, btxrd_labels = evaluate(
            trainable_model, btxrd_loader, criterion, device
        )
        btxrd_metrics = report_eval(
            "BTXRD (external)", btxrd_preds, btxrd_labels, btxrd_loss,
            idx_to_label, num_classes, prefix="btxrd",
        )
        results.update(btxrd_metrics)
        log_dict.update(btxrd_metrics)

    if use_wandb:
        if log_dict:
            wandb.log(log_dict)
        wandb.finish()

    print(f"\nBest {args.early_stopping_metric}: {best_metric:.4f} | best val macro-F1: {best_val_f1_macro:.4f}")
    print(f"Head checkpoint saved to: {ckpt_path}")
    return results


# ── Argparse ─────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LACE downstream malignancy classification (v1 and v2)."
    )

    parser.add_argument("--version",    required=True, choices=["v1", "v2"],
                        help="LACE version: v1 uses ViT CLS token, v2 uses MaskTokenDecoder.")
    parser.add_argument("--downstream_visual_mode", default="cls_fg",
                        choices=["cls", "fg", "cls_fg"],
                        help="v2 visual representation: cls [B,512], fg [B,512], cls_fg [B,1024].")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a LACE pretrain checkpoint (.pt).")
    parser.add_argument("--splits",     required=True,
                        help="Path to splits.json produced by LACE pretraining.")
    parser.add_argument("--out_dir",    default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run_name",   default=None,
                        help="Override the head-checkpoint name (best_<run_name>.pt). "
                             "Default is a timestamp (+ wandb id), so runs never clobber each other.")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--patience",       type=int,   default=10)
    parser.add_argument("--early_stopping_metric", default="val_bal_acc",
                        choices=["val_loss", "val_bal_acc"],
                        help="Metric to monitor for early stopping and best-checkpoint saving.")
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--dropout",        type=float, default=0.3,
                        help="Dropout for the v1 MalignancyMLP.")
    parser.add_argument("--meta_embed_dim", type=int,   default=32)
    parser.add_argument("--hidden_dims",    type=str,   nargs="+", default=[256, 128],
                        help="Hidden layer widths for the v1 MLP.")
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--head", default="mlp",
                        choices=["linear", "mlp", "mlp_no_meta"],
                        help="linear: linear probe; mlp: MLP+meta (default); mlp_no_meta: MLP without metadata.")

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

    # ── Image resolution ─────────────────────────────────────────────────────
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input resolution (default 224). ViT-B/16 supports arbitrary "
                             "sizes via pos-embedding interpolation.")

    # ── Held-out evaluation ────────────────────────────────────────────────────
    parser.add_argument("--eval_test", action="store_true",
                        help="Evaluate the split's held-out test set after training. "
                             "Off by default so hyperparameter sweeps never touch test/BTXRD.")
    parser.add_argument("--btxrd_manifest", default=None,
                        help="Optional path to a BTXRD downstream manifest JSON (built by "
                             "build_btxrd_downstream.py). When set together with --eval_test, "
                             "BTXRD is evaluated as an external test set using the train age stats.")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--binary", action="store_true",
                        help="Binary mode: benign vs malignant only (intermediate cases skipped).")
    parser.add_argument("--use_mask", nargs="?", const=True, default=False,
                        type=lambda x: str(x).lower() in ("true", "1", "yes"),
                        help="Apply lesion mask cropping to input images (via DownstreamDataset).")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="lace-downstream")
    parser.add_argument("--wandb_run",     default=None)
    parser.add_argument("--wandb_entity",  default=None)
    parser.add_argument("--sweep",         action="store_true")

    args = parser.parse_args(argv)
    raw = " ".join(str(x) for x in args.hidden_dims)
    args.hidden_dims = [int(x) for x in raw.strip("[]").replace(",", " ").split()]
    return args
