"""LACE v2 image+text downstream classifier — concatenated visual+text embeddings.

Frozen SharedViT visual representation (--downstream_visual_mode: cls [512] or
cls_fg [1024]) concatenated with a frozen BiomedCLIPTextEncoder projection of the
combined report text (befund_en + beurteilung_en, matching the BiomedCLIP img+text
baseline's text convention), fed into a linear or MLP probing head. Both encoders
come from the same LACE v2 pretraining checkpoint and are always frozen.

No BTXRD evaluation: BTXRD samples carry no report text, so the text-encoder
pathway this baseline depends on has nothing to embed for that dataset. There
is no --btxrd_manifest flag here (mirrors biomedclip/train/img_text_downstream.py).

Usage:
    python src/LACE/train/img_text_downstream.py \\
        --checkpoint results/lace_v2_pretrain/.../best_retrieval_checkpoint.pt \\
        --splits     data/internal_dataset/split_binary_final.json \\
        --downstream_visual_mode cls_fg --head mlp_no_meta --binary true
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
from PIL import Image, ImageFile
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.data.datasets import (
    EmbeddingDataset, LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES,
    LABEL_TO_IDX_BINARY, IDX_TO_LABEL_BINARY, NUM_CLASSES_BINARY,
)
from biomedclip.data.transforms import build_preprocess_val, crop_around_mask
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.classifier import LinearHead, MalignancyMLP
from biomedclip.utils.downstream_eval import report_eval, compute_auroc, safe_wandb_log
from biomedclip.utils.misc import DEFAULT_OUT_DIR
from LACE.data.transforms import build_train_transform_lace
from LACE.models.downstream import LACEv2Classifier
from LACE.models.encoders import BiomedCLIPTextEncoder
from LACE.models.mask_tokens import MaskTokenDecoder, MaskPredictionHead
from LACE.train.downstream import _load_vit, evaluate

ImageFile.LOAD_TRUNCATED_IMAGES = True

VISUAL_DIM = {"cls": 512, "fg": 512, "cls_fg": 1024}
TEXT_DIM   = 512


# ── Data ─────────────────────────────────────────────────────────────────────

class DownstreamDatasetImgText(Dataset):
    """Image + combined report text (befund_en + beurteilung_en) for one sample.

    Text is tokenised HF-style (input_ids + attention_mask) for
    BiomedCLIPTextEncoder.encode_beurteilung, which — despite the name — is a
    generic CLS-projection over whatever token ids it is given.
    """

    def __init__(
        self,
        samples: list[dict],
        age_sex_lookup: dict[str, tuple[float, float]],
        age_mean: float,
        age_std: float,
        preprocess,
        hf_tokenizer,
        max_text_len: int,
        use_mask: bool,
        label_to_idx: dict[str, int],
    ) -> None:
        self._label_to_idx = label_to_idx
        valid = []
        for s in samples:
            stem = Path(s["image"]).stem
            if stem not in age_sex_lookup:
                continue
            if s["label"] not in self._label_to_idx:
                continue
            valid.append(s)
        self.samples        = valid
        self.age_sex_lookup = age_sex_lookup
        self.age_mean       = age_mean
        self.age_std        = age_std
        self.preprocess     = preprocess
        self.hf_tokenizer   = hf_tokenizer
        self.max_text_len   = max_text_len
        self.use_mask       = use_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s    = self.samples[idx]
        stem = Path(s["image"]).stem
        age_raw, sex = self.age_sex_lookup[stem]

        image     = Image.open(s["image"]).convert("RGB")
        mask_path = Path(s.get("mask") or "")
        if self.use_mask and mask_path.name and mask_path.exists():
            image_arr = np.array(image.convert("L"), dtype=float)
            mask_arr  = np.array(Image.open(mask_path).convert("L"), dtype=float)
            cropped   = crop_around_mask(image_arr, mask_arr)
            image     = Image.fromarray(cropped.astype(np.uint8)).convert("RGB")

        image_tensor = self.preprocess(image)
        age_norm     = (age_raw - self.age_mean) / (self.age_std + 1e-6)
        label        = self._label_to_idx[s["label"]]

        text = " ".join(filter(None, [s.get("befund_en"), s.get("beurteilung_en")])) or s.get("report") or ""
        enc = self.hf_tokenizer(
            text if text else "[PAD]",
            max_length=self.max_text_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        return {
            "image":     image_tensor,
            "text_ids":  enc["input_ids"].squeeze(0),
            "text_mask": enc["attention_mask"].squeeze(0),
            "age":       torch.tensor(age_norm, dtype=torch.float32),
            "sex":       torch.tensor(sex,      dtype=torch.float32),
            "label":     torch.tensor(label,    dtype=torch.long),
        }


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_v2_visual_backbone(
    checkpoint: dict, device: torch.device, visual_mode: str,
) -> LACEv2Classifier:
    """Frozen SharedViT + MaskTokenDecoder wrapped for `._get_visual(images)` only.

    The classifier's own head is unused (n_classes=1, discarded) — this baseline
    trains its own head on the concatenated image+text embedding instead.
    """
    vit = _load_vit(checkpoint, device)

    n_mask_tokens = checkpoint.get("n_mask_tokens", 4)
    mask_decoder = MaskTokenDecoder(
        n_tokens=n_mask_tokens,
        n_heads=checkpoint.get("n_mask_heads", 8),
        sigma=checkpoint.get("gauss_sigma", 1.5),
    )
    mask_decoder.load_state_dict(checkpoint["mask_decoder_state"])
    mask_decoder.to(device)

    mask_head = MaskPredictionHead()
    mask_head.load_state_dict(checkpoint["mask_head_state"])
    mask_head.to(device)

    return LACEv2Classifier(
        vit=vit, mask_decoder=mask_decoder, mask_head=mask_head,
        n_classes=1, use_meta=False, linear_head=True, visual_mode=visual_mode,
    ).to(device)


def _load_text_encoder(checkpoint: dict, device: torch.device) -> BiomedCLIPTextEncoder:
    embed_dim = checkpoint.get("lora_config", {}).get("embed_dim", 512)
    text_enc  = BiomedCLIPTextEncoder(embed_dim=embed_dim)
    text_enc.load_state_dict(checkpoint["text_enc_state"])
    for p in text_enc.parameters():
        p.requires_grad_(False)
    return text_enc.to(device).eval()


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


# ── Representation extraction / training ────────────────────────────────────

@torch.no_grad()
def extract_img_text_representations(
    classifier: LACEv2Classifier,
    text_enc: BiomedCLIPTextEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-compute concatenated L2-normalised visual+text embeddings for one split."""
    classifier.eval()
    text_enc.eval()
    all_emb, all_age, all_sex, all_lbl = [], [], [], []
    for batch in tqdm(loader, desc="  embedding", leave=False):
        images = batch["image"].to(device)
        ids    = batch["text_ids"].to(device)
        mask   = batch["text_mask"].to(device)
        img_emb = F.normalize(classifier._get_visual(images), dim=-1)
        txt_emb = F.normalize(text_enc.encode_beurteilung(ids, mask), dim=-1)
        all_emb.append(torch.cat([img_emb, txt_emb], dim=-1).cpu())
        all_age.append(batch["age"])
        all_sex.append(batch["sex"])
        all_lbl.append(batch["label"])
    return (
        torch.cat(all_emb), torch.cat(all_age), torch.cat(all_sex), torch.cat(all_lbl),
    )


def train_one_epoch_aug(
    head: nn.Module,
    classifier: LACEv2Classifier,
    text_enc: BiomedCLIPTextEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Training loop with on-the-fly image augmentation (text is not augmented)."""
    head.train()
    classifier.eval()
    text_enc.eval()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        ids    = batch["text_ids"].to(device)
        mask   = batch["text_mask"].to(device)
        age    = batch["age"].to(device)
        sex    = batch["sex"].to(device)
        lbl    = batch["label"].to(device)
        with torch.no_grad():
            img_emb = F.normalize(classifier._get_visual(images), dim=-1)
            txt_emb = F.normalize(text_enc.encode_beurteilung(ids, mask), dim=-1)
            emb     = torch.cat([img_emb, txt_emb], dim=-1)
        optimizer.zero_grad()
        loss = criterion(head(emb, age, sex), lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


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


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = VISUAL_DIM[args.downstream_visual_mode] + TEXT_DIM
    print(f"Device: {device}  |  Head: {args.head}  |  Visual mode: {args.downstream_visual_mode} "
          f"(embed_dim={embed_dim})")

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
    if args.binary:
        label_to_idx, idx_to_label, num_classes = LABEL_TO_IDX_BINARY, IDX_TO_LABEL_BINARY, NUM_CLASSES_BINARY
        print("Mode: binary (benign vs malignant)")
    else:
        label_to_idx, idx_to_label, num_classes = LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES

    # ── Load LACE v2 checkpoint (frozen visual backbone + frozen text encoder) ──
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classifier = _load_v2_visual_backbone(ckpt, device, args.downstream_visual_mode)
    text_enc   = _load_text_encoder(ckpt, device)
    hf_tokenizer = getattr(text_enc.tokenizer, "tokenizer", text_enc.tokenizer)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f})")

    preprocess_val   = build_preprocess_val(classifier.vit.preprocess_val, args.image_size)
    preprocess_train = build_train_transform_lace(preprocess_val)

    # ── Data ──────────────────────────────────────────────────────────────────
    with open(args.splits, encoding="utf-8") as fh:
        splits = json.load(fh)

    all_samples    = splits["train"] + splits["val"] + splits["test"]
    age_sex_lookup = {Path(s["image"]).stem: (float(s["age"]), float(s["sex"])) for s in all_samples}
    train_ages     = [s["age"] for s in splits["train"]]
    age_mean = float(np.mean(train_ages))
    age_std  = float(np.std(train_ages))
    print(f"Age stats (train): mean={age_mean:.1f}, std={age_std:.1f}")

    val_kwargs   = dict(age_sex_lookup=age_sex_lookup, age_mean=age_mean, age_std=age_std,
                        preprocess=preprocess_val,   hf_tokenizer=hf_tokenizer,
                        max_text_len=args.max_text_len, use_mask=args.use_mask,
                        label_to_idx=label_to_idx)
    train_kwargs = dict(age_sex_lookup=age_sex_lookup, age_mean=age_mean, age_std=age_std,
                        preprocess=preprocess_train, hf_tokenizer=hf_tokenizer,
                        max_text_len=args.max_text_len, use_mask=args.use_mask,
                        label_to_idx=label_to_idx)

    train_ds = DownstreamDatasetImgText(splits["train"], **train_kwargs)
    val_ds   = DownstreamDatasetImgText(splits["val"],   **val_kwargs)
    test_ds  = DownstreamDatasetImgText(splits["test"],  **val_kwargs)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    use_pin       = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}
    train_loader     = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader_raw   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    print("Pre-computing val embeddings...")
    val_emb, val_age, val_sex, val_lbl = extract_img_text_representations(
        classifier, text_enc, val_loader_raw, device)
    val_loader = DataLoader(EmbeddingDataset(val_emb, val_age, val_sex, val_lbl),
                            batch_size=args.batch_size, shuffle=False)

    test_loader = None
    if args.eval_test:
        test_loader_raw = DataLoader(test_ds, shuffle=False, **loader_kwargs)
        print("Pre-computing test embeddings...")
        test_emb, test_age, test_sex, test_lbl = extract_img_text_representations(
            classifier, text_enc, test_loader_raw, device)
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
    head      = build_head(args, embed_dim, device, num_classes)
    criterion = build_classification_loss(args, label_counts, num_classes, class_weights, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = args.run_name or (
        f"run_{args.head}_{args.downstream_visual_mode}_{args.loss}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir  = Path(args.out_dir) / "lace_img_text_downstream" / run_name
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
        train_loss = train_one_epoch_aug(
            head, classifier, text_enc, train_loader, optimizer, criterion, device)
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
                "downstream_visual_mode": args.downstream_visual_mode,
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LACE v2 img+text downstream — concatenated visual+text embeddings."
    )
    # ── Checkpoint / splits ──────────────────────────────────────────────────
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a LACE v2 pretrain checkpoint (.pt).")
    parser.add_argument("--splits",     required=True,
                        help="Path to splits.json with befund_en/beurteilung_en text fields.")
    parser.add_argument("--out_dir",  default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--use_mask", nargs="?", const=True, default=False,
                        type=lambda x: str(x).lower() in ("true", "1", "yes"))
    parser.add_argument("--binary", type=lambda x: str(x).lower() in ("true", "1", "yes"),
                        default=False,
                        help="Binary classification (benign vs malignant). "
                             "Use with split_binary*.json — intermediate cases must already be excluded.")
    parser.add_argument("--eval_test", action="store_true",
                        help="Evaluate the split's held-out test set after training. Off by "
                             "default so hyperparameter sweeps never touch test. No BTXRD "
                             "evaluation — BTXRD carries no report text for the text encoder.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_text_len", type=int, default=384,
                        help="Tokenisation length for the combined befund+beurteilung text.")

    # ── Visual representation (v2 only) ──────────────────────────────────────
    parser.add_argument("--downstream_visual_mode", default="cls_fg",
                        choices=["cls", "fg", "cls_fg"],
                        help="cls [512] | fg [512] | cls_fg [1024], concatenated with 512-dim text.")

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
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="lace-img-text-downstream")
    parser.add_argument("--wandb_run",     default=None)
    parser.add_argument("--wandb_entity",  default=None)
    parser.add_argument("--sweep",         action="store_true")

    args = parser.parse_args(argv)
    raw = " ".join(str(x) for x in args.hidden_dims)
    args.hidden_dims = [int(x) for x in raw.strip("[]").replace(",", " ").split()]
    return args


if __name__ == "__main__":
    main(parse_args())
