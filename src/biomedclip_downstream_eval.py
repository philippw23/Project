"""Evaluate a *saved* BiomedCLIP downstream head — no training, no hyperparameter flags.

The head checkpoint written by `biomedclip_downstream.py` stores its full config
(`args`), the train age-normalization stats, and (for fine-tuned runs) the encoder
LoRA deltas. Evaluation only needs the head checkpoint and which test split to score:

    python src/biomedclip_downstream_eval.py \\
        --head_checkpoint results/biomedclip_downstream/run_.../best_head.pt \\
        --splits          data/internal_dataset/split_binary.json

Optional:
    --btxrd_manifest <path>   also score the external BTXRD test set (requires the
                              head to have been trained with --binary)
    --checkpoint <path>       override the backbone path stored in the checkpoint
    --seed <int>              override the stored seed
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader

from biomedclip.utils.misc import MODEL_TAG
from biomedclip.data.transforms import build_preprocess_val
from biomedclip.data.datasets import EmbeddingDataset
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.classifier import extract_embeddings
from biomedclip.models.lora import inject_lora, unfreeze_or_inject_downstream_lora
from LACE.models.encoders import enable_dynamic_img_size
from biomedclip.utils.downstream_eval import (
    resolve_label_maps, require_binary_for_btxrd, load_btxrd_samples,
    build_downstream_loader, report_eval,
)
from biomedclip_downstream import build_head, evaluate, EMBED_DIM


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved BiomedCLIP downstream head (no training).")
    p.add_argument("--head_checkpoint", required=True,
                   help="Trained head .pt (best_head.pt) — carries its own config.")
    p.add_argument("--splits", required=True, help="Split manifest whose 'test' set is scored.")
    p.add_argument("--btxrd_manifest", default=None,
                   help="Optional BTXRD manifest to also score as an external test set.")
    p.add_argument("--checkpoint", default=None,
                   help="Optional backbone checkpoint path override (else the stored one is used).")
    p.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    return p.parse_args(argv)


def _load_config(head_ckpt: dict, cli: argparse.Namespace) -> argparse.Namespace:
    if "args" not in head_ckpt:
        raise SystemExit(
            "This head checkpoint has no embedded config (it predates the config-saving "
            "change). Retrain the head with the updated biomedclip_downstream.py."
        )
    args = argparse.Namespace(**head_ckpt["args"])
    args.splits         = cli.splits
    args.btxrd_manifest = cli.btxrd_manifest
    args.eval_test      = True
    if cli.checkpoint:
        args.checkpoint = cli.checkpoint
    if cli.seed is not None:
        args.seed = cli.seed
    return args


def main(cli: argparse.Namespace) -> dict:
    head_ckpt = torch.load(cli.head_checkpoint, map_location="cpu", weights_only=False)
    args = _load_config(head_ckpt, cli)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  eval-only")

    require_binary_for_btxrd(args.binary, args.btxrd_manifest)
    label_to_idx, idx_to_label, num_classes = resolve_label_maps(args.binary)
    if args.binary:
        print("Mode: binary (benign vs malignant)")

    # ── Rebuild encoder ───────────────────────────────────────────────────────
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)
    preprocess_val = build_preprocess_val(preprocess_val, args.image_size)
    if not getattr(args, "freezed_biomedclip", False):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        lora_cfg = ckpt.get("lora_config") or {}
        if lora_cfg and not lora_cfg.get("no_lora"):
            inject_lora(model, lora_cfg["lora_layers"], lora_cfg["lora_r"], lora_cfg["lora_alpha"])
        model.load_state_dict(ckpt["model_state_dict"])
    encoder = model.visual.trunk.to(device)
    if args.image_size != 224:
        enable_dynamic_img_size(encoder)
    encoder.requires_grad_(False)
    encoder.eval()

    # Restore fine-tuned encoder LoRA deltas, if any.
    if head_ckpt.get("args", {}).get("finetune_lora_layers", 0) and "encoder_lora_state_dict" in head_ckpt:
        unfreeze_or_inject_downstream_lora(
            encoder, args.finetune_lora_layers,
            r=args.finetune_lora_r, alpha=args.finetune_lora_r * 2)
        enc_state = encoder.state_dict()
        enc_state.update(head_ckpt["encoder_lora_state_dict"])
        encoder.load_state_dict(enc_state)
        encoder.requires_grad_(False)
        encoder.eval()

    embed_dim = head_ckpt.get("embed_dim", EMBED_DIM)
    head = build_head(args, embed_dim, device, num_classes=num_classes)
    head.load_state_dict(head_ckpt["head_state_dict"])
    print(f"Loaded head checkpoint {cli.head_checkpoint} "
          f"(epoch {head_ckpt.get('epoch', '?')}, val_loss={head_ckpt.get('val_loss', float('nan')):.4f})")

    # ── Age normalization + loss ──────────────────────────────────────────────
    with open(args.splits, encoding="utf-8") as fh:
        splits = json.load(fh)
    if "age_mean" in head_ckpt and "age_std" in head_ckpt:
        age_mean, age_std = head_ckpt["age_mean"], head_ckpt["age_std"]
    else:
        train_ages = [s["age"] for s in splits["train"]]
        age_mean, age_std = float(np.mean(train_ages)), float(np.std(train_ages))
    print(f"Age stats (train): mean={age_mean:.1f}, std={age_std:.1f}")

    train_labels_all = torch.tensor(
        [label_to_idx[s["label"]] for s in splits["train"] if s["label"] in label_to_idx],
        dtype=torch.long)
    label_counts  = torch.bincount(train_labels_all, minlength=num_classes).float()
    class_weights = compute_class_weights(
        label_counts, num_classes=num_classes, mode=args.class_weighting,
        beta=args.cb_beta, device=device)
    criterion = build_classification_loss(args, label_counts, num_classes, class_weights, device)

    def _score(samples: list[dict]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        raw, n = build_downstream_loader(
            samples, age_mean, age_std, preprocess_val, args.use_mask,
            args.batch_size, label_to_idx, device)
        print(f"  samples: {n}")
        emb, age, sex, lbl = extract_embeddings(encoder, raw, device)
        loader = DataLoader(EmbeddingDataset(emb, age, sex, lbl),
                            batch_size=args.batch_size, shuffle=False)
        loss, _, preds, labels, probs = evaluate(head, loader, criterion, device)
        return loss, preds, labels, probs

    results: dict = {}
    print("Scoring internal test...")
    test_loss, test_preds, test_labels, test_probs = _score(splits["test"])
    results.update(report_eval("TEST", test_preds, test_labels, test_loss,
                               idx_to_label, num_classes, prefix="test", probs=test_probs))

    if args.btxrd_manifest:
        print(f"Scoring BTXRD from {args.btxrd_manifest}...")
        btxrd_loss, btxrd_preds, btxrd_labels, btxrd_probs = _score(load_btxrd_samples(args.btxrd_manifest))
        results.update(report_eval("BTXRD (external)", btxrd_preds, btxrd_labels, btxrd_loss,
                                   idx_to_label, num_classes, prefix="btxrd", probs=btxrd_probs))
    return results


if __name__ == "__main__":
    main(parse_args())
