"""Evaluate a *saved* LACE downstream head — no training, no hyperparameter flags.

The head checkpoint written by `LACE.train.downstream` stores its full config
(architecture + loss hyperparameters, the backbone checkpoint path, and the
train age-normalization stats). So evaluation only needs the head checkpoint and
which test split to score:

    python src/LACE/train/downstream_eval.py \\
        --head_checkpoint results/lace_v2_downstream/best_<run_id>.pt \\
        --splits          data/internal_dataset/test/split_binary_backup.json

Optional:
    --btxrd_manifest <path>   also score the external BTXRD test set
    --checkpoint <path>       override the backbone path stored in the checkpoint
                              (e.g. if the pretrain checkpoint has since moved)
    --seed <int>              override the stored seed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from biomedclip.data.datasets import (IDX_TO_LABEL, LABEL_TO_IDX, NUM_CLASSES,
                                       IDX_TO_LABEL_BINARY, LABEL_TO_IDX_BINARY,
                                       NUM_CLASSES_BINARY)
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.utils.downstream_eval import report_eval
from LACE.train.downstream import (build_v2_model, extract_v2_representations,
                                    evaluate, _precompute_repr_loader)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the CLI args for this evaluation script."""
    p = argparse.ArgumentParser(description="Evaluate a saved LACE downstream head (no training).")
    p.add_argument("--head_checkpoint", required=True,
                   help="Trained head .pt (best_<run_id>.pt) — carries its own config.")
    p.add_argument("--splits", required=True,
                   help="Split manifest whose 'test' set is scored.")
    p.add_argument("--btxrd_manifest", default=None,
                   help="Optional BTXRD downstream manifest to also score as an external test set.")
    p.add_argument("--checkpoint", default=None,
                   help="Optional backbone checkpoint path override (else the stored one is used).")
    p.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    return p.parse_args(argv)


def _load_config(head_ckpt: dict, cli: argparse.Namespace) -> argparse.Namespace:
    """Rebuild the training args Namespace from the stored config + CLI overrides."""
    if "args" not in head_ckpt:
        raise SystemExit(
            "This head checkpoint has no embedded config (it predates the config-saving "
            "change). Retrain the head with the updated downstream.py, or evaluate it by "
            "re-running LACE/train/downstream.py with --eval_test and the original flags."
        )
    args = argparse.Namespace(**head_ckpt["args"])
    # evaluation-time overrides
    args.splits         = cli.splits
    args.btxrd_manifest = cli.btxrd_manifest
    args.eval_test      = True
    if cli.checkpoint:
        args.checkpoint = cli.checkpoint
    if cli.seed is not None:
        args.seed = cli.seed
    return args


def main(cli: argparse.Namespace) -> dict:
    """Evaluate a saved LACE downstream head checkpoint on the specified test split(s)."""
    # Load the head checkpoint and rebuild the training args Namespace from its stored config.
    head_ckpt = torch.load(cli.head_checkpoint, map_location="cpu", weights_only=False)
    args = _load_config(head_ckpt, cli)

    # Set the random seeds and device, and print the evaluation mode.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  LACE version: {args.version}  |  eval-only")

    # Check that the backbone checkpoint exists (either the stored one or the CLI override).
    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"Backbone checkpoint not found: {args.checkpoint}\n"
            f"Pass --checkpoint <path> to point at the (possibly moved) pretrain checkpoint."
        )

    # Determine whether this is a binary or multi-class downstream task, to pick the right label mapping and number of classes.
    if args.binary:
        label_to_idx, idx_to_label, num_classes = (
            LABEL_TO_IDX_BINARY, IDX_TO_LABEL_BINARY, NUM_CLASSES_BINARY)
        print("Mode: binary (benign vs malignant)")
    else:
        label_to_idx, idx_to_label, num_classes = LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES

    # ── Backbone + head architecture, then load the trained head weights ──────
    classifier, head_wrapper, _, preprocess_val = build_v2_model(args, device, num_classes)
    trainable_model = head_wrapper
    extractor = lambda ldr: extract_v2_representations(classifier, ldr, device)

    trainable_model.load_state_dict(head_ckpt["model_state_dict"])
    print(f"Loaded head checkpoint {cli.head_checkpoint} "
          f"(epoch {head_ckpt.get('epoch', '?')}, val_loss={head_ckpt.get('val_loss', float('nan')):.4f})")

    # Train age-normalization: prefer the stored stats; fall back to the split.
    if "age_mean" in head_ckpt and "age_std" in head_ckpt:
        age_mean, age_std = head_ckpt["age_mean"], head_ckpt["age_std"]
    else:
        with open(args.splits, encoding="utf-8") as fh:
            train_ages = [s["age"] for s in json.load(fh)["train"]]
        age_mean, age_std = float(np.mean(train_ages)), float(np.std(train_ages))
    print(f"Age stats (train): mean={age_mean:.1f}, std={age_std:.1f}")

    with open(args.splits, encoding="utf-8") as fh:
        splits = json.load(fh)

    # ── Loss (for a comparable reported test loss) ────────────────────────────
    train_labels_all = torch.tensor(
        [label_to_idx[s["label"]] for s in splits["train"] if s["label"] in label_to_idx],
        dtype=torch.long,
    )
    label_counts  = torch.bincount(train_labels_all, minlength=num_classes).float()
    class_weights = compute_class_weights(
        label_counts, num_classes=num_classes, mode=args.class_weighting,
        beta=args.cb_beta, device=device,
    )
    criterion = build_classification_loss(args, label_counts, num_classes, class_weights, device)

    results: dict = {}

    # ── Internal test ─────────────────────────────────────────────────────────
    print("Pre-computing test representations...")
    test_loader, n_test = _precompute_repr_loader(
        splits["test"], age_mean, age_std, preprocess_val,
        args, extractor, device, label_to_idx,
    )
    print(f"Test samples: {n_test}")
    test_loss, _, test_preds, test_labels, test_probs = evaluate(trainable_model, test_loader, criterion, device)
    results.update(report_eval(
        "TEST", test_preds, test_labels, test_loss, idx_to_label, num_classes,
        prefix="test", probs=test_probs))

    # ── BTXRD external test (optional) ────────────────────────────────────────
    if args.btxrd_manifest:
        print(f"Pre-computing BTXRD representations from {args.btxrd_manifest}...")
        with open(args.btxrd_manifest, encoding="utf-8") as fh:
            btxrd_samples = json.load(fh)
        btxrd_loader, n_btxrd = _precompute_repr_loader(
            btxrd_samples, age_mean, age_std, preprocess_val,
            args, extractor, device, label_to_idx,
        )
        print(f"BTXRD samples: {n_btxrd}")
        btxrd_loss, _, btxrd_preds, btxrd_labels, btxrd_probs = evaluate(
            trainable_model, btxrd_loader, criterion, device)
        results.update(report_eval(
            "BTXRD (external)", btxrd_preds, btxrd_labels, btxrd_loss,
            idx_to_label, num_classes, prefix="btxrd", probs=btxrd_probs))

    return results


if __name__ == "__main__":
    main(parse_args())
