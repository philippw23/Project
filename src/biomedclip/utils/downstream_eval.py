"""Shared helpers for downstream test / BTXRD evaluation across baselines.

These utilities factor out the encoder-agnostic pieces that each baseline
downstream script (biomedclip / chexfound / gloria / imagenet_img / scratch_img)
needs to gain the same held-out-test and external-BTXRD evaluation capability as
LACE:

* `resolve_label_maps`     — pick 3-class or binary label maps from `--binary`
* `require_binary_for_btxrd` — hard error if BTXRD is requested without `--binary`
* `load_btxrd_samples`     — read a BTXRD downstream manifest (list of sample dicts)
* `build_downstream_loader`— build a raw-image DataLoader for a list of samples,
                             normalising age with train-derived stats (so external
                             sets like BTXRD use the same statistics as training)
* `report_eval`            — print a results block and return a flat `{prefix}/...`
                             metrics dict (mirrors LACE's `_report_eval`)

BTXRD is binary-only by construction (benign / malignant), so a comparable BTXRD
number requires the baseline to run in `--binary` mode; `require_binary_for_btxrd`
enforces this rather than silently producing a non-comparable 3-class result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader

from biomedclip.data.datasets import (
    DownstreamDataset,
    LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES,
    LABEL_TO_IDX_BINARY, IDX_TO_LABEL_BINARY, NUM_CLASSES_BINARY,
)


def resolve_label_maps(binary: bool) -> tuple[dict, dict, int]:
    """Return (label_to_idx, idx_to_label, num_classes) for the chosen class mode."""
    if binary:
        return LABEL_TO_IDX_BINARY, IDX_TO_LABEL_BINARY, NUM_CLASSES_BINARY
    return LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES


def require_binary_for_btxrd(binary: bool, btxrd_manifest: str | None) -> None:
    """Hard error if a BTXRD manifest is requested without binary mode.

    BTXRD carries only benign/malignant labels, so a 3-class run would produce a
    result that is not comparable to the binary LACE BTXRD numbers.
    """
    if btxrd_manifest and not binary:
        raise SystemExit(
            "--btxrd_manifest requires --binary: BTXRD is a binary (benign vs "
            "malignant) external test set, so scoring it with a 3-class head "
            "produces numbers that are not comparable to the binary results. "
            "Re-run with --binary (and a split_binary.json)."
        )


def load_btxrd_samples(btxrd_manifest: str) -> list[dict]:
    """Load a BTXRD downstream manifest (flat JSON list of sample dicts)."""
    with open(btxrd_manifest, encoding="utf-8") as fh:
        return json.load(fh)


def build_downstream_loader(
    samples: list[dict],
    age_mean: float,
    age_std: float,
    preprocess_val,
    use_mask: bool,
    batch_size: int,
    label_to_idx: dict,
    device,
    num_workers: int = 4,
) -> tuple[DataLoader, int]:
    """Build a raw-image DataLoader over `samples`.

    age/sex are read from each sample dict; ages are normalised with the given
    (train-derived) `age_mean`/`age_std` so external sets like BTXRD use the same
    statistics as the training data. Samples whose label is not in `label_to_idx`
    are dropped by `DownstreamDataset`.
    """
    lookup = {Path(s["image"]).stem: (float(s["age"]), float(s["sex"])) for s in samples}
    ds = DownstreamDataset(
        samples, lookup, age_mean, age_std,
        preprocess_val, use_mask, label_to_idx=label_to_idx,
    )
    loader = DataLoader(
        ds, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    return loader, len(ds)


def report_eval(
    name: str,
    preds: np.ndarray,
    labels: np.ndarray,
    loss: float,
    idx_to_label: dict,
    num_classes: int,
    prefix: str,
) -> dict:
    """Print a results block for an evaluated set and return a flat metrics dict
    keyed by `prefix` (e.g. 'test' or 'btxrd')."""
    label_names   = [idx_to_label[i] for i in range(num_classes)]
    present       = sorted(set(labels.tolist()) | set(preds.tolist()))
    present_names = [label_names[i] for i in present]

    acc = float((preds == labels).mean())
    bal = balanced_accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    wprec, wrec, _, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"{name} RESULTS")
    print("=" * 60)
    print(f"Loss: {loss:.4f}  |  Accuracy: {acc:.3f}")
    print(f"Balanced accuracy: {bal:.3f}")
    print(f"Macro F1: {f1m:.3f}")
    print(f"Weighted Precision: {wprec:.3f}  |  Weighted Recall: {wrec:.3f}")
    print()
    print(classification_report(
        labels, preds, labels=present, target_names=present_names,
        digits=3, zero_division=0,
    ))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(
        confusion_matrix(labels, preds, labels=present),
        index=present_names, columns=present_names,
    ).to_string())

    prec, rec, _, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    pc_prec, pc_rec, pc_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=present, zero_division=0
    )
    metrics = {
        f"{prefix}/loss": loss, f"{prefix}/acc": acc,
        f"{prefix}/balanced_acc":       bal,
        f"{prefix}/precision_macro":    prec,
        f"{prefix}/recall_macro":       rec,
        f"{prefix}/precision_weighted": wprec,
        f"{prefix}/recall_weighted":    wrec,
        f"{prefix}/f1_macro":           f1m,
    }
    for i, nm in enumerate(present_names):
        metrics[f"{prefix}/precision_{nm}"] = pc_prec[i]
        metrics[f"{prefix}/recall_{nm}"]    = pc_rec[i]
        metrics[f"{prefix}/f1_{nm}"]        = pc_f1[i]
    return metrics
