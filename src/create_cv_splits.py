"""Create stratified, patient-grouped K-fold CV splits for the internal dataset.

Pools an existing split manifest (train+val+test) and re-partitions it into K
folds. Each fold file holds `train` / `val` / `test` in the same sample-dict
schema as the input, where:

  * `test` = the held-out fold (≈ 1/K of the data),
  * `val`  = a stratified, patient-grouped slice carved from the remaining folds
             (used for within-fold early stopping / checkpointing),
  * `train`= everything else.

Splitting is stratified by `label` and grouped by `patid` (via
StratifiedGroupKFold) so no patient leaks across train/val/test within a fold.

Usage:
    python src/create_cv_splits.py \\
        --input  data/internal_dataset/test/split_binary_backup.json \\
        --out_dir data/internal_dataset/cv \\
        --folds 5 --val_frac 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT_DIR = Path(__file__).resolve().parent.parent


def _group_of(sample: dict) -> str:
    """Patient group key; fall back to the image stem when patid is absent."""
    pid = sample.get("patid")
    if pid is None or str(pid).strip() == "":
        return Path(sample["image"]).stem
    return str(pid)


def _stratified_grouped_subset(
    samples: list[dict],
    frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return (rest_idx, subset_idx) where subset ≈ `frac` of `samples`,
    stratified by label and grouped by patient (one StratifiedGroupKFold fold)."""
    n_splits = max(2, round(1.0 / frac))
    y = [s["label"] for s in samples]
    groups = [_group_of(s) for s in samples]
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, subset_idx = next(iter(skf.split(samples, y, groups)))
    return list(rest_idx), list(subset_idx)


def _dist(samples: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(s["label"] for s in samples).items()))


def build(args: argparse.Namespace) -> None:
    with open(args.input, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        pool = [s for key in ("train", "val", "test") for s in data.get(key, [])]
    else:
        pool = list(data)
    print(f"Pooled {len(pool)} samples from {args.input} | labels: {_dist(pool)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y      = [s["label"] for s in pool]
    groups = [_group_of(s) for s in pool]
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # val fraction relative to the (folds-1)/folds remaining pool
    remaining_frac = (args.folds - 1) / args.folds
    val_frac_of_rest = args.val_frac / remaining_frac

    written = []
    for fold, (rest_idx, test_idx) in enumerate(skf.split(pool, y, groups)):
        test_samples = [pool[i] for i in test_idx]
        rest_samples = [pool[i] for i in rest_idx]

        sub_rest_idx, val_local_idx = _stratified_grouped_subset(
            rest_samples, val_frac_of_rest, seed=args.seed + fold,
        )
        val_samples   = [rest_samples[i] for i in val_local_idx]
        train_samples = [rest_samples[i] for i in sub_rest_idx]

        # sanity: no patient overlap across the three splits
        g_tr = {_group_of(s) for s in train_samples}
        g_va = {_group_of(s) for s in val_samples}
        g_te = {_group_of(s) for s in test_samples}
        assert not (g_tr & g_va), f"fold {fold}: train/val patient overlap"
        assert not (g_tr & g_te), f"fold {fold}: train/test patient overlap"
        assert not (g_va & g_te), f"fold {fold}: val/test patient overlap"

        out_path = out_dir / f"split_binary_fold{fold}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"train": train_samples, "val": val_samples, "test": test_samples}, fh, indent=2)
        written.append(out_path)
        print(
            f"  fold {fold}: train={len(train_samples)} {_dist(train_samples)} | "
            f"val={len(val_samples)} {_dist(val_samples)} | "
            f"test={len(test_samples)} {_dist(test_samples)}"
        )

    print(f"\nWrote {len(written)} fold files to {out_dir}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create stratified patient-grouped K-fold CV splits.")
    p.add_argument("--input",   default=str(ROOT_DIR / "data" / "internal_dataset" / "test" / "split_binary_backup.json"),
                   help="Existing split manifest (train/val/test) or a flat list of samples.")
    p.add_argument("--out_dir", default=str(ROOT_DIR / "data" / "internal_dataset" / "cv"))
    p.add_argument("--folds",    type=int,   default=5)
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Validation fraction of the whole dataset (carved from each fold's train pool).")
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
