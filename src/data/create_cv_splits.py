"""Create stratified, patient-grouped 10-part CV splits for the internal dataset.

Keeps the given split manifest intact: the existing `train` split is divided
into 8 stratified, patient-grouped parts (parts 1-8), while the given `val`
and `test` splits are kept as-is and become part 9 and part 10. Together they
form a pool of 10 parts used for cross-validation.

Each fold file holds `train` / `val` / `test` in the same sample-dict schema
as the input, rotating through the pool:

  * fold i: `test` = part i, `val` = the part preceding it, `train` = the
    remaining 8 parts.

With this rotation the last fold (test = part 10, val = part 9) reproduces the
original split exactly.

The train split is partitioned stratified by `label` and grouped by `patid`
(via StratifiedGroupKFold) so no patient leaks across parts; the given
val/test splits are assumed to already be patient-disjoint from train (this is
asserted per fold).

Usage:
    python src/data/create_cv_splits.py --input  data/internal_dataset/split_final.json --out_dir data/internal_dataset/cv --seed 42
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

TRAIN_PARTS = 8  # given val/test become parts 9 and 10 -> pool of 10


def _group_of(sample: dict) -> str:
    """Patient group key; fall back to the image stem when patid is absent."""
    pid = sample.get("patid")
    if pid is None or str(pid).strip() == "":
        return Path(sample["image"]).stem
    return str(pid)


def _dist(samples: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(s["label"] for s in samples).items()))


def _partition_train(samples: list[dict], n_parts: int, seed: int) -> list[list[dict]]:
    """Split `samples` into `n_parts` stratified, patient-grouped parts."""
    y = [s["label"] for s in samples]
    groups = [_group_of(s) for s in samples]
    skf = StratifiedGroupKFold(n_splits=n_parts, shuffle=True, random_state=seed)
    return [[samples[i] for i in part_idx] for _, part_idx in skf.split(samples, y, groups)]


def build(args: argparse.Namespace) -> None:
    with open(args.input, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not all(k in data for k in ("train", "val", "test")):
        raise ValueError(f"{args.input} must be a split manifest with train/val/test keys")

    train, val, test = data["train"], data["val"], data["test"]
    print(
        f"Loaded {args.input}: train={len(train)} {_dist(train)} | "
        f"val={len(val)} {_dist(val)} | test={len(test)} {_dist(test)}"
    )

    # parts 1-8 from the given train split, part 9 = given val, part 10 = given test
    parts = _partition_train(train, TRAIN_PARTS, args.seed) + [val, test]
    n = len(parts)
    for i, part in enumerate(parts):
        print(f"  part {i + 1}: n={len(part)} {_dist(part)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for fold in range(n):
        test_samples = parts[fold]
        val_samples = parts[(fold - 1) % n]
        train_samples = [
            s for i, part in enumerate(parts) if i not in (fold, (fold - 1) % n) for s in part
        ]

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
    p = argparse.ArgumentParser(
        description="Create 10-part CV splits: 8 parts from the given train split, "
                    "plus the given val (part 9) and test (part 10)."
    )
    p.add_argument("--input",   default=str(ROOT_DIR / "data" / "internal_dataset" / "test" / "split_binary_backup.json"),
                   help="Existing split manifest with train/val/test keys.")
    p.add_argument("--out_dir", default=str(ROOT_DIR / "data" / "internal_dataset" / "cv_binary"))
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
