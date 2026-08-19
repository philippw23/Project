from __future__ import annotations

import json
import random
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import pandas as pd
from sklearn.model_selection import train_test_split

from .datasets import BoneTumorPairDataset


def load_age_sex_lookup(excel_path: Path) -> dict[str, tuple[float, float]]:
    """Return {filename_stem: (age_float, sex_binary)} from the Excel.

    sex: m/male -> 1.0,  f/female -> 0.0.
    Rows with missing or unparseable age/sex are skipped with a warning.
    """
    df = pd.read_excel(
        excel_path,
        sheet_name="internal_data_matched",
        usecols=[0, 4, 5],
        skiprows=1,
        header=0,
        dtype=str,
        engine="openpyxl",
    )
    df.columns = ["filename", "age", "sex"]
    df = df.dropna(subset=["filename"])
    df["filename"] = df["filename"].str.strip()

    lookup: dict[str, tuple[float, float]] = {}
    skipped = 0
    for _, row in df.iterrows():
        stem = Path(row["filename"]).stem
        try:
            age = float(row["age"])
        except (ValueError, TypeError):
            skipped += 1
            continue
        sex_raw = str(row["sex"]).strip().lower()
        if sex_raw == "m":
            sex = 1.0
        elif sex_raw == "f":
            sex = 0.0
        else:
            skipped += 1
            continue
        lookup[stem] = (age, sex)

    if skipped:
        warnings.warn(f"Skipped {skipped} rows with missing/invalid age or sex.")
    return lookup


def _stratified_split_two(
    samples: list,
    labels: list[str],
    split_size: float,
    seed: int,
) -> tuple[list, list]:
    """Split *samples* into two stratified groups."""
    try:
        a, b = train_test_split(
            samples, test_size=split_size, stratify=labels, random_state=seed
        )
    except ValueError:
        warnings.warn(
            "Stratified split failed (too few samples in some class). "
            "Falling back to random split.",
            stacklevel=3,
        )
        a, b = train_test_split(samples, test_size=split_size, random_state=seed)
    return a, b

def _group_by_patient(samples: list[dict]) -> list[list[dict]]:
    """Return samples grouped by patid, preserving insertion order."""
    groups: dict[str, list[dict]] = {}
    for s in samples:
        groups.setdefault(s["patid"], []).append(s)
    return list(groups.values())


def _mask_has_pixels(mask_path: Path) -> bool:
    """Return True iff *mask_path* exists and contains at least one foreground pixel."""
    if not mask_path.name or not mask_path.exists():
        return False
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return bool(np.any(mask > 0))


def build_stratified_splits(
    args,
    run_dir: Path | None = None,
) -> tuple[list, list, list]:
    """Create train / val / test splits from dataset_full.json.

    Reads from args.dataset (path to dataset_full.json produced by create_dataset.py).
    Filters to entries with non-null report, non-null label, and a valid segmentation mask.
    Encodes "unknown" age/sex as 0.5 float.
    Splits at the patient level to prevent leakage; stratifies by majority label.

    Returns (train, val, test) where each element is a list of dicts.
    """
    dataset_path = Path(args.dataset)
    with open(dataset_path, encoding="utf-8") as fh:
        all_entries = json.load(fh)

    binary = getattr(args, "binary", False)

    samples: list[dict] = []
    skipped = {"no_report": 0, "no_label": 0, "no_mask": 0, "intermediate": 0}
    for e in all_entries:
        if not e.get("report"):
            skipped["no_report"] += 1
            continue
        if not e.get("label"):
            skipped["no_label"] += 1
            continue
        if binary and e["label"] == "intermediate":
            skipped["intermediate"] += 1
            continue
        if not _mask_has_pixels(Path(e.get("mask") or "")):
            skipped["no_mask"] += 1
            continue
        sample = dict(e)
        age = e["age"]
        sample["age"] = float(age) if age != "unknown" else None
        sex = e["sex"]
        sample["sex"] = 1.0 if sex == "m" else (0.0 if sex == "f" else 0.5)
        samples.append(sample)

    intermediate_msg = f", {skipped['intermediate']} intermediate excluded" if binary else ""
    print(
        f"Complete samples (report+label+mask): {len(samples)}"
        f"  (skipped: {skipped['no_report']} no report, {skipped['no_label']} no label,"
        f" {skipped['no_mask']} no mask{intermediate_msg})"
    )

    if not samples:
        raise RuntimeError(f"No complete samples found. Check --dataset path: {dataset_path}")

    total = args.downstream_train_frac + args.downstream_val_frac + args.test_frac
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.4f}.")

    patient_groups = _group_by_patient(samples)

    def _majority_label(group: list[dict]) -> str:
        return Counter(s["label"] for s in group).most_common(1)[0][0]

    patient_labels = [_majority_label(g) for g in patient_groups]
    n_multi = sum(1 for g in patient_groups if len(g) > 1)
    print(
        f"Patients: {len(patient_groups)} total"
        f"  ({n_multi} with multiple images, {len(samples)} images total)"
    )

    train_val_groups, test_groups = _stratified_split_two(
        patient_groups, patient_labels, args.test_frac, args.seed
    )
    val_frac = args.downstream_val_frac / (args.downstream_train_frac + args.downstream_val_frac)
    train_groups, val_groups = _stratified_split_two(
        train_val_groups,
        [_majority_label(g) for g in train_val_groups],
        val_frac,
        args.seed,
    )

    train = [s for g in train_groups for s in g]
    val   = [s for g in val_groups   for s in g]
    test  = [s for g in test_groups  for s in g]

    known_ages = [s["age"] for s in train if s["age"] is not None]
    age_mean = float(sum(known_ages) / len(known_ages)) if known_ages else 0.0
    n_imputed = sum(1 for s in train + val + test if s["age"] is None)
    if n_imputed:
        print(f"  Imputing {n_imputed} unknown ages with training mean {age_mean:.1f} years")
    for s in train + val + test:
        if s["age"] is None:
            s["age"] = age_mean

    splits       = {"train": train,        "val": val,        "test": test}
    split_groups = {"train": train_groups, "val": val_groups, "test": test_groups}
    print("\nSplit statistics (patient-stratified):")
    for name, split in splits.items():
        dist     = dict(Counter(s["label"] for s in split))
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
        n_pats   = len(split_groups[name])
        print(f"  {name:<6} {len(split):>4} images  {n_pats:>4} patients  | {dist_str}")
    print()

    out_dir = run_dir if run_dir is not None else Path(args.out_dir) / "biomedclip_pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list] = {name: list(split) for name, split in splits.items()}
    manifest_path = out_dir / ("split_binary.json" if binary else "split.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  Split manifest saved -> {manifest_path}\n")

    return train, val, test


def build_pretrain_datasets(
    pretrain_samples: list[dict],
    preprocess_train,
    preprocess_val,
    tokenizer,
    use_mask: bool,
    seed: int,
    monitor_val_frac: float = 0.1,
) -> tuple[BoneTumorPairDataset, BoneTumorPairDataset]:
    """90/10 random split of pretrain samples into train and monitor-val datasets."""
    tuples = [(Path(s["image"]), Path(s["mask"]), s["report"]) for s in pretrain_samples]

    rng     = random.Random(seed)
    indices = list(range(len(tuples)))
    rng.shuffle(indices)
    split      = int(len(indices) * (1.0 - monitor_val_frac))
    train_samp = [tuples[i] for i in indices[:split]]
    val_samp   = [tuples[i] for i in indices[split:]]

    print(f"Pretrain loop split: {len(train_samp)} train / {len(val_samp)} monitor-val")

    train_ds = BoneTumorPairDataset(train_samp, preprocess_train, tokenizer, use_mask)
    val_ds   = BoneTumorPairDataset(val_samp,   preprocess_val,   tokenizer, use_mask)
    return train_ds, val_ds
