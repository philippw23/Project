from __future__ import annotations

import json
import random
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

from biomedclip.data.splits import _stratified_split_two
from LACE.data.datasets import InternalDatasetV2, InternalTripleDataset


def _normalise_id(val: str) -> str:
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val)


def build_lace_splits(
    args,
    run_dir: Path | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Load internal dataset and build stratified splits for LACE.

    Stores befund and beurteilung as separate fields (not concatenated).
    Pretrain samples require at least beurteilung for L_ITA; befund is optional.

    Returns (pretrain, downstream_train, downstream_val, test).
    """
    excel_path   = Path(args.excel)
    reports_path = Path(args.reports)
    images_dir   = Path(args.images)
    masks_dir    = Path(args.masks)

    try:
        df = pd.read_excel(
            excel_path,
            sheet_name="internal_data_matched",
            usecols=[0, 2, 4, 5, 7],
            skiprows=1,
            header=0,
            dtype=str,
            engine="openpyxl",
        )
    except Exception as exc:
        raise RuntimeError(f"Cannot read Excel '{excel_path}': {exc}") from exc

    df.columns    = ["filename", "malignancy", "age", "sex", "patid"]
    df = df.dropna(subset=["filename", "patid"])
    df["filename"]   = df["filename"].str.strip()
    df["patid"]      = df["patid"].str.strip()
    df["malignancy"] = df["malignancy"].fillna("").str.strip().str.lower()
    df["age"]        = df["age"].fillna("")
    df["sex"]        = df["sex"].fillna("")
    df["patid"]      = df["patid"].apply(_normalise_id)

    with open(reports_path, encoding="utf-8") as fh:
        raw_reports = json.load(fh)

    # Store befund and beurteilung as separate fields
    report_lookup: dict[str, dict] = {}
    for entry in raw_reports:
        pid = _normalise_id(str(entry.get("patid", "")).strip())
        if not pid:
            continue
        befund      = (entry.get("befund") or "").strip()
        beurteilung = (entry.get("beurteilung") or "").strip()
        if befund or beurteilung:
            report_lookup[pid] = {"befund": befund, "beurteilung": beurteilung}

    pretrain_cands:    list[dict] = []
    downstream_cands:  list[dict] = []
    skipped_no_image   = 0
    skipped_no_age_sex = 0

    for _, row in df.iterrows():
        stem       = Path(row["filename"]).stem
        patid      = row["patid"]
        label      = row["malignancy"]
        image_path = images_dir / f"{stem}.png"
        mask_path  = masks_dir  / f"{stem}.png"

        if not image_path.exists():
            skipped_no_image += 1
            continue

        rep = report_lookup.get(patid)

        if rep and rep.get("beurteilung"):
            pretrain_cands.append({
                "image":       str(image_path),
                "mask":        str(mask_path),
                "befund":      rep.get("befund", ""),
                "beurteilung": rep["beurteilung"],
            })

        if label:
            try:
                float(row["age"])
            except (ValueError, TypeError):
                skipped_no_age_sex += 1
                continue
            if str(row["sex"]).strip().lower() not in ("m", "male", "1", "f", "female", "0"):
                skipped_no_age_sex += 1
                continue
            sex_raw = str(row["sex"]).strip().lower()
            downstream_cands.append({
                "image":       str(image_path),
                "mask":        str(mask_path),
                "befund":      rep.get("befund", "") if rep else "",
                "beurteilung": rep.get("beurteilung", "") if rep else "",
                "label":       label,
                "age":         float(row["age"]),
                "sex":         1.0 if sex_raw in ("m", "male", "1") else 0.0,
            })

    print(
        f"LACE dataset: {len(pretrain_cands)} pretrain, {len(downstream_cands)} downstream "
        f"(skipped: {skipped_no_image} missing images, {skipped_no_age_sex} missing age/sex)"
    )

    if not downstream_cands:
        raise RuntimeError("No downstream candidates found. Check --excel and --images paths.")
    if not pretrain_cands:
        raise RuntimeError("No pretrain candidates found. Check --reports path.")

    total = args.downstream_train_frac + args.downstream_val_frac + args.test_frac
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.4f}.")

    labels     = [s["label"] for s in downstream_cands]
    n_total    = len(downstream_cands)
    n_test     = max(1, round(n_total * args.test_frac))
    test_frac  = n_test / n_total
    rest, test = _stratified_split_two(downstream_cands, labels, test_frac, args.seed)

    labels_rest          = [s["label"] for s in rest]
    val_frac_of_rest     = args.downstream_val_frac / (1.0 - args.test_frac)
    downstream_train, downstream_val = _stratified_split_two(
        rest, labels_rest, val_frac_of_rest, args.seed
    )

    excluded = {s["image"] for s in downstream_val} | {s["image"] for s in test}
    pretrain  = [s for s in pretrain_cands if s["image"] not in excluded]

    if not pretrain:
        raise RuntimeError("Pretrain set is empty after excluding val/test images.")

    print("\nLACE split statistics:")
    for name, split in [
        ("pretrain", pretrain),
        ("downstream_train", downstream_train),
        ("downstream_val", downstream_val),
        ("test", test),
    ]:
        dist = dict(Counter(s.get("label", "pretrain") for s in split))
        print(f"  {name:<20} {len(split):>4} samples  | {dist}")
    print()

    out_dir = run_dir if run_dir is not None else Path(args.out_dir) / "lace_pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pretrain":          pretrain,
        "downstream_train":  downstream_train,
        "downstream_val":    downstream_val,
        "test":              test,
    }
    manifest_path = out_dir / "splits.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  Split manifest saved -> {manifest_path}\n")

    return pretrain, downstream_train, downstream_val, test


def build_pretrain_datasets_lace(
    pretrain_samples: list[dict],
    preprocess_train,
    preprocess_val,
    tokenizer,
    seed: int,
    monitor_val_frac: float = 0.1,
    max_text_len: int = 128,
) -> tuple[InternalTripleDataset, InternalTripleDataset]:
    """90/10 random split of pretrain_samples into train and monitor-val datasets."""
    rng     = random.Random(seed)
    indices = list(range(len(pretrain_samples)))
    rng.shuffle(indices)
    split      = int(len(indices) * (1.0 - monitor_val_frac))
    train_samp = [pretrain_samples[i] for i in indices[:split]]
    val_samp   = [pretrain_samples[i] for i in indices[split:]]

    print(f"Pretrain loop split: {len(train_samp)} train / {len(val_samp)} monitor-val")

    train_ds = InternalTripleDataset(train_samp, preprocess_train, tokenizer, max_text_len)
    val_ds   = InternalTripleDataset(val_samp,   preprocess_val,   tokenizer, max_text_len)
    return train_ds, val_ds


def build_pretrain_datasets_lace_v2(
    pretrain_samples: list[dict],
    preprocess_train,
    preprocess_val,
    tokenizer,
    seed: int,
    monitor_val_frac: float = 0.1,
    max_text_len: int = 128,
) -> tuple[InternalDatasetV2, InternalDatasetV2]:
    """90/10 random split of pretrain_samples into train and monitor-val datasets (v2)."""
    rng     = random.Random(seed)
    indices = list(range(len(pretrain_samples)))
    rng.shuffle(indices)
    split      = int(len(indices) * (1.0 - monitor_val_frac))
    train_samp = [pretrain_samples[i] for i in indices[:split]]
    val_samp   = [pretrain_samples[i] for i in indices[split:]]

    print(f"Pretrain loop split (v2): {len(train_samp)} train / {len(val_samp)} monitor-val")

    train_ds = InternalDatasetV2(train_samp, preprocess_train, tokenizer, max_text_len)
    val_ds   = InternalDatasetV2(val_samp,   preprocess_val,   tokenizer, max_text_len)
    return train_ds, val_ds
