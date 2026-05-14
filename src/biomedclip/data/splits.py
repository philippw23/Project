from __future__ import annotations

import json
import random
import warnings
from collections import Counter
from pathlib import Path

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


def _load_all_samples(
    excel_path: Path,
    full_reports_path: Path,
    images_dir: Path,
    masks_dir: Path,
    english: bool = True,
) -> tuple[list, list]:
    """Build two independent sample pools with different requirements.

    Returns
    -------
    pretrain_cands : list of (image_path, mask_path, report_text)
    downstream_cands : list of
        (image_path, mask_path,
         befund_en, beurteilung_en, befund_phrases, beurteilung_phrases,
         label, age, sex, patid)
    """
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
        raise RuntimeError(f"Cannot read Excel file '{excel_path}': {exc}") from exc

    df.columns = ["filename", "malignancy", "age", "sex", "patid"]
    df = df.dropna(subset=["filename", "patid"])
    df["filename"]   = df["filename"].str.strip()
    df["patid"]      = df["patid"].str.strip()
    df["malignancy"] = df["malignancy"].fillna("").str.strip().str.lower()
    df["age"]        = df["age"].fillna("")
    df["sex"]        = df["sex"].fillna("")

    def _normalise_id(val: str) -> str:
        try:
            return str(int(float(val)))
        except ValueError:
            return val

    df["patid"] = df["patid"].apply(_normalise_id)

    with open(full_reports_path, encoding="utf-8") as fh:
        raw_reports = json.load(fh)

    report_lookup: dict[str, dict] = {}
    for entry in raw_reports:
        pid = _normalise_id(str(entry.get("patid", "")).strip())
        if not pid:
            continue
        report_lookup[pid] = {
            "befund":              (entry.get("befund")              or "").strip() or None,
            "beurteilung":         (entry.get("beurteilung")         or "").strip() or None,
            "befund_en":           (entry.get("befund_en")           or "").strip() or None,
            "beurteilung_en":      (entry.get("beurteilung_en")      or "").strip() or None,
            "befund_phrases":      entry.get("befund_phrases")       or None,
            "beurteilung_phrases": entry.get("beurteilung_phrases")  or None,
        }

    pretrain_cands:   list = []
    downstream_cands: list = []
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

        rep            = report_lookup.get(patid, {})
        befund_en      = rep.get("befund_en")
        beurteilung_en = rep.get("beurteilung_en")
        befund_phrases      = rep.get("befund_phrases")
        beurteilung_phrases = rep.get("beurteilung_phrases")

        b1 = (rep.get("befund_en") if english else rep.get("befund")) or ""
        b2 = (rep.get("beurteilung_en") if english else rep.get("beurteilung")) or ""
        report_text = " ".join(filter(None, [b1, b2]))
        if report_text:
            pretrain_cands.append((image_path, mask_path, report_text))

        if label:
            try:
                age_float = float(row["age"])
            except (ValueError, TypeError):
                skipped_no_age_sex += 1
                continue
            sex_raw = str(row["sex"]).strip().lower()
            if sex_raw == "m":
                sex_float = 1.0
            elif sex_raw == "f":
                sex_float = 0.0
            else:
                skipped_no_age_sex += 1
                continue
            downstream_cands.append((
                image_path, mask_path,
                befund_en, beurteilung_en, befund_phrases, beurteilung_phrases,
                label, age_float, sex_float, patid,
            ))

    print(
        f"Dataset: {len(pretrain_cands)} pretrain candidates (image+report), "
        f"{len(downstream_cands)} downstream candidates (image+label+age+sex) "
        f"(skipped: {skipped_no_image} missing images, "
        f"{skipped_no_age_sex} missing/invalid age or sex)"
    )
    return pretrain_cands, downstream_cands


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


def build_stratified_splits(
    args,
    run_dir: Path | None = None,
) -> tuple[list, list, list, list]:
    """Load all samples and create independent pretrain and downstream splits."""

    pretrain_cands, downstream_cands = _load_all_samples(
        excel_path=Path(args.excel),
        full_reports_path=Path(args.reports),
        images_dir=Path(args.images),
        masks_dir=Path(args.masks),
        english=args.english,
    )

    if not downstream_cands:
        raise RuntimeError(
            "No downstream candidates found (need image + label + age/sex). "
            "Check --excel, --images paths."
        )
    if not pretrain_cands:
        raise RuntimeError(
            "No pretrain candidates found (need image + report). "
            "Check --reports path."
        )

    total = args.downstream_train_frac + args.downstream_val_frac + args.test_frac
    if abs(total - 1.0) > 1e-4:
        raise ValueError(
            f"Split fractions must sum to 1.0 using "
            f"--downstream_train_frac + --downstream_val_frac + --test_frac, "
            f"got {total:.4f}."
        )

    no_report  = [s for s in downstream_cands if not (s[2] or s[3])]
    has_report = [s for s in downstream_cands if s[2] or s[3]]

    n_total       = len(downstream_cands)
    n_test        = max(1, round(n_total * args.test_frac))
    n_val         = max(1, round(n_total * args.downstream_val_frac))
    n_from_no_rep = min(len(no_report), n_val + n_test)
    n_supplement  = (n_val + n_test) - n_from_no_rep

    if n_from_no_rep == len(no_report):
        no_rep_pool, no_rep_train = no_report, []
    else:
        frac = n_from_no_rep / len(no_report)
        labels_nr = [s[6] for s in no_report]
        no_rep_train, no_rep_pool = _stratified_split_two(no_report, labels_nr, frac, args.seed)

    if n_supplement > 0 and has_report:
        frac = n_supplement / len(has_report)
        labels_hr = [s[6] for s in has_report]
        has_rep_train, has_rep_pool = _stratified_split_two(has_report, labels_hr, frac, args.seed)
    else:
        has_rep_pool, has_rep_train = [], has_report

    val_test_pool     = no_rep_pool + has_rep_pool
    labels_vt         = [s[6] for s in val_test_pool]
    test_frac_of_pool = n_test / len(val_test_pool)
    downstream_val, test = _stratified_split_two(val_test_pool, labels_vt, test_frac_of_pool, args.seed)

    downstream_train = no_rep_train + has_rep_train

    excluded_images = {s[0] for s in downstream_val} | {s[0] for s in test}
    pretrain = [s for s in pretrain_cands if s[0] not in excluded_images]

    if not pretrain:
        raise RuntimeError("Pretrain set is empty after excluding val/test images.")

    def _dist(split: list, label_idx: int) -> dict[str, int]:
        return dict(Counter(s[label_idx] for s in split))

    def _count_reports(split: list, report_idx: int) -> int:
        return sum(1 for s in split if s[report_idx] is not None and str(s[report_idx]).strip())

    downstream_splits = {
        "downstream_train": downstream_train,
        "downstream_val":   downstream_val,
        "test":             test,
    }
    print("\nDownstream split statistics (mutually exclusive):")
    for name, split in downstream_splits.items():
        dist = _dist(split, label_idx=6)
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
        n_reports = _count_reports(split, report_idx=2)
        print(
            f"  {name:<20} {len(split):>4} samples"
            f"  | reports: {n_reports:>4}"
            f"  | {dist_str}"
        )

    n_pretrain_with_label = sum(
        1 for s in pretrain if s[0] in {ds[0] for ds in downstream_train}
    )
    print(
        f"\nPretrain set ({len(pretrain)} samples, val/test excluded):"
        f"\n  {len(pretrain_cands) - len(pretrain)} val/test images removed"
        f"\n  {n_pretrain_with_label} samples overlap with downstream_train"
        f"\n  {len(pretrain) - n_pretrain_with_label} are unlabeled (report only)"
    )
    print()

    out_dir = run_dir if run_dir is not None else Path(args.out_dir) / "biomedclip_pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list] = {}
    manifest["pretrain"] = [
        {"image": str(s[0]), "mask": str(s[1]), "report": s[2]}
        for s in pretrain
    ]
    for name, split in downstream_splits.items():
        manifest[name] = [
            {
                "image":               str(s[0]),
                "mask":                str(s[1]),
                "befund_en":           s[2],
                "beurteilung_en":      s[3],
                "befund_phrases":      s[4],
                "beurteilung_phrases": s[5],
                "label":               s[6],
                "age":                 s[7],
                "sex":                 s[8],
                "patid":               s[9],
            }
            for s in split
        ]

    manifest_path = out_dir / "splits.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"  Split manifest saved -> {manifest_path}\n")
    return pretrain, downstream_train, downstream_val, test


def build_pretrain_datasets(
    pretrain_samples: list,
    preprocess_train,
    preprocess_val,
    tokenizer,
    use_mask: bool,
    seed: int,
    monitor_val_frac: float = 0.1,
) -> tuple[BoneTumorPairDataset, BoneTumorPairDataset]:
    """Split the pretraining samples into a training and a monitoring-val set."""
    unlabeled = list(pretrain_samples)

    rng     = random.Random(seed)
    indices = list(range(len(unlabeled)))
    rng.shuffle(indices)
    split      = int(len(indices) * (1.0 - monitor_val_frac))
    train_samp = [unlabeled[i] for i in indices[:split]]
    val_samp   = [unlabeled[i] for i in indices[split:]]

    print(f"Pretrain loop split: {len(train_samp)} train / {len(val_samp)} monitor-val")

    train_ds = BoneTumorPairDataset(train_samp, preprocess_train, tokenizer, use_mask)
    val_ds   = BoneTumorPairDataset(val_samp,   preprocess_val,   tokenizer, use_mask)
    return train_ds, val_ds
