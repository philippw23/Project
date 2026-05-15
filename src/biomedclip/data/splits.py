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


def _old_load_all_samples(
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
    image_to_meta:    dict = {}
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

        try:
            age_raw = float(row["age"])
        except (ValueError, TypeError):
            age_raw = None
        sex_str = str(row["sex"]).strip().lower()
        sex_raw = 1.0 if sex_str == "m" else (0.0 if sex_str == "f" else None)

        image_to_meta[image_path] = {"patid": patid, "age": age_raw, "sex": sex_raw}

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
    return pretrain_cands, downstream_cands, image_to_meta, report_lookup


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


def old_build_stratified_splits(
    args,
    run_dir: Path | None = None,
) -> tuple[list, list, list, list]:
    """Load all samples and create independent pretrain and downstream splits."""

    pretrain_cands, downstream_cands, image_to_meta, report_lookup = _old_load_all_samples(
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
    n_excluded  = len(pretrain_cands) - len(pretrain)
    n_unlabeled = len(pretrain) - n_pretrain_with_label
    print(
        f"\nPretrain set: {len(pretrain_cands)} candidates"
        f" − {n_excluded} val/test exclusions"
        f" = {len(pretrain)} samples"
        f"\n  {n_pretrain_with_label} labeled (overlaps downstream_train, full metadata available)"
        f"\n  {n_unlabeled} unlabeled (report only, no label/age/sex)"
    )
    if n_unlabeled > 0:
        downstream_image_set = (
            {s[0] for s in downstream_train}
            | {s[0] for s in downstream_val}
            | {s[0] for s in test}
        )
        print("\n  Pretrain-only cases (not in any downstream split):")
        for s in pretrain:
            if s[0] not in downstream_image_set:
                patid = image_to_meta.get(s[0], {}).get("patid", "unknown")
                print(f"    patid={patid}  file={s[0].name}")
    print()

    out_dir = run_dir if run_dir is not None else Path(args.out_dir) / "biomedclip_pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup for enriching pretrain entries with downstream metadata.
    # "report" = befund_en + " " + beurteilung_en (concatenated in _load_all_samples).
    downstream_lookup = {s[0]: s for s in downstream_cands}

    manifest: dict[str, list] = {}
    pretrain_entries = []
    for s in pretrain:
        entry: dict = {"image": str(s[0]), "mask": str(s[1]), "report": s[2]}
        ds = downstream_lookup.get(s[0])
        if ds is not None:
            entry.update({
                "patid":               ds[9],
                "age":                 ds[7],
                "sex":                 ds[8],
                "label":               ds[6],
                "befund_en":           ds[2],
                "beurteilung_en":      ds[3],
                "befund_phrases":      ds[4],
                "beurteilung_phrases": ds[5],
            })
        else:
            meta  = image_to_meta.get(s[0], {})
            patid = meta.get("patid")
            rep   = report_lookup.get(patid, {}) if patid else {}
            entry.update({
                "patid":               patid,
                "age":                 meta.get("age"),
                "sex":                 meta.get("sex"),
                "label":               None,
                "befund_en":           rep.get("befund_en"),
                "beurteilung_en":      rep.get("beurteilung_en"),
                "befund_phrases":      rep.get("befund_phrases"),
                "beurteilung_phrases": rep.get("beurteilung_phrases"),
            })
        pretrain_entries.append(entry)
    manifest["pretrain"] = pretrain_entries
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


def old_build_pretrain_datasets(
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


# ---------------------------------------------------------------------------
# New split logic: single pool of complete cases (image + report + label + age + sex)
# ---------------------------------------------------------------------------

def _load_complete_samples(
    excel_path: Path,
    full_reports_path: Path,
    images_dir: Path,
    masks_dir: Path,
    english: bool = True,
) -> list[dict]:
    """Return only cases that have every required field.

    A case is included iff it has: image file, non-empty report, malignancy
    label, parseable age, and m/f sex.  Each element is a dict with keys:
        image, mask, report, label, age, sex, patid,
        befund_en, beurteilung_en, befund_phrases, beurteilung_phrases
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

    samples: list[dict] = []
    skipped = {"no_image": 0, "no_report": 0, "no_label": 0, "no_age": 0, "no_sex": 0}

    for _, row in df.iterrows():
        stem       = Path(row["filename"]).stem
        patid      = row["patid"]
        label      = row["malignancy"]
        image_path = images_dir / f"{stem}.png"
        mask_path  = masks_dir  / f"{stem}.png"

        if not image_path.exists():
            skipped["no_image"] += 1
            continue

        rep = report_lookup.get(patid, {})
        b1  = (rep.get("befund_en") if english else rep.get("befund")) or ""
        b2  = (rep.get("beurteilung_en") if english else rep.get("beurteilung")) or ""
        report_text = " ".join(filter(None, [b1, b2]))
        if not report_text:
            skipped["no_report"] += 1
            continue

        if not label:
            skipped["no_label"] += 1
            continue

        try:
            age_float = float(row["age"])
        except (ValueError, TypeError):
            skipped["no_age"] += 1
            continue

        sex_str = str(row["sex"]).strip().lower()
        if sex_str == "m":
            sex_float = 1.0
        elif sex_str == "f":
            sex_float = 0.0
        else:
            skipped["no_sex"] += 1
            continue

        samples.append({
            "image":               image_path,
            "mask":                mask_path,
            "report":              report_text,
            "label":               label,
            "age":                 age_float,
            "sex":                 sex_float,
            "patid":               patid,
            "befund_en":           rep.get("befund_en"),
            "beurteilung_en":      rep.get("beurteilung_en"),
            "befund_phrases":      rep.get("befund_phrases"),
            "beurteilung_phrases": rep.get("beurteilung_phrases"),
        })

    print(
        f"Complete samples (image+report+label+age+sex): {len(samples)}"
        f"  (skipped: {skipped['no_image']} no image, {skipped['no_report']} no report,"
        f" {skipped['no_label']} no label, {skipped['no_age']} no age, {skipped['no_sex']} no sex)"
    )
    return samples


def _group_by_patient(samples: list[dict]) -> list[list[dict]]:
    """Return samples grouped by patid, preserving insertion order."""
    groups: dict[str, list[dict]] = {}
    for s in samples:
        groups.setdefault(s["patid"], []).append(s)
    return list(groups.values())


def build_stratified_splits(
    args,
    run_dir: Path | None = None,
) -> tuple[list, list, list]:
    """Create train / val / test splits from cases that have all required fields.

    Splitting is done at the *patient* level so that all images from the same
    patient land in the same fold (preventing leakage).  Stratification uses
    the majority label among a patient's images.

    Every case in every split has: image, mask, report, label, age, sex.
    Returns (train, val, test) where each element is a list of dicts.
    """
    samples = _load_complete_samples(
        excel_path=Path(args.excel),
        full_reports_path=Path(args.reports),
        images_dir=Path(args.images),
        masks_dir=Path(args.masks),
        english=args.english,
    )

    if not samples:
        raise RuntimeError(
            "No complete samples found. Check --excel, --images, --reports paths."
        )

    total = args.downstream_train_frac + args.downstream_val_frac + args.test_frac
    if abs(total - 1.0) > 1e-4:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total:.4f}."
        )

    # Group by patient; use majority label per patient for stratification.
    patient_groups = _group_by_patient(samples)
    def _majority_label(group: list[dict]) -> str:
        return Counter(s["label"] for s in group).most_common(1)[0][0]
    patient_labels = [_majority_label(g) for g in patient_groups]

    n_multi = sum(1 for g in patient_groups if len(g) > 1)
    print(
        f"Patients: {len(patient_groups)} total"
        f"  ({n_multi} with multiple images, {len(samples)} images total)"
    )

    # Split at patient level first, then flatten to images.
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

    splits        = {"train": train,        "val": val,        "test": test}
    split_groups  = {"train": train_groups, "val": val_groups, "test": test_groups}
    print("\nSplit statistics (patient-stratified; all cases have image + report + label + age + sex):")
    for name, split in splits.items():
        dist     = dict(Counter(s["label"] for s in split))
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
        n_pats   = len(split_groups[name])
        print(f"  {name:<6} {len(split):>4} images  {n_pats:>4} patients  | {dist_str}")
    print()

    out_dir = run_dir if run_dir is not None else Path(args.out_dir) / "biomedclip_pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list] = {}
    for name, split in splits.items():
        manifest[name] = [
            {
                "image":               str(s["image"]),
                "mask":                str(s["mask"]),
                "report":              s["report"],
                "label":               s["label"],
                "age":                 s["age"],
                "sex":                 s["sex"],
                "patid":               s["patid"],
                "befund_en":           s["befund_en"],
                "beurteilung_en":      s["beurteilung_en"],
                "befund_phrases":      s["befund_phrases"],
                "beurteilung_phrases": s["beurteilung_phrases"],
            }
            for s in split
        ]

    manifest_path = out_dir / "split.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"  Split manifest saved -> {manifest_path}\n")
    return train, val, test
