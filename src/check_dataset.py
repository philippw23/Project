"""Dataset completeness check.

Takes the union of all stems from images/, segmentations/, and Excel as the
total population, then reports what each source is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXCEL   = ROOT_DIR / "data" / "internal_dataset" / "metadata.xlsx"
DEFAULT_REPORTS = ROOT_DIR / "data" / "internal_dataset" / "text" / "sanitized_reports.json"
DEFAULT_IMAGES  = ROOT_DIR / "data" / "internal_dataset" / "images"
DEFAULT_MASKS   = ROOT_DIR / "data" / "internal_dataset" / "segmentations"


def _normalise_id(val: str) -> str:
    try:
        return str(int(float(val)))
    except ValueError:
        return val


def check_dataset(
    excel_path: Path = DEFAULT_EXCEL,
    reports_path: Path = DEFAULT_REPORTS,
    images_dir: Path = DEFAULT_IMAGES,
    masks_dir: Path = DEFAULT_MASKS,
) -> None:

    # --- Collect all stems from each source ---
    image_stems = {p.stem for p in images_dir.glob("*.png")}
    mask_stems  = {p.stem for p in masks_dir.glob("*.png")}

    df = pd.read_excel(
        excel_path,
        sheet_name="internal_data_matched",
        usecols=[0, 2, 7],
        skiprows=1,
        header=0,
        dtype=str,
        engine="openpyxl",
    )
    df.columns = ["file_name", "malignancy", "anonym_patid"]
    df = df.dropna(subset=["file_name"])
    df["file_name"]    = df["file_name"].str.strip().apply(lambda x: Path(x).stem)
    df["anonym_patid"] = df["anonym_patid"].fillna("").str.strip().apply(_normalise_id)
    df["malignancy"]   = df["malignancy"].fillna("").str.strip().str.lower()
    excel_stems = set(df["file_name"])

    # Union of all three sources = total population
    all_stems = image_stems | mask_stems | excel_stems
    total = len(all_stems)

    print(f"{'='*60}")
    print(f"Source counts")
    print(f"{'='*60}")
    print(f"  Images (images/):        {len(image_stems)}")
    print(f"  Masks  (segmentations/): {len(mask_stems)}")
    print(f"  Excel rows:              {len(excel_stems)}")
    print(f"  Union (total):           {total}")
    print()

    # --- Load reports ---
    with open(reports_path, encoding="utf-8") as fh:
        raw_reports = json.load(fh)

    report_pids: set[str] = set()
    stem_to_patid = dict(zip(df["file_name"], df["anonym_patid"]))
    for entry in raw_reports:
        pid = _normalise_id(str(entry.get("patid", "")).strip())
        befund      = (entry.get("befund")      or "").strip()
        beurteilung = (entry.get("beurteilung") or "").strip()
        if pid and (befund or beurteilung):
            report_pids.add(pid)

    print(f"Unique patids with report in sanitized_reports.json: {len(report_pids)}")
    excel_pids = set(df["anonym_patid"].unique()) - {""}
    print(f"Unique patids in Excel: {len(excel_pids)}")
    print(f"Excel patids covered by reports: {len(excel_pids & report_pids)} / {len(excel_pids)}")
    print()

    # --- Patids appearing more than once in Excel ---
    pid_counts = df[df["anonym_patid"] != ""]["anonym_patid"].value_counts()
    duplicates = pid_counts[pid_counts > 1]
    print(f"{'='*60}")
    print(f"Patids with more than 1 image in Excel: {len(duplicates)}")
    print(f"{'='*60}")
    for pid, count in duplicates.items():
        filenames = df[df["anonym_patid"] == pid]["file_name"].tolist()
        print(f"  patid={pid}  ({count}x)  ->  {filenames}")
    print()

    # --- Per-stem checks across the full population ---
    missing_image    : list[str] = []
    missing_mask     : list[str] = []
    missing_excel    : list[str] = []
    missing_report   : list[str] = []
    missing_label    : list[str] = []
    truncated_images : list[str] = []

    for stem in sorted(all_stems):
        image_path = images_dir / f"{stem}.png"
        mask_path  = masks_dir  / f"{stem}.png"

        if stem not in image_stems:
            missing_image.append(stem)
        else:
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except (UnidentifiedImageError, Exception):
                truncated_images.append(stem)

        if stem not in mask_stems:
            missing_mask.append(stem)

        if stem not in excel_stems:
            missing_excel.append(stem)
            # No Excel entry → no patid → no report possible
            if stem in image_stems:
                missing_report.append(f"{stem}  (no Excel entry)")
        else:
            row = df[df["file_name"] == stem].iloc[0]
            patid = row["anonym_patid"]
            if not patid or patid not in report_pids:
                if stem in image_stems:
                    missing_report.append(f"{stem}  (patid={patid or 'n/a'})")
            if not row["malignancy"]:
                missing_label.append(stem)

    # --- Summary ---
    n_images = len(image_stems)
    n_excel  = len(excel_stems)

    def _section(title: str, items: list[str], n_total: int) -> None:
        print(f"{'='*60}")
        print(f"{title}: {len(items)} / {n_total}")
        print(f"{'='*60}")
        for item in items:
            print(f"  {item}")
        if not items:
            print("  (none)")
        print()

    _section("Images without Excel entry",     missing_excel,     n_images)
    _section("Images without mask",            missing_mask,      n_images)
    _section("Excel entries without image",    missing_image,     n_excel)
    _section("Missing reports",                missing_report,    n_images)
    _section("Missing malignancy label",       missing_label,     n_excel)
    _section("Truncated / corrupt images",     truncated_images,  n_images)

    # --- Pre-compute per-stem boolean flags ---
    has_image  = image_stems
    has_mask   = mask_stems
    has_excel  = excel_stems
    has_report = {
        s for s in image_stems
        if stem_to_patid.get(s, "") in report_pids
    }
    has_label = {
        s for s in excel_stems
        if df[df["file_name"] == s].iloc[0]["malignancy"]
    }

    # --- Funnel: how many drop out at each step ---
    steps = [
        ("all images",                    has_image),
        ("+ excel entry",                 has_image & has_excel),
        ("+ report",                      has_image & has_excel & has_report),
        ("+ label",                       has_image & has_excel & has_report & has_label),
        ("+ mask",                        has_image & has_excel & has_report & has_label & has_mask),
    ]

    print(f"{'='*60}")
    print("Funnel: cumulative requirements")
    print(f"{'='*60}")
    prev_set = None
    for name, s in steps:
        if prev_set is None:
            print(f"  {len(s):>5}  {name}")
        else:
            lost = prev_set - s
            print(f"  {len(s):>5}  {name}  (-{len(lost)} without this)")
        prev_set = s
    print()

    # --- Same-set check across all combinations ---
    all_combos = {
        "image + excel + report + label + mask": has_image & has_excel & has_report & has_label & has_mask,
        "image + excel + report + label":        has_image & has_excel & has_report & has_label,
        "image + excel + report":                has_image & has_excel & has_report,
        "image + excel + label":                 has_image & has_excel & has_label,
        "image + excel":                         has_image & has_excel,
        "image + mask":                          has_image & has_mask,
        "image":                                 has_image,
    }

    print(f"{'='*60}")
    print("All combinations (with identity check)")
    print(f"{'='*60}")
    seen: list[tuple[str, set]] = []
    for name, s in all_combos.items():
        same_as = next((n for n, t in seen if t == s), None)
        suffix = f"  ← same as '{same_as}'" if same_as else ""
        print(f"  {len(s):>5}  {name}{suffix}")
        seen.append((name, s))


if __name__ == "__main__":
    check_dataset()
