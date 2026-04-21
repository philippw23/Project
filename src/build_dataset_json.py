"""Build a unified dataset JSON with one entry per image.

Each entry contains all fields from sanitized_reports.json (matched via patid)
plus image, mask, and label. Missing values are empty strings.

Output: data/dataset.json  (1267 entries, one per image in data/images/)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT_DIR      = Path(__file__).resolve().parent.parent
DEFAULT_EXCEL   = ROOT_DIR / "data" / "metadata.xlsx"
DEFAULT_REPORTS = ROOT_DIR / "data" / "text" / "sanitized_reports.json"
DEFAULT_IMAGES  = ROOT_DIR / "data" / "images"
DEFAULT_MASKS   = ROOT_DIR / "data" / "segmentations"
DEFAULT_OUT     = ROOT_DIR / "data" / "dataset.json"


def _normalise_id(val: str) -> str:
    try:
        return str(int(float(val)))
    except ValueError:
        return val


def build(
    excel_path: Path = DEFAULT_EXCEL,
    reports_path: Path = DEFAULT_REPORTS,
    images_dir: Path = DEFAULT_IMAGES,
    masks_dir: Path = DEFAULT_MASKS,
    out_path: Path = DEFAULT_OUT,
) -> None:

    # --- Load Excel: filename, malignancy, patid ---
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

    # stem → {malignancy, patid}
    excel_lookup: dict[str, dict] = {
        row["file_name"]: {
            "malignancy":   row["malignancy"],
            "anonym_patid": row["anonym_patid"],
        }
        for _, row in df.iterrows()
    }

    # --- Load reports: patid → full report dict ---
    with open(reports_path, encoding="utf-8") as fh:
        raw_reports = json.load(fh)

    report_lookup: dict[str, dict] = {}
    for entry in raw_reports:
        pid = _normalise_id(str(entry.get("patid", "")).strip())
        if pid:
            report_lookup[pid] = entry

    # --- Build one entry per image ---
    image_paths = sorted(images_dir.glob("*.png"))
    entries: list[dict] = []

    for img_path in image_paths:
        stem = img_path.stem
        mask_path = masks_dir / img_path.name

        excel_row = excel_lookup.get(stem, {})
        patid     = excel_row.get("anonym_patid", "")
        report    = report_lookup.get(patid, {})

        entry: dict = {
            "image":        str(img_path),
            "mask":         str(mask_path) if mask_path.exists() else "",
            "label":        excel_row.get("malignancy", ""),
            "patid":        patid,
            "befund":       str(report.get("befund",       "") or ""),
            "beurteilung":  str(report.get("beurteilung",  "") or ""),
        }

        entries.append(entry)

    # --- Write output ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)

    # --- Stats ---
    n_with_report = sum(1 for e in entries if e["befund"] or e["beurteilung"])
    n_with_mask   = sum(1 for e in entries if e["mask"])
    n_with_label  = sum(1 for e in entries if e["label"])
    print(f"Written {len(entries)} entries to {out_path}")
    print(f"  with report : {n_with_report} / {len(entries)}")
    print(f"  with mask   : {n_with_mask}   / {len(entries)}")
    print(f"  with label  : {n_with_label}  / {len(entries)}")


if __name__ == "__main__":
    build()
