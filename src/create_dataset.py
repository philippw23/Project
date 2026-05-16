"""Assemble data/internal_dataset/dataset_full.json from metadata.xlsx and full_reports.json.

Reads metadata.xlsx row by row; for each row looks up the corresponding report
in full_reports.json using report_accnr (preferred) or patid (fallback).
A row is included only if the image file exists on disk.
Report fields are null when no matching report is found.

Usage:
    python src/create_dataset.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT_DIR          = Path(__file__).resolve().parent.parent
EXCEL_PATH        = ROOT_DIR / "data" / "internal_dataset" / "metadata.xlsx"
FULL_REPORTS_PATH = ROOT_DIR / "data" / "internal_dataset" / "text" / "full_reports.json"
IMAGES_DIR        = ROOT_DIR / "data" / "internal_dataset" / "images"
MASKS_DIR         = ROOT_DIR / "data" / "internal_dataset" / "segmentations"
OUTPUT_PATH       = ROOT_DIR / "data" / "internal_dataset" / "dataset_full.json"


def _normalise_id(val: str) -> str:
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val)


def main() -> None:
    df = pd.read_excel(
        EXCEL_PATH,
        sheet_name="internal_data_matched",
        usecols=[0, 2, 4, 5, 6, 7],
        skiprows=1,
        header=None,
        dtype=str,
        engine="openpyxl",
    )
    df.columns = ["file_name", "malignancy", "age", "sex", "report_accnr", "anonym_patid"]
    df = df.dropna(subset=["file_name"])
    df["file_name"]    = df["file_name"].str.strip()
    df["malignancy"]   = df["malignancy"].fillna("").str.strip().str.lower()
    df["age"]          = df["age"].fillna("").str.strip()
    df["sex"]          = df["sex"].fillna("").str.strip().str.lower()
    df["report_accnr"] = df["report_accnr"].fillna("").str.strip()
    df["anonym_patid"] = df["anonym_patid"].fillna("").str.strip()
    df["anonym_patid"] = df["anonym_patid"].apply(lambda x: _normalise_id(x) if x else "")

    with open(FULL_REPORTS_PATH, encoding="utf-8") as fh:
        raw_reports = json.load(fh)

    accnr_lookup: dict[str, dict] = {}
    patid_lookup: dict[str, dict] = {}
    for entry in raw_reports:
        accnr = str(entry.get("accnr", "")).strip()
        patid = _normalise_id(str(entry.get("patid", "")).strip())
        report_data = {
            "befund":              (entry.get("befund")              or "").strip() or None,
            "beurteilung":         (entry.get("beurteilung")         or "").strip() or None,
            "befund_en":           (entry.get("befund_en")           or "").strip() or None,
            "beurteilung_en":      (entry.get("beurteilung_en")      or "").strip() or None,
            "befund_phrases":      entry.get("befund_phrases")       or [],
            "beurteilung_phrases": entry.get("beurteilung_phrases")  or [],
        }
        if accnr:
            accnr_lookup[accnr] = report_data
        if patid:
            patid_lookup[patid] = report_data

    entries = []
    skipped_no_image = 0

    for _, row in df.iterrows():
        stem       = Path(row["file_name"]).stem
        image_path = IMAGES_DIR / f"{stem}.png"
        mask_path  = MASKS_DIR  / f"{stem}.png"

        if not image_path.exists():
            skipped_no_image += 1
            continue

        report_accnr = row["report_accnr"]
        patid        = row["anonym_patid"]

        rep: dict | None = None
        if report_accnr:
            rep = accnr_lookup.get(report_accnr)
        elif patid:
            rep = patid_lookup.get(patid)

        if rep is not None:
            befund              = rep["befund"]
            beurteilung         = rep["beurteilung"]
            befund_en           = rep["befund_en"]
            beurteilung_en      = rep["beurteilung_en"]
            befund_phrases      = rep["befund_phrases"]
            beurteilung_phrases = rep["beurteilung_phrases"]
            b1 = befund_en or ""
            b2 = beurteilung_en or ""
            report = " ".join(filter(None, [b1, b2])) or None
        else:
            befund = beurteilung = befund_en = beurteilung_en = report = None
            befund_phrases = beurteilung_phrases = []

        try:
            age: float | str = float(row["age"])
        except (ValueError, TypeError):
            age = "unknown"

        sex_raw = row["sex"]
        if sex_raw == "m":
            sex = "m"
        elif sex_raw == "f":
            sex = "f"
        else:
            sex = "unknown"

        label = row["malignancy"] or None

        entries.append({
            "image":               str(image_path),
            "mask":                str(mask_path),
            "report":              report,
            "label":               label,
            "age":                 age,
            "sex":                 sex,
            "patid":               patid,
            "befund":              befund,
            "beurteilung":         beurteilung,
            "befund_en":           befund_en,
            "beurteilung_en":      beurteilung_en,
            "befund_phrases":      befund_phrases,
            "beurteilung_phrases": beurteilung_phrases,
        })

    print(f"Total entries: {len(entries)} (skipped {skipped_no_image} missing images)")
    n_with_report = sum(1 for e in entries if e["report"] is not None)
    n_with_label  = sum(1 for e in entries if e["label"]  is not None)
    n_unknown_age = sum(1 for e in entries if e["age"] == "unknown")
    n_unknown_sex = sum(1 for e in entries if e["sex"] == "unknown")
    print(f"  With report:  {n_with_report}")
    print(f"  With label:   {n_with_label}")
    print(f"  Unknown age:  {n_unknown_age}")
    print(f"  Unknown sex:  {n_unknown_sex}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
