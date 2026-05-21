"""Check segmentation masks for multiple distinct lesions.

Counts 8-connected components with >= 50 pixels in each mask.
Reports cases with more than one component to stdout.

Usage:
    python src/check_multi_lesion.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT_DIR  = Path(__file__).resolve().parent.parent
MASKS_DIR = ROOT_DIR / "data" / "internal_dataset" / "segmentations"
EXCEL_PATH = ROOT_DIR / "data" / "internal_dataset" / "metadata.xlsx"

MIN_COMPONENT_PIXELS = 50
STRUCTURE_8 = np.ones((3, 3), dtype=int)


def _load_metadata(excel_path: Path) -> dict[str, tuple[str, str]]:
    """Return {file_stem: (entity, patid)} from Excel columns 0, 1, 7."""
    df = pd.read_excel(
        excel_path,
        sheet_name="internal_data_matched",
        usecols=[0, 1, 7],
        skiprows=1,
        header=0,
        dtype=str,
        engine="openpyxl",
    )
    df.columns = ["file_name", "entity", "anonym_patid"]
    df = df.dropna(subset=["file_name"])
    df["file_name"]    = df["file_name"].str.strip()
    df["entity"]       = df["entity"].fillna("").str.strip()
    df["anonym_patid"] = df["anonym_patid"].fillna("").str.strip()

    def _normalise_id(val: str) -> str:
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return val

    df["anonym_patid"] = df["anonym_patid"].apply(_normalise_id)

    return {
        Path(row["file_name"]).stem: (row["entity"], row["anonym_patid"])
        for _, row in df.iterrows()
    }


def count_lesions(mask_path: Path) -> int:
    """Return number of 8-connected components with >= MIN_COMPONENT_PIXELS pixels."""
    mask   = np.array(Image.open(mask_path).convert("L"))
    binary = mask > 0
    if not binary.any():
        return 0
    labeled, n = ndimage.label(binary, structure=STRUCTURE_8)
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    return int(sum(1 for s in sizes if s >= MIN_COMPONENT_PIXELS))


def main() -> None:
    meta        = _load_metadata(EXCEL_PATH)
    mask_paths  = sorted(MASKS_DIR.glob("*.png"))
    multi: list[tuple[str, str, str, int]] = []

    for mask_path in mask_paths:
        stem = mask_path.stem
        n    = count_lesions(mask_path)
        if n > 1:
            entity, patid = meta.get(stem, ("unknown", "unknown"))
            multi.append((stem, entity or "unknown", patid or "unknown", n))

    print(f"{'=' * 80}")
    print(f"Masks with multiple lesions: {len(multi)} / {len(mask_paths)}")
    print(f"{'=' * 80}")
    for stem, entity, patid, n in multi:
        print(f"  {stem}  |  {entity}  |  patid={patid}  |  {n} lesions")
    if not multi:
        print("  (none)")
    print()


if __name__ == "__main__":
    main()
