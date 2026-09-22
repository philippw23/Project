"""Build a BTXRD downstream manifest to use BTXRD as an external test set.

BTXRD does not ship in the internal-split sample format, so this one-off script
converts it:

  * labels / age / sex  ← data/BTXRD/dataset.xlsx
  * image               ← data/BTXRD/preprocessed_images/<id>.png
  * mask                ← rasterized from data/BTXRD/Annotations/<id>.json,
                          square-padded (centered) to match the preprocessed
                          image, saved as data/BTXRD/downstream_masks/<id>.png

Only tumor rows with exactly one of {benign, malignant} and valid age/sex are
kept (matches the internal --binary setup). The output is a flat JSON list of
sample dicts:

    {"image": <abs png>, "mask": <abs png or "">, "label": "benign"|"malignant",
     "age": float, "sex": 0.0|1.0, "patid": <image stem>}

which `DownstreamDataset` / the k-fold CV orchestrator consume directly via
`--btxrd_manifest`.

Usage:
    python src/data/build_btxrd_downstream.py
    python src/data/build_btxrd_downstream.py --btxrd_dir data/BTXRD --out data/BTXRD/btxrd_downstream_binary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from LACE.data.transforms import rasterize_shapes

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _to_int(val) -> int | None:
    try:
        return int(round(float(val)))
    except (ValueError, TypeError):
        return None


def _rasterize_and_save(annot_path: Path, out_mask_path: Path) -> bool:
    """Rasterize a LabelImg annotation JSON to a centered square-padded PNG mask
    that lines up with the preprocessed (square-padded) BTXRD image.

    Returns True if a non-empty mask was written.
    """
    with open(annot_path, encoding="utf-8") as fh:
        annot = json.load(fh)

    img_h = int(annot["imageHeight"])
    img_w = int(annot["imageWidth"])
    mask_arr = rasterize_shapes(annot["shapes"], img_h, img_w)  # (h, w) uint8, 255 inside

    side     = max(img_h, img_w)
    pad_top  = (side - img_h) // 2
    pad_left = (side - img_w) // 2
    padded   = np.zeros((side, side), dtype=mask_arr.dtype)
    padded[pad_top:pad_top + img_h, pad_left:pad_left + img_w] = mask_arr

    if not np.any(padded > 0):
        return False
    Image.fromarray(padded, mode="L").save(out_mask_path, format="PNG")
    return True


def build(args: argparse.Namespace) -> None:
    btxrd_dir = Path(args.btxrd_dir)
    xlsx_path = Path(args.xlsx) if args.xlsx else btxrd_dir / "dataset.xlsx"
    images_dir = Path(args.images_dir) if args.images_dir else btxrd_dir / "preprocessed_images"
    annot_dir  = Path(args.annot_dir)  if args.annot_dir  else btxrd_dir / "Annotations"
    masks_out  = Path(args.masks_out)  if args.masks_out  else btxrd_dir / "downstream_masks"
    out_path   = Path(args.out)        if args.out        else btxrd_dir / "btxrd_downstream_binary.json"
    masks_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(xlsx_path)
    required = {"image_id", "age", "gender", "tumor", "benign", "malignant"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"dataset.xlsx is missing expected columns: {sorted(missing)}")

    samples: list[dict] = []
    stats = {
        "no_tumor": 0, "ambiguous_label": 0, "bad_age_sex": 0,
        "missing_image": 0, "with_mask": 0, "no_mask": 0,
    }
    labels_count = {"benign": 0, "malignant": 0}

    for _, row in df.iterrows():
        if _to_int(row["tumor"]) != 1:
            stats["no_tumor"] += 1
            continue

        is_benign    = _to_int(row["benign"])    == 1
        is_malignant = _to_int(row["malignant"]) == 1
        if is_benign == is_malignant:          # neither, or both → ambiguous
            stats["ambiguous_label"] += 1
            continue
        label = "benign" if is_benign else "malignant"

        try:
            age = float(row["age"])
        except (ValueError, TypeError):
            stats["bad_age_sex"] += 1
            continue
        if not np.isfinite(age):
            stats["bad_age_sex"] += 1
            continue

        gender = str(row["gender"]).strip().lower()
        if gender in ("m", "male"):
            sex = 1.0
        elif gender in ("f", "female"):
            sex = 0.0
        else:
            stats["bad_age_sex"] += 1
            continue

        stem = Path(str(row["image_id"])).stem
        img_path = images_dir / f"{stem}.png"
        if not img_path.exists():
            stats["missing_image"] += 1
            continue

        annot_path = annot_dir / f"{stem}.json"
        mask_field = ""
        if annot_path.exists():
            out_mask = masks_out / f"{stem}.png"
            if _rasterize_and_save(annot_path, out_mask):
                mask_field = str(out_mask.resolve())
                stats["with_mask"] += 1
            else:
                stats["no_mask"] += 1
        else:
            stats["no_mask"] += 1

        samples.append({
            "image": str(img_path.resolve()),
            "mask":  mask_field,
            "label": label,
            "age":   age,
            "sex":   sex,
            "patid": stem,
        })
        labels_count[label] += 1

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(samples, fh, indent=2)

    print(f"BTXRD downstream manifest: {len(samples)} samples → {out_path}")
    print(f"  labels: {labels_count}")
    print(f"  masks:  {stats['with_mask']} with mask, {stats['no_mask']} without (full-image fallback)")
    print(
        f"  dropped: {stats['no_tumor']} non-tumor, {stats['ambiguous_label']} ambiguous label, "
        f"{stats['bad_age_sex']} bad age/sex, {stats['missing_image']} missing image"
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the BTXRD external-test downstream manifest.")
    p.add_argument("--btxrd_dir",  default=str(ROOT_DIR / "data" / "BTXRD"))
    p.add_argument("--xlsx",       default=None, help="Defaults to <btxrd_dir>/dataset.xlsx")
    p.add_argument("--images_dir", default=None, help="Defaults to <btxrd_dir>/preprocessed_images")
    p.add_argument("--annot_dir",  default=None, help="Defaults to <btxrd_dir>/Annotations")
    p.add_argument("--masks_out",  default=None, help="Defaults to <btxrd_dir>/downstream_masks")
    p.add_argument("--out",        default=None, help="Defaults to <btxrd_dir>/btxrd_downstream_binary.json")
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
