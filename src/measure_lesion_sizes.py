"""Offline measurement of native crop / padded-square sizes over the dataset split.

Reuses the exact geometry the training pipeline applies during preprocessing:
  - masked-crop arm : compute_crop_box(img, mask, context_fraction) -> square side
  - no-mask arm     : pad_to_square -> side = max(H, W)

Reports distributions (not just means) so the small-lesion tail is visible, and
translates them into the two decision-relevant quantities:
  - upsampling ceiling : native side vs. target resolution (past native = invented detail)
  - patches-on-lesion  : lesion bbox as a fraction of the fed image x grid_size

Run:  python src/measure_lesion_sizes.py [--split path] [--context_fraction 0.15]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from biomedclip.data.transforms import compute_crop_box

PATCH = 16
TARGETS = [224, 512, 768, 1024]


def _stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "p10": float(np.percentile(a, 10)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _fmt(name: str, s: dict) -> str:
    return (f"  {name:<22} n={s['n']:>4}  mean={s['mean']:7.1f}  "
            f"p10={s['p10']:7.1f}  median={s['median']:7.1f}  "
            f"p90={s['p90']:7.1f}  min={s['min']:6.0f}  max={s['max']:6.0f}")


def measure(samples: list[dict], context_fraction: float) -> dict:
    crop_sides, pad_sides, lesion_sides, lesion_frac = [], [], [], []
    n_no_mask = 0
    for s in samples:
        img_path = Path(s["image"])
        msk_path = Path(str(s.get("mask") or ""))
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            W, H = im.size                       # PIL size is (W, H)
        pad_side = max(H, W)
        pad_sides.append(pad_side)

        if not (msk_path.name and msk_path.exists()):
            n_no_mask += 1
            continue
        mask = np.array(Image.open(msk_path).convert("L"))
        if not np.any(mask > 0):
            n_no_mask += 1
            continue

        rows, cols = np.where(mask > 0)
        lesion_side = max(int(rows.max() - rows.min() + 1),
                          int(cols.max() - cols.min() + 1))
        lesion_sides.append(lesion_side)
        lesion_frac.append(lesion_side / pad_side)

        img_arr = np.array(im.convert("RGB")) if False else np.zeros((H, W), np.uint8)
        r0, r1, c0, c1 = compute_crop_box(img_arr, mask, context_fraction=context_fraction)
        crop_sides.append(max(r1 - r0, c1 - c0))

    return {
        "crop_sides": crop_sides,
        "pad_sides": pad_sides,
        "lesion_sides": lesion_sides,
        "lesion_frac": lesion_frac,
        "n_no_mask": n_no_mask,
    }


def report(title: str, m: dict) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    if m["crop_sides"]:
        print("Native MASKED-CROP side (px)  [crop_around_mask, cf given]:")
        print(_fmt("crop_side", _stats(m["crop_sides"])))
    if m["lesion_sides"]:
        print("Native LESION bbox side (px):")
        print(_fmt("lesion_side", _stats(m["lesion_sides"])))
        print("Lesion side / padded-image side (fraction):")
        print(_fmt("lesion_frac", _stats(m["lesion_frac"])))
    print("Native PADDED-SQUARE side (px)  [max(H,W), the no-mask arm input]:")
    print(_fmt("pad_side", _stats(m["pad_sides"])))
    print(f"  (samples with no usable mask: {m['n_no_mask']})")

    if m["crop_sides"]:
        med_crop = float(np.median(m["crop_sides"]))
        print("\n  MASKED-CROP arm — upsampling check (median crop = "
              f"{med_crop:.0f}px native):")
        for t in TARGETS:
            factor = t / med_crop
            tag = "UPSAMPLE (invented detail)" if factor > 1 else "downsample (detail kept)"
            print(f"    resize to {t:>4}px -> x{factor:4.2f}  {tag}")
    if m["lesion_frac"]:
        med_frac = float(np.median(m["lesion_frac"]))
        print(f"\n  NO-MASK arm — patches-on-lesion (median lesion = "
              f"{med_frac*100:.1f}% of image):")
        for t in TARGETS:
            grid = t // PATCH
            span = med_frac * grid
            print(f"    {t:>4}px -> {grid}x{grid} grid -> ~{span:4.1f} patches "
                  f"across lesion (~{span**2:5.1f} of {grid*grid} patches)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="/mnt/nfs/homedirs/philippw/Project/data/internal_dataset/split.json")
    ap.add_argument("--context_fraction", type=float, default=0.15)
    args = ap.parse_args()

    data = json.load(open(args.split))
    all_samples: list[dict] = []
    for name in ("train", "val", "test"):
        if name in data:
            m = measure(data[name], args.context_fraction)
            report(f"SPLIT: {name}  (context_fraction={args.context_fraction})", m)
            all_samples.extend(data[name])
    m_all = measure(all_samples, args.context_fraction)
    report(f"ALL SPLITS COMBINED  (context_fraction={args.context_fraction})", m_all)


if __name__ == "__main__":
    main()
