"""Prepare YOLO dataset structure from split_clahe.json.

Reads split_clahe.json and writes:
  data/yolo/
    images/{train,val,test}/   <- copies of CLAHE images
    labels/{train,val,test}/   <- one .txt per image (YOLO normalised bbox format)
    dataset.yaml

Usage:
    python src/yolo/data/prepare_yolo_dataset.py
    python src/yolo/data/prepare_yolo_dataset.py \
        --splits data/internal_dataset/split_clahe.json \
        --out_dir data/yolo
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/

import yaml
from PIL import Image

from yolo.data.bbox_from_mask import LABEL_TO_IDX, bboxes_from_mask

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SPLITS = ROOT_DIR / "data" / "internal_dataset" / "split_clahe.json"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "yolo"


def prepare(splits_path: Path, out_dir: Path) -> None:
    with open(splits_path, encoding="utf-8") as fh:
        splits: dict[str, list[dict]] = json.load(fh)

    for split_name, entries in splits.items():
        if split_name not in ("train", "val", "test"):
            continue

        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        written = skipped_no_label = skipped_no_box = 0
        no_box_masks: list[str] = []

        for entry in entries:
            image_path = Path(entry["image"])
            mask_path = Path(entry["mask"])
            label_str = (entry.get("label") or "").strip().lower()

            if label_str not in LABEL_TO_IDX:
                skipped_no_label += 1
                continue

            class_id = LABEL_TO_IDX[label_str]

            with Image.open(image_path) as img:
                w, h = img.size

            boxes = bboxes_from_mask(mask_path, class_id, w, h)
            if not boxes:
                skipped_no_box += 1
                no_box_masks.append(str(mask_path))
                continue

            dst_img = img_dir / image_path.name
            if not dst_img.exists():
                shutil.copy2(image_path, dst_img)

            dst_lbl = lbl_dir / (image_path.stem + ".txt")
            with open(dst_lbl, "w") as fh:
                for cls, xc, yc, wn, hn in boxes:
                    fh.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

            written += 1

        print(
            f"[{split_name}] {written} written, "
            f"{skipped_no_label} no-label, {skipped_no_box} no-box"
        )
        for p in no_box_masks:
            print(f"  no-box: {p}")

    yaml_path = out_dir / "dataset.yaml"
    yaml_data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 3,
        "names": ["benign", "intermediate", "malignant"],
    }
    with open(yaml_path, "w") as fh:
        yaml.dump(yaml_data, fh, default_flow_style=False, sort_keys=False)
    print(f"Saved dataset.yaml -> {yaml_path}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from split_clahe.json.")
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    prepare(Path(args.splits), Path(args.out_dir))


if __name__ == "__main__":
    main(parse_args())
