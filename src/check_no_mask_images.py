"""Check malignancy distribution of images in a given directory.

Cross-references image filenames against dataset_full.json by stem.

Usage:
    python src/check_no_mask_images.py --dir data/internal_dataset/no_mask_images
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT_DIR / "data" / "internal_dataset" / "dataset_full.json"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show malignancy distribution of images in a directory."
    )
    parser.add_argument("--dir", required=True, help="Directory containing PNG images")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                        help="Path to dataset_full.json (default: %(default)s)")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    img_dir = Path(args.dir)
    stems = {p.stem for p in img_dir.glob("*.png")}

    if not stems:
        print(f"No PNG images found in {img_dir}")
        return

    with open(args.dataset, encoding="utf-8") as fh:
        dataset = json.load(fh)

    stem_to_label = {Path(e["image"]).stem: e.get("label") for e in dataset}

    labels: list[str] = []
    not_found: list[str] = []
    for stem in sorted(stems):
        if stem in stem_to_label:
            labels.append(stem_to_label[stem] or "no_label")
        else:
            not_found.append(stem)

    print(f"Images in {img_dir}: {len(stems)}")
    print(f"Matched in dataset: {len(labels)}, not found: {len(not_found)}")
    print()
    for label, count in sorted(Counter(labels).items()):
        print(f"  {label}: {count}")

    if not_found:
        print("\nNot found in dataset:")
        for stem in not_found:
            print(f"  {stem}")


if __name__ == "__main__":
    main(parse_args())
