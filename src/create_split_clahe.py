"""Create split_clahe.json from split.json by remapping image paths to images_clahe/.

Usage:
    python src/create_split_clahe.py
    python src/create_split_clahe.py --split data/internal_dataset/split.json \
        --out data/internal_dataset/split_clahe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT = ROOT_DIR / "data" / "internal_dataset" / "split.json"
DEFAULT_OUT   = ROOT_DIR / "data" / "internal_dataset" / "split_clahe.json"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap split.json image paths to images_clahe/.")
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out",   default=str(DEFAULT_OUT))
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    with open(args.split, encoding="utf-8") as fh:
        split = json.load(fh)

    for entries in split.values():
        for entry in entries:
            if "image" in entry:
                entry["image"] = entry["image"].replace("/images/", "/images_clahe/")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main(parse_args())
