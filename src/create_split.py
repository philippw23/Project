"""Standalone script to create and save the train/val/test split manifest.

Runs `build_stratified_splits` from the BiomedCLIP data pipeline and writes
`split.json` to the specified output directory — no model loading required.

Usage:
    python src/create_split.py [args]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from biomedclip.utils.misc import (
    DEFAULT_EXCEL,
    DEFAULT_REPORTS,
    DEFAULT_IMAGES_DIR,
    DEFAULT_MASKS_DIR,
    DEFAULT_SPLIT_DIR,
)
from biomedclip.data.splits import build_stratified_splits


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and save the stratified train/val/test split manifest."
    )
    parser.add_argument("--excel",   default=str(DEFAULT_EXCEL),
                        help="Path to metadata.xlsx (default: %(default)s)")
    parser.add_argument("--reports", default=str(DEFAULT_REPORTS),
                        help="Path to reports JSON (default: %(default)s)")
    parser.add_argument("--english", action="store_true",
                        help="Use English translations (befund_en / beurteilung_en)")
    parser.add_argument("--images",  default=str(DEFAULT_IMAGES_DIR),
                        help="Directory containing image PNGs (default: %(default)s)")
    parser.add_argument("--masks",   default=str(DEFAULT_MASKS_DIR),
                        help="Directory containing segmentation mask PNGs (default: %(default)s)")
    parser.add_argument("--out_dir", default=str(DEFAULT_SPLIT_DIR),
                        help="Output directory for split.json (default: %(default)s)")
    parser.add_argument("--downstream_train_frac", type=float, default=0.8,
                        help="Fraction of downstream data for classifier training (default: %(default)s)")
    parser.add_argument("--downstream_val_frac",   type=float, default=0.1,
                        help="Fraction of downstream data for classifier validation (default: %(default)s)")
    parser.add_argument("--test_frac",             type=float, default=0.1,
                        help="Fraction of downstream data held out for final evaluation (default: %(default)s)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed (default: %(default)s)")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    build_stratified_splits(args, run_dir=Path(args.out_dir))


if __name__ == "__main__":
    main(parse_args())
