"""Standalone script to create and save the train/val/test split manifest.

Runs `build_stratified_splits` from the BiomedCLIP data pipeline and writes
`split.json` to the specified output directory — no model loading required.

Usage:
    python src/data/create_split.py [args]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biomedclip.utils.misc import DEFAULT_DATASET_JSON, DEFAULT_SPLIT_DIR
from biomedclip.data.splits import build_stratified_splits


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and save the stratified train/val/test split manifest."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_JSON),
                        help="Path to dataset_full.json (default: %(default)s)")
    parser.add_argument("--out_dir", default=str(DEFAULT_SPLIT_DIR),
                        help="Output directory for split.json (default: %(default)s)")
    parser.add_argument("--downstream_train_frac", type=float, default=0.8,
                        help="Fraction of data for training (default: %(default)s)")
    parser.add_argument("--downstream_val_frac",   type=float, default=0.1,
                        help="Fraction of data for validation (default: %(default)s)")
    parser.add_argument("--test_frac",             type=float, default=0.1,
                        help="Fraction of data held out for final evaluation (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--binary", action="store_true",
                        help="Exclude intermediate cases; keep only benign and malignant labels. "
                             "Saves to split_binary.json instead of split.json.")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    build_stratified_splits(args, run_dir=Path(args.out_dir))


if __name__ == "__main__":
    main(parse_args())
