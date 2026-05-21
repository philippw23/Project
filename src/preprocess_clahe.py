"""Offline CLAHE preprocessing for internal bone-tumor radiographs.

Reads images from data/internal_dataset/images/ by default, applies grayscale
CLAHE, and writes RGB-compatible PNGs with matching filenames to
data/internal_dataset/images_clahe/.

Usage:
    python src/preprocess_clahe.py
    python src/preprocess_clahe.py --overwrite
    python src/preprocess_clahe.py --input_dir data/internal_dataset/images \
        --output_dir data/internal_dataset/images_clahe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "internal_dataset" / "images"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "internal_dataset" / "images_clahe"


def _load_grayscale(path: Path) -> np.ndarray:
    """Load an image as uint8 grayscale for CLAHE."""
    with Image.open(path) as img:
        return np.array(img.convert("L"), dtype=np.uint8)


def _save_rgb_png(gray: np.ndarray, path: Path) -> None:
    """Save a grayscale array as RGB PNG for compatibility with RGB loaders."""
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")


def apply_clahe(
    input_dir: Path,
    output_dir: Path,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
    overwrite: bool,
) -> tuple[int, int, list[tuple[Path, str]]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.png"))
    if not image_paths:
        print(f"No PNG images found in {input_dir}")
        return 0, 0, []

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    processed = 0
    skipped = 0
    failed: list[tuple[Path, str]] = []

    for src in tqdm(image_paths, desc="CLAHE/internal"):
        dst = output_dir / f"{src.stem}.png"
        if dst.exists() and not overwrite:
            skipped += 1
            continue

        try:
            gray = _load_grayscale(src)
            enhanced = clahe.apply(gray)
            _save_rgb_png(enhanced, dst)
            processed += 1
        except (OSError, ValueError, cv2.error) as exc:
            failed.append((src, str(exc)))

    return processed, skipped, failed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply CLAHE preprocessing to internal dataset radiographs."
    )
    parser.add_argument(
        "--input_dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing source PNG images (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CLAHE-enhanced PNG images (default: %(default)s)",
    )
    parser.add_argument(
        "--clip_limit",
        type=float,
        default=4.0,
        help="CLAHE clip limit (default: %(default)s)",
    )
    parser.add_argument(
        "--tile_grid_width",
        type=int,
        default=10,
        help="CLAHE tile grid width (default: %(default)s)",
    )
    parser.add_argument(
        "--tile_grid_height",
        type=int,
        default=5,
        help="CLAHE tile grid height (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images instead of skipping them.",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tile_grid_size = (args.tile_grid_width, args.tile_grid_height)

    print("Applying CLAHE preprocessing")
    print(f"  input_dir      : {input_dir}")
    print(f"  output_dir     : {output_dir}")
    print(f"  clip_limit     : {args.clip_limit}")
    print(f"  tile_grid_size : {tile_grid_size}")
    print(f"  overwrite      : {args.overwrite}")

    processed, skipped, failed = apply_clahe(
        input_dir=input_dir,
        output_dir=output_dir,
        clip_limit=args.clip_limit,
        tile_grid_size=tile_grid_size,
        overwrite=args.overwrite,
    )

    print("\nSummary")
    print(f"  processed : {processed}")
    print(f"  skipped   : {skipped}")
    print(f"  failed    : {len(failed)}")
    if failed:
        print("Failed images:")
        for path, reason in failed:
            print(f"  {path.name}: {reason}")


if __name__ == "__main__":
    main(parse_args())
