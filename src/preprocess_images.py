"""Offline image preprocessing: square-pad all images (and masks) to preprocessed_images/.

Each image is padded (black border, centered) so the shorter side matches the
longer side, producing a square PNG.  The corresponding segmentation mask is
padded with the same dimensions so crop_around_mask stays aligned.

Usage:
    python src/preprocess_images.py                   # both datasets
    python src/preprocess_images.py --dataset internal
    python src/preprocess_images.py --dataset btxrd
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT_DIR = Path(__file__).resolve().parent.parent

INTERNAL_IMAGES = ROOT_DIR / "data" / "internal_dataset" / "images"
INTERNAL_MASKS  = ROOT_DIR / "data" / "internal_dataset" / "segmentations"
INTERNAL_OUT    = ROOT_DIR / "data" / "internal_dataset" / "preprocessed_images"
INTERNAL_MASKS_OUT = ROOT_DIR / "data" / "internal_dataset" / "preprocessed_segmentations"

BTXRD_IMAGES = ROOT_DIR / "data" / "BTXRD" / "images"
BTXRD_OUT    = ROOT_DIR / "data" / "BTXRD" / "preprocessed_images"


def _square_pad(image: Image.Image, fill: int = 0) -> tuple[Image.Image, int, int]:
    """Pad image to square, centered. Returns (padded_image, pad_left, pad_top)."""
    w, h = image.size
    side = max(w, h)
    pad_left = (side - w) // 2
    pad_top  = (side - h) // 2
    result = Image.new(image.mode, (side, side), fill)
    result.paste(image, (pad_left, pad_top))
    return result, pad_left, pad_top


def _process_internal(images_dir: Path, masks_dir: Path, out_dir: Path, masks_out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(images_dir.glob("*.png"))
    if not paths:
        print(f"No images found in {images_dir}")
        return

    truncated = []
    for p in tqdm(paths, desc="internal/images"):
        try:
            img = Image.open(p).convert("RGB")
            padded, pad_left, pad_top = _square_pad(img)
            padded.save(out_dir / (p.stem + ".png"), format="PNG")
        except OSError:
            truncated.append(p.name)
            continue

        mask_path = masks_dir / p.name
        if mask_path.exists():
            try:
                mask = Image.open(mask_path).convert("L")
                padded_mask, _, _ = _square_pad(mask, fill=0)
                padded_mask.save(masks_out_dir / (p.stem + ".png"), format="PNG")
            except OSError:
                pass  # mask unreadable — image is saved, mask simply absent

    print(f"  {len(paths) - len(truncated)} images  → {out_dir}")
    n_masks = len(list(masks_out_dir.glob("*.png")))
    print(f"  {n_masks} masks → {masks_out_dir}")
    if truncated:
        print(f"  {len(truncated)} truncated images skipped:")
        for name in truncated:
            print(f"    {name}")


def _process_btxrd(images_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(images_dir.glob("*.jpeg"))
    if not paths:
        print(f"No images found in {images_dir}")
        return

    truncated = []
    for p in tqdm(paths, desc="BTXRD/images"):
        try:
            img = Image.open(p).convert("RGB")
            padded, _, _ = _square_pad(img)
            padded.save(out_dir / (p.stem + ".png"), format="PNG")
        except OSError:
            truncated.append(p.name)

    print(f"  {len(paths) - len(truncated)} images → {out_dir}")
    if truncated:
        print(f"  {len(truncated)} truncated images skipped:")
        for name in truncated:
            print(f"    {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=["internal", "btxrd", "both"],
        default="both",
        help="Which dataset to preprocess (default: both)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset in ("internal", "both"):
        print("Processing internal dataset …")
        _process_internal(INTERNAL_IMAGES, INTERNAL_MASKS, INTERNAL_OUT, INTERNAL_MASKS_OUT)

    if args.dataset in ("btxrd", "both"):
        print("Processing BTXRD dataset …")
        _process_btxrd(BTXRD_IMAGES, BTXRD_OUT)


if __name__ == "__main__":
    main()
