"""Preprocess images by cropping around segmentation masks.

For each image that has a corresponding mask, crops around the lesion using
the same logic as biomedclip_pretrain.py. Images without a mask are copied
unchanged. Saves results to data/cropped_images/.

Usage:
    python src/preprocess_crop_images.py
    python src/preprocess_crop_images.py --images data/images --masks data/segmentations --out data/cropped_images
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
from biomedclip.data.transforms import crop_around_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", default=str(ROOT_DIR / "data" / "images"))
    p.add_argument("--masks",  default=str(ROOT_DIR / "data" / "segmentations"))
    p.add_argument("--out",    default=str(ROOT_DIR / "data" / "cropped_images"))
    p.add_argument("--no_mask", default=str(ROOT_DIR / "data" / "no_mask_images"))
    p.add_argument("--context_fraction", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images)
    masks_dir  = Path(args.masks)
    out_dir      = Path(args.out)
    no_mask_dir = Path(args.no_mask)
    out_dir.mkdir(parents=True, exist_ok=True)
    no_mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        print(f"No PNG images found in {images_dir}")
        return

    truncated = []
    no_mask = []
    for image_path in tqdm(image_paths, desc="Cropping"):
        mask_path = masks_dir / image_path.name

        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            truncated.append(image_path.name)
            continue
        image_arr = np.array(image.convert("L"), dtype=float)

        if mask_path.exists():
            mask_arr = np.array(Image.open(mask_path).convert("L"), dtype=float)
            cropped_arr = crop_around_mask(image_arr, mask_arr, context_fraction=args.context_fraction, pad_mode="mean")
            result = Image.fromarray(cropped_arr.astype(np.uint8)).convert("RGB")
            result.save(out_dir / image_path.name)
        else:
            no_mask.append(image_path.name)
            image.save(no_mask_dir / image_path.name)

    if truncated:
        print(f"{len(truncated)} truncated images skipped:")
        for name in truncated:
            print(f"  {name}")
    print(f"\nDone. {len(image_paths) - len(no_mask) - len(truncated)} cropped images -> {out_dir}")
    if no_mask:
        print(f"{len(no_mask)} images had no mask -> {no_mask_dir}:")
        for name in no_mask:
            print(f"  {name}")


if __name__ == "__main__":
    main()
