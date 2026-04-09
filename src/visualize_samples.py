"""Visualize random image/mask pairs for the default dataset or BTXRD."""
import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_IMAGES_DIR = ROOT_DIR / "data" / "images"
DEFAULT_MASKS_DIR = ROOT_DIR / "data" / "segmentations"
BTXRD_IMAGES_DIR = ROOT_DIR / "data" / "BTXRD" / "images"
BTXRD_ANNOTATIONS_DIR = ROOT_DIR / "data" / "BTXRD" / "Annotations"
SMALL_MASK_THRESHOLD = 0.10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Display random image/segmentation samples."
    )
    parser.add_argument(
        "--btxrd",
        action="store_true",
        help="Use the BTXRD dataset instead of the default dataset.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Maximum number of pairs to display.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    return parser.parse_args(argv)


def find_paired_images(dataset: str) -> list[Path]:
    if dataset == "default":
        return sorted(
            img
            for img in DEFAULT_IMAGES_DIR.glob("*.png")
            if (DEFAULT_MASKS_DIR / img.name).exists()
        )

    return sorted(
        img
        for img in BTXRD_IMAGES_DIR.glob("*.jpeg")
        if (BTXRD_ANNOTATIONS_DIR / img.with_suffix(".json").name).exists()
    )


def build_btxrd_mask(annotation_path: Path, width: int, height: int) -> np.ndarray:
    """Return a binary mask (0/255) from polygon annotations."""
    with annotation_path.open() as handle:
        data = json.load(handle)

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        points = [(point[0], point[1]) for point in shape.get("points", [])]
        if len(points) >= 3:
            draw.polygon(points, fill=255)

    return np.array(mask_img, dtype=float)


def load_sample(image_path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("L")
    image_arr = np.array(image, dtype=float)

    if dataset == "default":
        mask_path = DEFAULT_MASKS_DIR / image_path.name
        mask = np.array(Image.open(mask_path).convert("L"), dtype=float)
    else:
        annotation_path = BTXRD_ANNOTATIONS_DIR / image_path.with_suffix(".json").name
        mask = build_btxrd_mask(annotation_path, *image.size)

    return image_arr, mask


def crop_around_mask(image_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop around the tumor.

    If the mask covers less than SMALL_MASK_THRESHOLD of the image, the crop
    region is padded by 50% of the bounding-box size on each side so that small
    tumors are shown with enough surrounding context.
    """
    binary_mask = mask > 0
    if not np.any(binary_mask):
        return image_arr

    rows, cols = np.where(binary_mask)
    top, bottom = rows.min(), rows.max()
    left, right = cols.min(), cols.max()

    box_height = bottom - top + 1
    box_width = right - left + 1

    mask_fraction = float(binary_mask.mean())
    if mask_fraction < SMALL_MASK_THRESHOLD:
        pad_y = math.ceil(box_height * 0.70)
        pad_x = math.ceil(box_width * 0.70)
    else:
        pad_y = math.ceil(box_height * 0.40)
        pad_x = math.ceil(box_width * 0.40)

    crop_top = max(0, top - pad_y)
    crop_bottom = min(image_arr.shape[0], bottom + pad_y + 1)
    crop_left = max(0, left - pad_x)
    crop_right = min(image_arr.shape[1], right + pad_x + 1)

    # Make the crop square by extending the shorter side from its center,
    # with a minimum side length of 20% of the original image size.
    crop_h = crop_bottom - crop_top
    crop_w = crop_right - crop_left
    min_side = math.ceil(min(image_arr.shape[0], image_arr.shape[1]) * 0.20)
    side = max(crop_h, crop_w, min_side)
    cy = (crop_top + crop_bottom) // 2
    cx = (crop_left + crop_right) // 2
    half = side // 2
    crop_top = max(0, cy - half)
    crop_bottom = min(image_arr.shape[0], crop_top + side)
    crop_left = max(0, cx - half)
    crop_right = min(image_arr.shape[1], crop_left + side)

    return image_arr[crop_top:crop_bottom, crop_left:crop_right]


def main(argv=None):
    args = parse_args(argv)
    dataset = "btxrd" if args.btxrd else "default"
    rng = random.Random(args.seed)

    paired = find_paired_images(dataset)
    if not paired:
        raise FileNotFoundError(
            f"No matching image/annotation pairs found for dataset '{dataset}'."
        )

    sample_count = min(args.samples, len(paired))
    samples = rng.sample(paired, sample_count)

    cols = 5
    groups = math.ceil(sample_count / cols)
    fig, axes = plt.subplots(groups * 3, cols, figsize=(18, max(9, groups * 10)))
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis("off")

    for index, image_path in enumerate(samples):
        image_arr, mask = load_sample(image_path, dataset)
        masked = image_arr * (mask / 255.0)
        cropped = crop_around_mask(image_arr, mask)

        group = index // cols
        col = index % cols
        original_row = group * 3
        masked_row = original_row + 1
        crop_row = original_row + 2

        axes[original_row][col].imshow(image_arr, cmap="gray")
        axes[original_row][col].set_title(image_path.stem, fontsize=7)

        axes[masked_row][col].imshow(masked, cmap="gray")

        axes[crop_row][col].imshow(cropped, cmap="gray")

    for group in range(groups):
        base_row = group * 3
        axes[base_row][0].set_ylabel("Original", fontsize=11)
        axes[base_row + 1][0].set_ylabel("Masked", fontsize=11)
        axes[base_row + 2][0].set_ylabel("Tumor Crop", fontsize=11)

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"visualize_samples_{dataset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
