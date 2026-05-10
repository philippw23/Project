"""Visualize random image/mask pairs for the default dataset or BTXRD."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biomedclip.data.transforms import crop_around_mask

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_IMAGES_DIR = ROOT_DIR / "data" / "internal_dataset" / "images"
DEFAULT_MASKS_DIR  = ROOT_DIR / "data" / "internal_dataset" / "segmentations"
BTXRD_IMAGES_DIR   = ROOT_DIR / "data" / "BTXRD" / "preprocessed_images"
BTXRD_ANNOTATIONS_DIR = ROOT_DIR / "data" / "BTXRD" / "Annotations"


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
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Image stem(s) or filenames to display instead of random sampling.",
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
        for img in BTXRD_IMAGES_DIR.glob("*.png")
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
        with annotation_path.open() as fh:
            annot = json.load(fh)
        orig_h, orig_w = annot["imageHeight"], annot["imageWidth"]
        mask = build_btxrd_mask(annotation_path, orig_w, orig_h)
        # pad mask to match the square-padded preprocessed image
        side     = max(orig_h, orig_w)
        pad_top  = (side - orig_h) // 2
        pad_left = (side - orig_w) // 2
        padded   = np.zeros((side, side), dtype=mask.dtype)
        padded[pad_top:pad_top + orig_h, pad_left:pad_left + orig_w] = mask
        mask = padded

    return image_arr, mask


def make_green_overlay(image_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB float [0,1] image with a diffused green overlay on the tumor."""
    binary_mask = (mask > 0).astype(float)

    rows, cols = np.where(binary_mask > 0)
    if len(rows) == 0:
        gray = image_arr / image_arr.max() if image_arr.max() > 0 else image_arr
        return np.stack([gray, gray, gray], axis=-1)

    bbox_h = rows.max() - rows.min() + 1
    bbox_w = cols.max() - cols.min() + 1
    bbox_diag = math.hypot(bbox_h, bbox_w)
    sigma = max(5.0, bbox_diag * 0.04)

    alpha = gaussian_filter(binary_mask, sigma=sigma)
    if alpha.max() > 0:
        alpha = alpha / alpha.max()
    alpha = alpha * 0.6  # max opacity

    gray = image_arr / image_arr.max() if image_arr.max() > 0 else image_arr.copy()

    r = gray * (1.0 - alpha)
    g = gray * (1.0 - alpha) + alpha
    b = gray * (1.0 - alpha)

    return np.stack([r, g, b], axis=-1)


def main(argv=None):
    args = parse_args(argv)
    dataset = "btxrd" if args.btxrd else "default"
    rng = random.Random(args.seed)

    paired = find_paired_images(dataset)
    if not paired:
        raise FileNotFoundError(
            f"No matching image/annotation pairs found for dataset '{dataset}'."
        )

    if args.images is not None:
        requested = {Path(name).stem for name in args.images}
        samples = [p for p in paired if p.stem in requested]
        if not samples:
            available = ", ".join(p.stem for p in paired[:10])
            raise FileNotFoundError(
                f"None of the requested images {sorted(requested)} were found. "
                f"Available stems (first 10): {available}"
            )
    else:
        sample_count = min(args.samples, len(paired))
        samples = rng.sample(paired, sample_count)

    sample_count = len(samples)
    cols = 5
    groups = math.ceil(sample_count / cols)
    fig, axes = plt.subplots(groups * 3, cols, figsize=(18, max(9, groups * 10)))
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis("off")

    for index, image_path in enumerate(samples):
        image_arr, mask = load_sample(image_path, dataset)
        overlay = make_green_overlay(image_arr, mask)
        cropped = crop_around_mask(image_arr, mask)

        group = index // cols
        col = index % cols
        original_row = group * 3
        overlay_row = original_row + 1
        crop_row = original_row + 2

        axes[original_row][col].imshow(image_arr, cmap="gray")
        axes[original_row][col].set_title(image_path.stem, fontsize=7)

        axes[overlay_row][col].imshow(overlay)

        axes[crop_row][col].imshow(cropped, cmap="gray")

    for group in range(groups):
        base_row = group * 3
        axes[base_row][0].set_ylabel("Original", fontsize=11)
        axes[base_row + 1][0].set_ylabel("Tumor Highlight", fontsize=11)
        axes[base_row + 2][0].set_ylabel("Tumor Crop", fontsize=11)

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"visualize_samples_{dataset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
