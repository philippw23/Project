"""Display 10 random image/segmentation mask pairs: 5 per row, original above masked.

Masks are built from polygon and rectangle annotations in the BTXRD JSON files.
"""
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

IMAGES_DIR = Path(__file__).parent / "data" / "BTXRD" / "images"
ANNOTATIONS_DIR = Path(__file__).parent / "data" / "BTXRD" / "Annotations"

# Find images that have a matching annotation file
paired = [
    img for img in IMAGES_DIR.glob("*.jpeg")
    if (ANNOTATIONS_DIR / img.with_suffix(".json").name).exists()
]

if not paired:
    raise FileNotFoundError(
        "No matching image/annotation pairs found. "
        "Make sure filenames are identical in both directories."
    )

# Randomly sample 10 pairs
samples = random.sample(paired, min(10, len(paired)))


def build_mask(annotation_path: Path, width: int, height: int) -> np.ndarray:
    """Return a binary mask (0/255) built from all polygon/rectangle shapes."""
    with annotation_path.open() as f:
        data = json.load(f)

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        pts = shape["points"]
        flat = [(p[0], p[1]) for p in pts]
        if len(flat) >= 3:
            draw.polygon(flat, fill=255)

    return np.array(mask_img, dtype=float)


# Layout: 4 rows x 5 cols — even rows: originals, odd rows: masked regions
fig, axes = plt.subplots(4, 5, figsize=(18, 14))

for i, img_path in enumerate(samples):
    ann_path = ANNOTATIONS_DIR / img_path.with_suffix(".json").name

    image = Image.open(img_path).convert("L")
    w, h = image.size
    image_arr = np.array(image, dtype=float)

    mask = build_mask(ann_path, w, h)

    # Apply mask: keep only the annotated region
    masked = image_arr * (mask / 255.0)

    group = i // 5   # 0 = first 5 pairs, 1 = second 5 pairs
    col = i % 5

    axes[group * 2][col].imshow(image_arr, cmap="gray")
    axes[group * 2][col].set_title(img_path.stem, fontsize=7)
    axes[group * 2][col].axis("off")

    axes[group * 2 + 1][col].imshow(masked, cmap="gray")
    axes[group * 2 + 1][col].axis("off")

axes[0][0].set_ylabel("Original", fontsize=11)
axes[1][0].set_ylabel("Masked", fontsize=11)
axes[2][0].set_ylabel("Original", fontsize=11)
axes[3][0].set_ylabel("Masked", fontsize=11)

plt.tight_layout()
plt.show()
