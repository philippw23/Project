"""Display 10 random image/segmentation mask pairs: 5 per row, original above masked."""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGES_DIR = Path(__file__).parent.parent / "data" / "images"
MASKS_DIR = Path(__file__).parent.parent / "data" / "segmentations"

# Find images that have a matching segmentation mask
paired = [
    img for img in IMAGES_DIR.glob("*.png")
    if (MASKS_DIR / img.name).exists()
]

if not paired:
    raise FileNotFoundError(
        "No matching image/segmentation pairs found. "
        "Make sure filenames are identical in both directories."
    )

# Randomly sample 10 pairs
samples = random.sample(paired, min(10, len(paired)))

# Layout: 4 rows x 5 cols — even rows: originals, odd rows: masked regions
fig, axes = plt.subplots(4, 5, figsize=(18, 14))

for i, img_path in enumerate(samples):
    mask_path = MASKS_DIR / img_path.name

    image = np.array(Image.open(img_path).convert("L"), dtype=float)
    mask = np.array(Image.open(mask_path).convert("L"), dtype=float)

    # Apply mask: keep only the lesion region
    masked = image * (mask / 255.0)

    group = i // 5   # 0 = first 5 pairs, 1 = second 5 pairs
    col = i % 5

    axes[group * 2][col].imshow(image, cmap="gray")
    axes[group * 2][col].axis("off")

    axes[group * 2 + 1][col].imshow(masked, cmap="gray")
    axes[group * 2 + 1][col].axis("off")

axes[0][0].set_ylabel("Original", fontsize=11)
axes[1][0].set_ylabel("Masked", fontsize=11)
axes[2][0].set_ylabel("Original", fontsize=11)
axes[3][0].set_ylabel("Masked", fontsize=11)

plt.tight_layout()
plt.show()
