from __future__ import annotations

import numpy as np
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import torch

from biomedclip.data.transforms import build_train_transform, _compute_crop_1d

PATCH_SIZE = 16
GRID_SIZE  = 14   # 224 / 16
N_PATCHES  = GRID_SIZE * GRID_SIZE   # 196
RESAMPLE_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def extract_centered_crop(
    image: Image.Image,
    mask_arr: np.ndarray,
    crop_size: int = 224,
) -> Image.Image:
    """Return a square PIL crop of crop_size centered on the mask centroid.

    Falls back to the full image resized to crop_size when the mask is empty.
    Reuses _compute_crop_1d from biomedclip for boundary clamping logic.
    """
    binary = mask_arr > 0
    if not np.any(binary):
        return image.resize((crop_size, crop_size), RESAMPLE_BICUBIC)

    rows, cols = np.where(binary)
    cy = float(rows.mean())
    cx = float(cols.mean())

    img_arr = np.array(image.convert("RGB"))
    H, W = img_arr.shape[:2]

    r_start, r_end, pad_top,  pad_bottom = _compute_crop_1d(cy, crop_size, H)
    c_start, c_end, pad_left, pad_right  = _compute_crop_1d(cx, crop_size, W)

    crop = img_arr[r_start:r_end, c_start:c_end]
    if pad_top or pad_bottom or pad_left or pad_right:
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    return Image.fromarray(crop.astype(np.uint8))


def rasterize_shapes(
    shapes: list[dict],
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Convert a LabelImg v5.4.1 shapes list to a binary mask.

    Handles 'polygon' and 'rectangle' shape types using PIL.
    Returns a uint8 array of shape (img_h, img_w) with 255 inside annotations.
    """
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        pts   = shape["points"]
        stype = shape.get("shape_type", "polygon")
        if stype == "polygon" and len(pts) >= 3:
            flat = [coord for pt in pts for coord in pt]
            draw.polygon(flat, fill=255)
        elif stype == "rectangle" and len(pts) == 2:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            draw.rectangle(
                [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                fill=255,
            )
    return np.array(mask)


def mask_to_patch_labels(
    mask_arr: np.ndarray,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert a (H, W) binary mask to a [196] patch-label tensor.

    Applies the same spatial transform as preprocess_val (resize shortest side
    to 224, then center crop 224×224) using NEAREST interpolation so patch
    labels stay aligned with the ViT input grid.
    Majority vote (threshold=0.5): 1 = lesion patch, 0 = background patch.
    """
    mask_pil = Image.fromarray((mask_arr > 0).astype(np.uint8) * 255, mode="L")
    mask_resized = TF.resize(mask_pil, 224, interpolation=InterpolationMode.NEAREST)
    mask_cropped = TF.center_crop(mask_resized, [224, 224])
    mask_224 = np.array(mask_cropped, dtype=np.float32) / 255.0
    
    patch_grid = mask_224.reshape((GRID_SIZE, PATCH_SIZE, GRID_SIZE, PATCH_SIZE))
    patch_means = patch_grid.mean(axis=(1, 3))
    patch_labels = (patch_means > threshold).astype(np.float32)
    return torch.from_numpy(patch_labels.reshape(-1))


def build_train_transform_lace(preprocess_val) -> transforms.Compose:
    """Training augmentation for LACE images (delegates to biomedclip version)."""
    return build_train_transform(preprocess_val)
