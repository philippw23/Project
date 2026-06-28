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
) -> tuple[Image.Image, np.ndarray]:
    """Return a (image_crop, mask_crop) pair of crop_size centered on the mask centroid.

    Both the image and mask are cropped with identical coordinates so that
    patch labels derived from mask_crop stay aligned with the ViT input grid.
    Falls back to resizing the full image/mask when the mask is empty.
    """
    binary = mask_arr > 0
    if not np.any(binary):
        img_crop  = image.resize((crop_size, crop_size), RESAMPLE_BICUBIC)
        mask_crop = np.zeros((crop_size, crop_size), dtype=mask_arr.dtype)
        return img_crop, mask_crop

    rows, cols = np.where(binary)
    cy = float(rows.mean())
    cx = float(cols.mean())

    img_arr = np.array(image.convert("RGB"))
    H, W = img_arr.shape[:2]

    r_start, r_end, pad_top,  pad_bottom = _compute_crop_1d(cy, crop_size, H)
    c_start, c_end, pad_left, pad_right  = _compute_crop_1d(cx, crop_size, W)

    img_crop = img_arr[r_start:r_end, c_start:c_end]
    mask_crop = mask_arr[r_start:r_end, c_start:c_end]
    if pad_top or pad_bottom or pad_left or pad_right:
        pad2d = ((pad_top, pad_bottom), (pad_left, pad_right))
        img_crop  = np.pad(img_crop,  (*pad2d, (0, 0)), mode="constant", constant_values=0)
        mask_crop = np.pad(mask_crop, pad2d,             mode="constant", constant_values=0)
    return Image.fromarray(img_crop.astype(np.uint8)), mask_crop


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


def mask_to_patch_labels(mask_arr: np.ndarray) -> torch.Tensor:
    """Convert a (H, W) binary mask to a [196] soft coverage-fraction tensor.

    Applies the same spatial transform as preprocess_val (resize shortest side
    to 224, then center crop 224×224) using NEAREST interpolation so patch
    labels stay aligned with the ViT input grid. Each value is the fraction of
    the 16×16 patch covered by the mask (0.0–1.0).
    """
    mask_pil = Image.fromarray((mask_arr > 0).astype(np.uint8) * 255, mode="L")
    mask_resized = TF.resize(mask_pil, 224, interpolation=InterpolationMode.NEAREST)
    mask_cropped = TF.center_crop(mask_resized, [224, 224])
    mask_224 = np.array(mask_cropped, dtype=np.float32) / 255.0
    return patchify_mask_224(mask_224)


def patchify_mask_224(mask_224: np.ndarray, min_coverage: float = 0.10) -> torch.Tensor:
    """Patchify a 224x224 binary mask into [196] soft coverage fractions.

    Each value is the fraction of the 16x16 patch covered by the mask (0.0–1.0).
    Patches with coverage below min_coverage are zeroed out — they are boundary
    slivers where the mask clips a corner and should be treated as background.
    """
    patch_grid  = mask_224.reshape((GRID_SIZE, PATCH_SIZE, GRID_SIZE, PATCH_SIZE))
    patch_means = patch_grid.mean(axis=(1, 3)).reshape(-1).astype(np.float32)
    patch_means[patch_means < min_coverage] = 0.0
    return torch.from_numpy(patch_means)


def synchronized_train_transform(
    img_pil: Image.Image,
    mask_pil: Image.Image,
    mean,
    std,
    size: int = 224,
    scale: tuple[float, float] = (0.8, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    rotation_deg: float = 10.0,
    sharpness_p: float = 0.3,
    blur_kernel: int = 3,
    blur_sigma: tuple[float, float] = (0.1, 1.0),
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train-time augmentation that keeps image and mask spatially aligned.

    Mirrors ``build_train_transform`` (RandomResizedCrop + RandomRotation +
    sharpness + Gaussian blur + normalize), but the geometric ops use the same
    sampled parameters for the image and the mask, so the returned patch labels
    stay registered to the augmented image. Photometric ops are image-only.

    Returns ``(image_tensor [3, size, size], patch_labels [196])``.
    """
    # ── geometric augmentation (shared parameters) ────────────────────────────
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        img_pil, scale=list(scale), ratio=list(ratio)
    )
    img = TF.resized_crop(img_pil, i, j, h, w, [size, size],
                          interpolation=InterpolationMode.BICUBIC)
    msk = TF.resized_crop(mask_pil, i, j, h, w, [size, size],
                          interpolation=InterpolationMode.NEAREST)

    angle = float(torch.empty(1).uniform_(-rotation_deg, rotation_deg).item())
    img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
    msk = TF.rotate(msk, angle, interpolation=InterpolationMode.NEAREST, fill=0)

    # ── photometric augmentation (image only) ─────────────────────────────────
    img = transforms.RandomAdjustSharpness(sharpness_factor=2, p=sharpness_p)(img)
    img = transforms.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)(img)

    img_t = TF.to_tensor(img)
    img_t = TF.normalize(img_t, mean=list(mean), std=list(std))

    mask_224 = (np.array(msk, dtype=np.float32) > 127.0).astype(np.float32)
    patch_labels = patchify_mask_224(mask_224, threshold)
    return img_t, patch_labels


def build_train_transform_lace(preprocess_val) -> transforms.Compose:
    """Training augmentation for LACE images (delegates to biomedclip version)."""
    return build_train_transform(preprocess_val)
