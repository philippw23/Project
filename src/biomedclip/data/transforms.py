from __future__ import annotations

import math

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def _compute_crop_1d(
    center: float,
    size: int,
    dim_size: int,
) -> tuple[int, int, int, int]:
    """Compute crop parameters along a single image axis.

    Centers a window of length *size* on *center*, then shifts it as far as
    possible into [0, dim_size) to maximise real pixels before any padding is
    added.  When size > dim_size the entire axis is used and padding is split
    to keep *center* as close to the output centre as possible.

    Returns (slice_start, slice_end, pad_before, pad_after).
    """
    if size <= dim_size:
        start = round(center - size / 2)
        start = max(0, min(start, dim_size - size))
        return start, start + size, 0, 0

    pad_total = size - dim_size
    ideal_pad_before = round(size / 2 - center)
    pad_before = max(0, min(pad_total, ideal_pad_before))
    return 0, dim_size, pad_before, pad_total - pad_before


def _crop_side(
    box_h: int,
    box_w: int,
    H: int,
    W: int,
    context_fraction: float,
    context_mode: str,
    min_crop_size: int,
) -> int:
    """Compute the square crop side for a lesion bbox.

    context_mode="image"  (default): context_px = ceil(min(H, W) * context_fraction)
        A fixed pixel margin derived from the *image* size — small lesions end up
        with proportionally more surrounding context than large lesions.
    context_mode="lesion": context_px = ceil(max(box_h, box_w) * context_fraction)
        Margin scales with the *lesion* size instead, so every lesion gets the
        same relative amount of context. min_crop_size then acts as an explicit
        floor to avoid degenerate crops for very small lesions.
    """
    if context_mode == "lesion":
        context_px = math.ceil(max(box_h, box_w) * context_fraction)
    elif context_mode == "image":
        context_px = math.ceil(min(H, W) * context_fraction)
    else:
        raise ValueError(f"context_mode must be 'image' or 'lesion', got {context_mode!r}")

    side = max(box_h, box_w) + 2 * context_px
    if min_crop_size > 0:
        side = max(side, min_crop_size)
    return side


def crop_around_mask(
    image_arr: np.ndarray,
    mask: np.ndarray,
    enable_crop: bool = True,
    context_fraction: float = 0.15,
    context_mode: str = "image",
    min_crop_size: int = 0,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
) -> np.ndarray:
    """Crop a square region around the lesion defined by *mask*.

    See :func:`_crop_side` for how context_mode="image" vs. "lesion" change the
    context-margin computation, and min_crop_size for the optional floor on the
    output side length.
    """
    if not enable_crop:
        return image_arr

    binary_mask = mask > 0
    if not np.any(binary_mask):
        return image_arr

    rows, cols = np.where(binary_mask)
    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())

    box_h = r_max - r_min + 1
    box_w = c_max - c_min + 1

    H, W = image_arr.shape[:2]

    side = _crop_side(box_h, box_w, H, W, context_fraction, context_mode, min_crop_size)

    cy = (r_min + r_max) / 2
    cx = (c_min + c_max) / 2

    r_start, r_end, pad_top, pad_bottom = _compute_crop_1d(cy, side, H)
    c_start, c_end, pad_left, pad_right = _compute_crop_1d(cx, side, W)

    crop = image_arr[r_start:r_end, c_start:c_end]

    if pad_top or pad_bottom or pad_left or pad_right:
        pad_width = (
            ((pad_top, pad_bottom), (pad_left, pad_right))
            if image_arr.ndim == 2
            else ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        )
        kwargs = {"constant_values": pad_value} if pad_mode == "constant" else {}
        crop = np.pad(crop, pad_width, mode=pad_mode, **kwargs)

    return crop


def compute_crop_box(
    image_arr: np.ndarray,
    mask: np.ndarray,
    context_fraction: float = 0.15,
) -> tuple[int, int, int, int]:
    """Return the crop box (r_start, r_end, c_start, c_end) in original image coordinates.

    Uses the same geometry as :func:`crop_around_mask`.  When the mask is empty
    or the box would exceed the image boundary the coordinates are clamped to the
    image extent (matching the zero-padding behaviour of the crop function).
    """
    binary_mask = mask > 0
    if not np.any(binary_mask):
        H, W = image_arr.shape[:2]
        return 0, H, 0, W

    rows, cols = np.where(binary_mask)
    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())

    box_h = r_max - r_min + 1
    box_w = c_max - c_min + 1

    H, W = image_arr.shape[:2]
    context_px = math.ceil(min(H, W) * context_fraction)
    side = max(box_h, box_w) + 2 * context_px

    cy = (r_min + r_max) / 2
    cx = (c_min + c_max) / 2

    r_start, r_end, _, _ = _compute_crop_1d(cy, side, H)
    c_start, c_end, _, _ = _compute_crop_1d(cx, side, W)

    return r_start, r_end, c_start, c_end


class SquarePad:
    """Pad the shorter side so the image becomes square (black border, centered)."""

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        side = max(w, h)
        result = Image.new(image.mode, (side, side), 0)
        result.paste(image, ((side - w) // 2, (side - h) // 2))
        return result


def build_preprocess_val(reference_preprocess, image_size: int) -> transforms.Compose:
    """Build a val/test pipeline at an arbitrary resolution.

    Derives the resize→crop ratio from reference_preprocess so the overshoot
    convention (e.g. 256→224 for BiomedCLIP) scales correctly to any target size.
    """
    if image_size == 224:
        return reference_preprocess
    resize_t = next(t for t in reference_preprocess.transforms if isinstance(t, transforms.Resize))
    crop_t   = next(t for t in reference_preprocess.transforms if isinstance(t, transforms.CenterCrop))
    norm_t   = next(t for t in reference_preprocess.transforms if isinstance(t, transforms.Normalize))
    ref_resize = resize_t.size if isinstance(resize_t.size, int) else resize_t.size[0]
    ref_crop   = crop_t.size   if isinstance(crop_t.size,   int) else crop_t.size[0]
    new_resize = int(image_size * ref_resize / ref_crop)
    return transforms.Compose([
        transforms.Resize(new_resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        norm_t,
    ])


def build_train_transform(preprocess_val) -> transforms.Compose:
    """Build a custom training augmentation pipeline.

    Normalization mean/std are taken from preprocess_val so they always match
    the model, regardless of which checkpoint is loaded. Crop size is read
    dynamically from preprocess_val so it scales correctly with --image_size.

    Augmentations chosen for bone-tumour X-rays:
      - RandomResizedCrop: simulates varying patient positioning and zoom
      - RandomRotation(10°): small tilts from patient/table angle
      - RandomAdjustSharpness: varies image sharpness/contrast
      - GaussianBlur: simulates different acquisition sharpness
    Horizontal/vertical flips are intentionally omitted — left/right anatomy
    is clinically meaningful in radiographs.
    """
    norm = next(t for t in preprocess_val.transforms if isinstance(t, transforms.Normalize))
    crop = next(t for t in preprocess_val.transforms if isinstance(t, transforms.CenterCrop))
    image_size = crop.size if isinstance(crop.size, int) else crop.size[0]

    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        norm,
    ])
