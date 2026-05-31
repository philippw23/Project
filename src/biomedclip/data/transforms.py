from __future__ import annotations

import math

import numpy as np
from PIL import Image
from torchvision import transforms


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


def crop_around_mask(
    image_arr: np.ndarray,
    mask: np.ndarray,
    enable_crop: bool = True,
    context_fraction: float = 0.15,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
) -> np.ndarray:
    """Crop a square region around the lesion defined by *mask*.

    The crop side is computed as::

        context_px = ceil(min(H, W) * context_fraction)
        side = max(bbox_height, bbox_width) + 2 * context_px

    Because *context_px* is a fixed number of pixels (derived from image size,
    not lesion size), small lesions automatically receive proportionally more
    surrounding context than large lesions.
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

    context_px = math.ceil(min(H, W) * context_fraction)
    side = max(box_h, box_w) + 2 * context_px

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


def crop_around_mask_pair(
    image_arr: np.ndarray,
    mask: np.ndarray,
    context_fraction: float = 0.15,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop both image and mask to the same square region around the lesion.

    Uses identical geometry to :func:`crop_around_mask` (bbox + context margin,
    padded to a square when the region extends past the image boundary).  The
    returned image and mask crops are always square with the same side length.

    Falls back to returning the original arrays when the mask is empty.
    """
    binary_mask = mask > 0
    if not np.any(binary_mask):
        return image_arr, mask

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

    r_start, r_end, pad_top,  pad_bottom = _compute_crop_1d(cy, side, H)
    c_start, c_end, pad_left, pad_right  = _compute_crop_1d(cx, side, W)

    img_crop  = image_arr[r_start:r_end, c_start:c_end]
    mask_crop = mask[r_start:r_end, c_start:c_end]

    if pad_top or pad_bottom or pad_left or pad_right:
        img_pad = (
            ((pad_top, pad_bottom), (pad_left, pad_right))
            if image_arr.ndim == 2
            else ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        )
        img_crop  = np.pad(img_crop, img_pad, mode="constant", constant_values=pad_value)
        mask_crop = np.pad(mask_crop, ((pad_top, pad_bottom), (pad_left, pad_right)),
                           mode="constant", constant_values=0)

    return img_crop, mask_crop


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


def build_train_transform(preprocess_val) -> transforms.Compose:
    """Build a custom training augmentation pipeline.

    Normalization mean/std are taken from preprocess_val so they always match
    the model, regardless of which checkpoint is loaded.

    Augmentations chosen for bone-tumour X-rays:
      - RandomResizedCrop: simulates varying patient positioning and zoom
      - RandomRotation(10°): small tilts from patient/table angle
      - RandomAdjustSharpness: varies image sharpness/contrast
      - GaussianBlur: simulates different acquisition sharpness
    Horizontal/vertical flips are intentionally omitted — left/right anatomy
    is clinically meaningful in radiographs.
    """
    norm = next(t for t in preprocess_val.transforms if isinstance(t, transforms.Normalize))

    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        norm,
    ])
