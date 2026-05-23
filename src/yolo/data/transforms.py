"""Image transforms for the YOLO downstream classifier.

SquarePad must be applied first so that normalised bbox coordinates
(computed by bboxes_from_mask with the same padding logic) remain valid
after resizing.  Spatial augmentations (crop, flip, affine) are intentionally
omitted because they would invalidate the stored bbox targets.

YOLO models expect pixels in [0, 1] — no ImageNet mean/std normalisation.
"""
from __future__ import annotations

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


class SquarePad:
    """Pad image to a square with black borders (matches bboxes_from_mask geometry)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h   = img.size
        side   = max(w, h)
        pl     = (side - w) // 2          # pad left
        pt     = (side - h) // 2          # pad top
        pr     = side - w - pl            # pad right
        pb     = side - h - pt            # pad bottom
        return TF.pad(img, [pl, pt, pr, pb], fill=0)


def build_train_transform(imgsz: int = 640) -> transforms.Compose:
    """Augmentations that preserve bbox validity (no spatial transforms)."""
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((imgsz, imgsz)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ])


def build_val_transform(imgsz: int = 640) -> transforms.Compose:
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])
