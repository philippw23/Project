from __future__ import annotations

from torchvision import transforms

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def build_train_transform() -> transforms.Compose:
    """Moderate augmentation appropriate for bone-tumour X-rays.

    No horizontal/vertical flip: left-right anatomy is clinically meaningful.
    RandomAffine covers realistic patient positioning variation.
    GaussianBlur simulates acquisition sharpness differences.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def build_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
