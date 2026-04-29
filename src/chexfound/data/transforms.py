from __future__ import annotations

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_preprocess_val_chexfound() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(512, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_train_transform_chexfound(preprocess_val: transforms.Compose) -> transforms.Compose:
    norm = next(t for t in preprocess_val.transforms if isinstance(t, transforms.Normalize))
    return transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        norm,
    ])
