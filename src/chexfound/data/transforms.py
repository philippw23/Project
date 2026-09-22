# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


# ── ImageNet normalisation constants ─────────────────────────────────────────

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)

# Legacy aliases used elsewhere in this project
IMAGENET_MEAN = IMAGENET_DEFAULT_MEAN
IMAGENET_STD  = IMAGENET_DEFAULT_STD


# ── Upstream CheXFound transforms ────────────────────────────────────────────

class GaussianBlur(transforms.RandomApply):
    """Apply Gaussian Blur to a PIL image with probability p."""

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """Convert PIL / ndarray to tensor, or pass through if already a tensor."""

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


class RescaleImage:
    """Per-channel min-max rescale to [0, 1]."""

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        elif not torch.is_tensor(image):
            raise TypeError("Input should be numpy.ndarray or torch.Tensor")
        min_val = image.reshape(image.shape[0], -1).min(dim=1)[0].reshape(-1, 1, 1)
        max_val = image.reshape(image.shape[0], -1).max(dim=1)[0].reshape(-1, 1, 1)
        return (image - min_val) / (max_val - min_val)


class RandomRot90:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image):
        if np.random.uniform() < self.p:
            k = np.random.choice([1, 2, 3])
            image = image.rotate(90 * k, expand=True)
        return image


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    rot90: bool = False,
) -> transforms.Compose:
    transforms_list = [transforms.RandomResizedCrop(crop_size, scale=(0.75, 1), interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    if rot90:
        transforms_list.append(RandomRot90(p=0.5))
    transforms_list.extend([MaybeToTensor(), RescaleImage(), make_normalize_transform(mean=mean, std=std)])
    return transforms.Compose(transforms_list)


def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    flip: bool = False,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        RescaleImage(),
        make_normalize_transform(mean=mean, std=std),
    ]
    if flip:
        transforms_list.append(transforms.RandomHorizontalFlip(p=1))
    return transforms.Compose(transforms_list)


# ── Local downstream helpers (kept for chexfound/train/downstream.py) ────────

def build_preprocess_val_chexfound(image_size: int = 512) -> transforms.Compose:
    # No resize→crop overshoot: resize the short edge to image_size and take the
    # centre square. The upstream 256/224 overshoot suits DINOv2's wide
    # RandomResizedCrop scale=(0.08, 1.0); build_train_transform_chexfound below
    # uses scale=(0.8, 1.0), so an overshoot would only add a train/test scale
    # mismatch. Keep resize == crop.
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_train_transform_chexfound(preprocess_val: transforms.Compose) -> transforms.Compose:
    norm = next(t for t in preprocess_val.transforms if isinstance(t, transforms.Normalize))
    crop = next(t for t in preprocess_val.transforms if isinstance(t, transforms.CenterCrop))
    image_size = crop.size[0] if isinstance(crop.size, (list, tuple)) else crop.size
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        norm,
    ])
