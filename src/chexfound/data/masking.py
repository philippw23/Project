# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np


class MaskingGenerator:
    """Generates random rectangular block masks over a 2-D patch grid.

    Masks are produced in patch-grid coordinates, so every masked position
    corresponds to exactly one fully-masked patch token. The masking strategy
    repeatedly places random rectangles with random aspect ratios until the
    requested number of patches is covered.

    Args:
        input_size: Height and width of the patch grid as ``(H, W)`` or a
            single integer for a square grid (e.g. 32 for a 512px image with
            16px patches).
        num_masking_patches: Maximum and default number of patches to mask.
        min_num_patches: Minimum area (in patches) of a single mask rectangle.
        max_num_patches: Maximum area of a single rectangle. Defaults to
            ``num_masking_patches``.
        min_aspect: Minimum aspect ratio of a mask rectangle.
        max_aspect: Maximum aspect ratio. Defaults to ``1 / min_aspect``.
    """

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        """Attempt to place one random rectangle onto ``mask``.

        Tries up to 10 times to find a valid rectangle (random area and aspect
        ratio) that fits within the grid and does not exceed ``max_mask_patches``
        new patches. Stops and returns as soon as one valid rectangle is placed.

        Args:
            mask: Boolean numpy array of shape ``(H, W)`` tracking already-masked
                patches. Modified in-place.
            max_mask_patches: Maximum number of *new* patches this rectangle may
                cover (already-masked patches do not count).

        Returns:
            Number of newly masked patches added (0 if no valid rectangle found).
        """
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        """Generate a patch mask with the requested number of masked patches.

        Repeatedly calls ``_mask`` to place rectangles until ``num_masking_patches``
        positions are covered or no further progress is possible.

        Args:
            num_masking_patches: Target number of patches to mask. Pass 0 to
                return an all-False mask (no masking).

        Returns:
            Boolean numpy array of shape ``(H, W)`` where ``True`` marks a
            masked patch position.
        """
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask
