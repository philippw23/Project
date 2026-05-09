# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    """Collate a batch of DINO/iBOT samples and generate iBOT patch masks.

    Stacks global and local crops from all samples, then assigns a randomly
    sampled patch mask to each global-crop image. A fraction ``mask_probability``
    of images receive a non-zero mask whose mask ratio is drawn uniformly from
    ``mask_ratio_tuple = (min_ratio, max_ratio)`` via linearly spaced probability
    bins. The remaining images get an all-zero mask (no patches masked).

    Args:
        samples_list: List of ``(augmented_dict, label)`` tuples returned by the
            dataset. ``augmented_dict`` must contain ``"global_crops"`` and
            ``"local_crops"`` keys with pre-transformed tensors.
        mask_ratio_tuple: ``(min_ratio, max_ratio)`` controlling the range of
            masked-patch fractions for the iBOT objective.
        mask_probability: Fraction of images in the batch that receive a
            non-zero mask (e.g. 0.5 masks half the batch).
        dtype: Target dtype for the stacked crop tensors (e.g. ``torch.float32``).
        n_tokens: Total number of patch tokens per image
            (``(global_crops_size // patch_size) ** 2``).
        mask_generator: Callable that accepts an integer number of patches to
            mask and returns a 2-D boolean array of shape
            ``(n_patches_h, n_patches_w)``.

    Returns:
        dict with keys:
            - ``collated_global_crops``: ``(B, C, H, W)`` tensor of global crops.
            - ``collated_local_crops``: ``(B*n_local, C, h, w)`` tensor of local crops.
            - ``collated_masks``: ``(B, N)`` bool tensor; ``True`` = masked patch.
            - ``mask_indices_list``: 1-D tensor of flat indices of all masked patches.
            - ``masks_weight``: Per-patch loss weight (``1 / n_masked_in_image``),
              broadcast to masked positions only.
            - ``upperbound``: Sum of maximum masked-patch counts across masked
              images; used to pre-allocate buffers in the iBOT loss.
            - ``n_masked_patches``: Scalar tensor with the total number of masked
              patches across the batch.
    """
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
