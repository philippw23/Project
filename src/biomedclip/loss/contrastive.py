from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_loss(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP) loss.

    Computes pairwise cosine similarities between all image and text embeddings
    in the batch, then treats the i-th image and i-th text as the positive pair
    and all other combinations as negatives.  The loss is the average of the
    image→text and text→image cross-entropies (symmetric).

    The in-batch negatives come from the current single-process batch.
    """
    # L2-normalise so that dot products equal cosine similarity.
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat  = F.normalize(text_feat,  dim=-1)

    local_batch_size = len(image_feat)

    all_image_feat = image_feat
    all_text_feat  = text_feat

    # exp(logit_scale) is the inverse temperature; clamped to prevent divergence.
    scale = logit_scale.exp().clamp(max=100.0)

    # Row i of logits_per_image: similarity of image i against every text.
    # The correct text for image i sits at column i.
    logits_per_image = image_feat @ all_text_feat.t() * scale
    logits_per_text  = text_feat  @ all_image_feat.t() * scale

    # Ground-truth labels are the indices along the diagonal of the similarity matrix.
    labels = torch.arange(local_batch_size, device=image_feat.device)

    loss_i = F.cross_entropy(logits_per_image, labels)  # image→text direction
    loss_t = F.cross_entropy(logits_per_text,  labels)  # text→image direction
    return (loss_i + loss_t) / 2.0
