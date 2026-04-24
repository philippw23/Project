from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_loss(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP) loss.

    Both feature tensors are L2-normalised before computing logits.
    logit_scale is clamped to avoid numerical instability.
    """
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat  = F.normalize(text_feat,  dim=-1)

    scale = logit_scale.exp().clamp(max=100.0)
    logits_per_image = image_feat @ text_feat.t() * scale   # (B, B)
    logits_per_text  = text_feat  @ image_feat.t() * scale  # (B, B)

    labels = torch.arange(len(image_feat), device=image_feat.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text,  labels)
    return (loss_i + loss_t) / 2.0
