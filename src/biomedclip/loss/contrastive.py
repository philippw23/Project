from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _gather_with_local_grad(features: torch.Tensor) -> torch.Tensor:
    """Gather features from all ranks while preserving local gradients."""
    if not (dist.is_available() and dist.is_initialized()):
        return features

    gathered = [torch.empty_like(features) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, features.contiguous())
    gathered[dist.get_rank()] = features
    return torch.cat(gathered, dim=0)


def clip_loss(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    logit_scale: torch.Tensor,
    gather_distributed: bool = False,
) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP) loss.

    Both feature tensors are L2-normalised before computing logits.
    logit_scale is clamped to avoid numerical instability.
    If gather_distributed is true, negatives are gathered across DDP ranks.
    """
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat  = F.normalize(text_feat,  dim=-1)

    local_batch_size = len(image_feat)
    if gather_distributed:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        label_offset = rank * local_batch_size
        all_image_feat = _gather_with_local_grad(image_feat)
        all_text_feat  = _gather_with_local_grad(text_feat)
    else:
        label_offset = 0
        all_image_feat = image_feat
        all_text_feat  = text_feat

    scale = logit_scale.exp().clamp(max=100.0)
    logits_per_image = image_feat @ all_text_feat.t() * scale
    logits_per_text  = text_feat  @ all_image_feat.t() * scale

    labels = torch.arange(local_batch_size, device=image_feat.device) + label_offset
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text,  labels)
    return (loss_i + loss_t) / 2.0
