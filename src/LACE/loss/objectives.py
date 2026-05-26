from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_soft_semantic_loss(
    img_features: torch.Tensor,
    phrase_embeddings: torch.Tensor,
    phrase_mask: torch.Tensor,
    τ: torch.Tensor,
    τ_s: float = 0.07,
) -> torch.Tensor:
    """Unidirectional KL-divergence alignment with phrase-phrase soft targets.

    Image→phrase similarity is matched to phrase→phrase semantic similarity.
    Text-text similarity serves as a more reliable soft target than image-image
    similarity in the high visual heterogeneity bone tumor domain.

    Caller is responsible for expanding img_features to [N_total, D] by indexing
    into image embeddings with phrase_to_image (derived from phrase_mask).

    Args:
        img_features:      [N_total, D]  one image embedding per valid phrase
        phrase_embeddings: [B, J, D]     L2-normalised phrase embeddings (padded)
        phrase_mask:       [B, J]        True = valid phrase, False = padding
        τ:                 scalar        learned contrastive temperature (clamped [0.01, 0.5])
        τ_s:               float         fixed soft-target temperature (default 0.07)
    Returns:
        scalar KL-divergence loss (batchmean, i.e. divided by N_total)
    """
    valid_phrases = phrase_embeddings[phrase_mask]              # [N_total, D]
    τ_clamped = τ.clamp(0.01, 0.5)
    s_i = img_features @ valid_phrases.T / τ_clamped           # [N_total, N_total]
    with torch.no_grad():
        t_j = valid_phrases @ valid_phrases.T / τ_s            # [N_total, N_total]
        soft_targets = F.softmax(t_j, dim=1)
    return F.kl_div(F.log_softmax(s_i, dim=1), soft_targets, reduction='batchmean')


def seg_loss(
    H_logits: torch.Tensor,
    gt_patch_labels: torch.Tensor,
) -> torch.Tensor:
    """Segmentation loss: Dice + BCE on mean heatmap vs GT patch labels.

    Args:
        H_logits:        [B, N, m]  raw logits from MaskTokenModule
        gt_patch_labels: [B, m]     float32, 1 = lesion patch, 0 = background
    """
    pred_logits = H_logits.mean(dim=1)           # [B, m]
    pred_probs  = torch.sigmoid(pred_logits)     # [B, m]
    gt = gt_patch_labels.float()

    # soft Dice
    inter = (pred_probs * gt).sum(-1)
    dice  = 1.0 - (
        2.0 * inter / (pred_probs.sum(-1) + gt.sum(-1) + 1e-8)
    ).mean()

    # BCE (per patch independently)
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt)

    return dice + bce


def ortho_loss(
    patch_tokens: torch.Tensor,
    patch_labels: torch.Tensor,
    min_lesion_patches: int = 1,
) -> torch.Tensor:
    """Lesion-background orthogonality regularizer.

    Pushes the mean lesion patch embedding away from the mean background patch
    embedding by minimising their cosine similarity.

    Operates in the raw 768-dim ViT feature space (not the projected space).
    Images with fewer than min_lesion_patches lesion patches or no background
    patches are excluded. Returns a differentiable 0.0 if no valid images remain.

    Args:
        patch_tokens:  [B, N, d_vit] raw patch token embeddings
        patch_labels:  [B, N] float32: 1 = lesion patch, 0 = background
        min_lesion_patches: minimum number of lesion patches required
    """
    lesion_mask = patch_labels.unsqueeze(-1)
    bg_mask     = (1.0 - patch_labels).unsqueeze(-1)

    lesion_count = lesion_mask.sum(dim=1)
    bg_count     = bg_mask.sum(dim=1)

    valid = (
        (lesion_count.squeeze(-1) >= min_lesion_patches) &
        (bg_count.squeeze(-1) >= 1)
    )

    if not valid.any():
        return patch_tokens.sum() * 0.0

    pt = patch_tokens[valid]
    lm = lesion_mask[valid]
    bm = bg_mask[valid]
    lc = lesion_count[valid].clamp(min=1)
    bc = bg_count[valid].clamp(min=1)

    v_lesion = (pt * lm).sum(dim=1) / lc
    v_bg     = (pt * bm).sum(dim=1) / bc

    return F.cosine_similarity(v_lesion, v_bg, dim=-1).mean()
