from __future__ import annotations

import torch
import torch.nn.functional as F


def phrase_soft_targets(
    valid_phrases: torch.Tensor,
    τ_s: float,
    phrase_to_image: torch.Tensor | None = None,
    same_image_boost: float = 0.0,
) -> torch.Tensor:
    """Compute the phrase-phrase soft target distribution used by LACE losses.

    Args:
        valid_phrases:    [N, D]  L2-normalised phrase embeddings
        τ_s:              float   soft-target temperature
        phrase_to_image:  [N]     source image index per phrase (for boost)
        same_image_boost: float   logit boost for same-image pairs (0.0 = off)
    Returns:
        [N, N] row-stochastic soft target matrix
    """
    t_j = valid_phrases @ valid_phrases.T / τ_s
    if same_image_boost > 0.0 and phrase_to_image is not None:
        same_img = phrase_to_image.unsqueeze(1) == phrase_to_image.unsqueeze(0)
        t_j = t_j + same_image_boost * same_img.float()
    return F.softmax(t_j, dim=1)


def multi_positive_soft_semantic_loss(
    img_features: torch.Tensor,
    phrase_embeddings: torch.Tensor,
    phrase_mask: torch.Tensor,
    τ: torch.Tensor,
    τ_s: float = 0.07,
    phrase_to_image: torch.Tensor | None = None,
    same_image_boost: float = 0.0,
    reweight_by_n_phrases: bool = False,
) -> torch.Tensor:
    """Unidirectional KL-divergence alignment with phrase-phrase soft targets.

    Image→phrase similarity is matched to phrase→phrase semantic similarity.
    Text-text similarity serves as a more reliable soft target than image-image
    similarity in the high visual heterogeneity bone tumor domain.

    Caller is responsible for expanding img_features to [N_total, D] by indexing
    into image embeddings with phrase_to_image (derived from phrase_mask).

    Args:
        img_features:           [N_total, D]  one image embedding per valid phrase
        phrase_embeddings:      [B, J, D]     L2-normalised phrase embeddings (padded)
        phrase_mask:            [B, J]        True = valid phrase, False = padding
        τ:                      scalar        learned contrastive temperature (clamped [0.01, 0.5])
        τ_s:                    float         fixed soft-target temperature (default 0.07)
        phrase_to_image:        [N_total]     source image index for each valid phrase
        same_image_boost:       float         added to same-image phrase-phrase logits before
                                              softmax, guaranteeing they dominate the target
                                              distribution (0.0 = original behaviour)
        reweight_by_n_phrases:  bool          if True and phrase_to_image is provided, each image
                                              contributes equally regardless of phrase count by
                                              weighting phrase i as 1 / (n_phrases_i * B)
    Returns:
        scalar KL-divergence loss
    """
    valid_phrases = phrase_embeddings[phrase_mask]              # [N_total, D]
    τ_clamped = τ.clamp(0.01, 0.5)
    s_i = img_features @ valid_phrases.T / τ_clamped           # [N_total, N_total]
    with torch.no_grad():
        soft_targets = phrase_soft_targets(valid_phrases, τ_s, phrase_to_image, same_image_boost)
    if reweight_by_n_phrases and phrase_to_image is not None:
        B = phrase_mask.shape[0]
        n_per_image = torch.bincount(phrase_to_image, minlength=B).float()  # [B]
        weights = 1.0 / (n_per_image[phrase_to_image] * B)                 # [N_total]
        kl_rows = F.kl_div(
            F.log_softmax(s_i, dim=1), soft_targets, reduction='none'
        ).sum(dim=-1)                                                        # [N_total]
        return (weights * kl_rows).sum()
    return F.kl_div(F.log_softmax(s_i, dim=1), soft_targets, reduction='batchmean')


def symmetric_soft_semantic_loss(
    img_features: torch.Tensor,
    phrase_embeddings: torch.Tensor,
    phrase_mask: torch.Tensor,
    τ: torch.Tensor,
    τ_s_phrase: float = 0.07,
    τ_s_img: float = 0.07,
    phrase_to_image: torch.Tensor | None = None,
    same_image_boost: float = 0.0,
    reweight_by_n_phrases: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric KL-divergence alignment with separate phrase and image soft targets.

    I2T: each image matches phrase-phrase semantic structure.
    T2I: each phrase matches image-image semantic structure.

    The T2I logit matrix is the transpose of the I2T matrix, so both directions
    share a single img@phrase matrix multiply.

    Args:
        img_features:           [N_total, D]  one image embedding per valid phrase
        phrase_embeddings:      [B, J, D]     L2-normalised phrase embeddings (padded)
        phrase_mask:            [B, J]        True = valid phrase
        τ:                      scalar        learned contrastive temperature
        τ_s_phrase:             float         soft-target temperature for phrase-phrase anchor (I2T)
        τ_s_img:                float         soft-target temperature for image-image anchor (T2I)
        phrase_to_image:        [N_total]     source image index for each valid phrase
        same_image_boost:       float         added to same-image phrase-phrase logits before
                                              softmax in the I2T target only (0.0 = original behaviour)
        reweight_by_n_phrases:  bool          if True and phrase_to_image is provided, each image
                                              contributes equally regardless of phrase count
    Returns:
        (total, l_i2t, l_t2i)  — total = l_i2t + l_t2i
    """
    valid_phrases = phrase_embeddings[phrase_mask]              # [N_total, D]
    τ_clamped = τ.clamp(0.01, 0.5)
    s = img_features @ valid_phrases.T / τ_clamped             # [N_total, N_total]

    with torch.no_grad():
        t_phrase = phrase_soft_targets(valid_phrases, τ_s_phrase, phrase_to_image, same_image_boost)
        t_img    = F.softmax(img_features @ img_features.T / τ_s_img, dim=1)

    if reweight_by_n_phrases and phrase_to_image is not None:
        B = phrase_mask.shape[0]
        n_per_image = torch.bincount(phrase_to_image, minlength=B).float()  # [B]
        weights = 1.0 / (n_per_image[phrase_to_image] * B)                 # [N_total]
        l_i2t = (weights * F.kl_div(
            F.log_softmax(s,   dim=1), t_phrase, reduction='none'
        ).sum(dim=-1)).sum()
        l_t2i = (weights * F.kl_div(
            F.log_softmax(s.T, dim=1), t_img,    reduction='none'
        ).sum(dim=-1)).sum()
    else:
        l_i2t = F.kl_div(F.log_softmax(s,   dim=1), t_phrase, reduction='batchmean')
        l_t2i = F.kl_div(F.log_softmax(s.T, dim=1), t_img,    reduction='batchmean')
    return l_i2t + l_t2i, l_i2t, l_t2i


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

    # +1 shifts range from [-1,1] to [0,2] so total loss stays non-negative.
    # Gradients are unchanged — 0 = maximally separated, 2 = identical.
    return (F.cosine_similarity(v_lesion, v_bg, dim=-1) + 1.0).mean()
