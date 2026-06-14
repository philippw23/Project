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
    t2i_mode: str = "image_image",
    descriptor_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric alignment: I2T soft KL + T2I controlled by t2i_mode.

    I2T: each image matches phrase-phrase semantic structure (soft KL).
    T2I: each phrase matches an anchor distribution controlled by t2i_mode:
         - "image_image": image-image cosine similarity (soft KL)
         - "text_text":   phrase-phrase cosine similarity (soft KL, same as I2T anchor)
         - "descriptor":  21-dim binary descriptor cosine similarity (soft KL, MedCLIP eq. 4)
         - "infonce":     hard InfoNCE — phrase vs unique image embeddings [N_total, B]

    For soft modes the T2I logit matrix is the transpose of the shared I2T matrix.
    For "infonce" a separate [N_total, B] logit matrix is built from unique image features.

    Args:
        img_features:           [N_total, D]   one image embedding per valid phrase
        phrase_embeddings:      [B, J, D]      L2-normalised phrase embeddings (padded)
        phrase_mask:            [B, J]         True = valid phrase
        τ:                      scalar         learned contrastive temperature
        τ_s_phrase:             float          soft-target temperature for phrase-phrase (I2T)
        τ_s_img:                float          soft-target temperature for image-image
                                               (only used when t2i_mode="image_image")
        phrase_to_image:        [N_total]      source image index for each valid phrase
        same_image_boost:       float          added to same-image phrase-phrase logits before
                                               softmax in I2T target only (0.0 = off)
        reweight_by_n_phrases:  bool           if True, each image contributes equally
        t2i_mode:               str            "image_image"|"text_text"|"descriptor"|"infonce"
        descriptor_features:    [N_total, 21]  precomputed binary descriptor vectors per phrase
                                               (required when t2i_mode="descriptor")
    Returns:
        (total, l_i2t, l_t2i)  — total = l_i2t + l_t2i
    """
    assert phrase_to_image is not None or t2i_mode != "infonce", \
        "phrase_to_image is required for t2i_mode='infonce'"
    valid_phrases = phrase_embeddings[phrase_mask]              # [N_total, D]
    τ_clamped = τ.clamp(0.01, 0.5)
    B = phrase_mask.shape[0]

    # Per-phrase weights for equal-image contribution
    weights = None
    if reweight_by_n_phrases and phrase_to_image is not None:
        n_per_image = torch.bincount(phrase_to_image, minlength=B).float()
        weights = 1.0 / (n_per_image[phrase_to_image] * B)     # [N_total]

    # I2T: img → phrase soft KL
    s = img_features @ valid_phrases.T / τ_clamped             # [N_total, N_total]
    with torch.no_grad():
        t_phrase = phrase_soft_targets(valid_phrases, τ_s_phrase, phrase_to_image, same_image_boost)
    if weights is not None:
        l_i2t = (weights * F.kl_div(
            F.log_softmax(s, dim=1), t_phrase, reduction='none'
        ).sum(dim=-1)).sum()
    else:
        l_i2t = F.kl_div(F.log_softmax(s, dim=1), t_phrase, reduction='batchmean')

    # T2I
    if t2i_mode == "infonce":
        # Hard InfoNCE: phrase → unique image [N_total, B]
        D = img_features.shape[-1]
        img_unique = torch.zeros(B, D, device=img_features.device, dtype=img_features.dtype)
        img_unique.scatter_(0, phrase_to_image.unsqueeze(1).expand(-1, D), img_features)
        s_t2i = valid_phrases @ img_unique.T / τ_clamped       # [N_total, B]
        if weights is not None:
            l_t2i = (weights * F.cross_entropy(s_t2i, phrase_to_image, reduction='none')).sum()
        else:
            l_t2i = F.cross_entropy(s_t2i, phrase_to_image)
    else:
        with torch.no_grad():
            if t2i_mode == "text_text":
                t_t2i = t_phrase
            elif t2i_mode == "descriptor" and descriptor_features is not None:
                d = F.normalize(descriptor_features.float(), dim=-1)
                t_t2i = F.softmax(d @ d.T, dim=1)              # no temperature, MedCLIP eq. 4
            else:  # image_image
                t_t2i = F.softmax(img_features @ img_features.T / τ_s_img, dim=1)
        if weights is not None:
            l_t2i = (weights * F.kl_div(
                F.log_softmax(s.T, dim=1), t_t2i, reduction='none'
            ).sum(dim=-1)).sum()
        else:
            l_t2i = F.kl_div(F.log_softmax(s.T, dim=1), t_t2i, reduction='batchmean')

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


def gloria_local_loss(
    P_proj: torch.Tensor,
    phrase_embeddings: torch.Tensor,
    phrase_mask: torch.Tensor,
    τ: torch.Tensor,
    τ2: float = 0.07,
    τ_s: float = 0.07,
    τ_s_img: float = 0.07,
    phrase_to_image: torch.Tensor | None = None,
    same_image_boost: float = 0.0,
    reweight_by_n_phrases: bool = False,
    λ_t2i: float = 1.0,
    t2i_mode: str = "infonce",
    img_cls_features: torch.Tensor | None = None,
    descriptor_features: torch.Tensor | None = None,
    patch_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GLoRIA-style local phrase-patch alignment (GLoRIA eq. 7 + eq. 8).

    If ``patch_mask`` ([B_sim, M], 1 = lesion patch) is given, phrase→patch
    attention is restricted to lesion patches (background patches masked to
    -inf before softmax) in both I2T and T2I directions. Images with no lesion
    patch fall back to attending all patches (avoids degenerate all-(-inf) rows).

    I2T (soft KL): phrase-specific attention features vs all phrases in batch,
                   anchored by phrase-phrase soft targets.
    T2I: phrase embeddings vs cross-image attention features [N_total, B_sim],
         loss type controlled by t2i_mode:
         - "infonce":     hard InfoNCE, diagonal positive (default)
         - "text_text":   soft KL, per-image aggregated phrase-phrase targets
         - "image_image": soft KL, image-image cosine similarity targets; requires img_cls_features
         - "descriptor":  soft KL, per-image aggregated descriptor similarity targets

    Args:
        P_proj:              [B_sim, M, D]   L2-normalised projected patch embeddings
        phrase_embeddings:   [B_sim, J, D]   L2-normalised phrase embeddings (padded)
        phrase_mask:         [B_sim, J]      True = valid phrase
        τ:                   scalar          learned contrastive temperature (clamped [0.01, 0.5])
        τ2:                  float           attention softmax temperature
        τ_s:                 float           soft-target temperature for phrase-phrase (I2T and T2I text_text)
        τ_s_img:             float           soft-target temperature for image-image (T2I image_image)
        phrase_to_image:     [N_total]       source image index (0..B_sim-1) per valid phrase
        same_image_boost:    float           logit boost for same-image pairs in I2T soft target
        reweight_by_n_phrases: bool          weight each phrase by 1/n_i so images contribute equally
        λ_t2i:               float           weight for T2I loss
        t2i_mode:            str             one of "infonce", "text_text", "image_image", "descriptor"
        img_cls_features:    [B_sim, D]      global CLS embeddings of sim_valid images
                                             (required when t2i_mode="image_image")
        descriptor_features: [B_sim, 21]     descriptor vectors per image
                                             (required when t2i_mode="descriptor")
    Returns:
        (total, l_i2t, l_t2i)
    """
    assert phrase_to_image is not None, "phrase_to_image is required for gloria_local_loss"
    B_sim = P_proj.shape[0]
    valid_phrases = phrase_embeddings[phrase_mask]          # [N_total, D]
    τ_clamped = τ.clamp(0.01, 0.5)

    # Per-phrase weights for equal-image contribution
    weights = None
    if reweight_by_n_phrases:
        n_per_image = torch.bincount(phrase_to_image, minlength=B_sim).float()
        weights     = 1.0 / (n_per_image[phrase_to_image] * B_sim)

    # ── I2T: phrase-specific attention features from own image ────────────────
    c_j_list = []
    for i in range(B_sim):
        phrases_i = valid_phrases[phrase_to_image == i]     # [n_i, D]
        if phrases_i.shape[0] == 0:
            continue
        P_i      = P_proj[i]                                # [M, D]
        logits_i = phrases_i @ P_i.T / τ2                   # [n_i, M]
        if patch_mask is not None:
            m_i = patch_mask[i].bool()                      # [M]
            if m_i.any():                                   # else: attend all patches
                logits_i = logits_i.masked_fill(~m_i.unsqueeze(0), float("-inf"))
        alpha_i = F.softmax(logits_i, dim=-1)               # [n_i, M]
        c_j_list.append(alpha_i @ P_i)                      # [n_i, D]
    c_j = F.normalize(torch.cat(c_j_list, dim=0), dim=-1)  # [N_total, D]

    s_i2t = c_j @ valid_phrases.T / τ_clamped              # [N_total, N_total]
    with torch.no_grad():
        soft_targets = phrase_soft_targets(
            valid_phrases, τ_s, phrase_to_image, same_image_boost
        )
    if weights is not None:
        l_i2t = (weights * F.kl_div(
            F.log_softmax(s_i2t, dim=1), soft_targets, reduction='none'
        ).sum(dim=-1)).sum()
    else:
        l_i2t = F.kl_div(F.log_softmax(s_i2t, dim=1), soft_targets, reduction='batchmean')

    # ── T2I: cross-image attention features ───────────────────────────────────
    # s_t2i[n, k] = similarity of phrase n to its attention feature on image k
    attn_scores  = torch.einsum('nd,kmd->nkm', valid_phrases, P_proj) / τ2  # [N_total, B_sim, M]
    if patch_mask is not None:
        pm      = patch_mask.bool()                       # [B_sim, M]
        has_les = pm.any(dim=1)                           # [B_sim]
        block   = (~pm) & has_les.unsqueeze(1)            # background to block (only where lesion exists)
        attn_scores = attn_scores.masked_fill(block.unsqueeze(0), float("-inf"))
    attn_weights = F.softmax(attn_scores, dim=-1)
    c_cross      = F.normalize(
        torch.einsum('nkm,kmd->nkd', attn_weights, P_proj), dim=-1
    )                                                                         # [N_total, B_sim, D]
    s_t2i = torch.einsum('nd,nkd->nk', valid_phrases, c_cross) / τ_clamped  # [N_total, B_sim]

    if t2i_mode == "infonce":
        if weights is not None:
            l_t2i = (weights * F.cross_entropy(s_t2i, phrase_to_image, reduction='none')).sum()
        else:
            l_t2i = F.cross_entropy(s_t2i, phrase_to_image)
    else:
        with torch.no_grad():
            if t2i_mode == "text_text":
                # Aggregate phrase similarities per image: t[n, k] = sum of phrase_n·phrase_j for j in image k
                pairwise = valid_phrases @ valid_phrases.T / τ_s             # [N_total, N_total]
                t_t2i = torch.zeros(valid_phrases.shape[0], B_sim,
                                    device=valid_phrases.device, dtype=valid_phrases.dtype)
                for k in range(B_sim):
                    t_t2i[:, k] = pairwise[:, phrase_to_image == k].sum(dim=-1)
                t_t2i = F.softmax(t_t2i, dim=-1)
            elif t2i_mode == "image_image" and img_cls_features is not None:
                # Image-image cosine similarity, indexed by source image of each phrase
                t_t2i = F.softmax(
                    img_cls_features[phrase_to_image] @ img_cls_features.T / τ_s_img, dim=-1
                )                                                             # [N_total, B_sim]
            else:  # descriptor (or image_image fallback without features)
                if descriptor_features is not None:
                    d = F.normalize(descriptor_features.float(), dim=-1)     # [B_sim, 21]
                    t_t2i = F.softmax(
                        (d[phrase_to_image] @ d.T), dim=-1                   # [N_total, B_sim]
                    )
                else:
                    # Fallback: uniform (should not happen with correct args)
                    t_t2i = torch.full(
                        (valid_phrases.shape[0], B_sim), 1.0 / B_sim,
                        device=valid_phrases.device, dtype=valid_phrases.dtype,
                    )
        if weights is not None:
            l_t2i = (weights * F.kl_div(
                F.log_softmax(s_t2i, dim=-1), t_t2i, reduction='none'
            ).sum(dim=-1)).sum()
        else:
            l_t2i = F.kl_div(F.log_softmax(s_t2i, dim=-1), t_t2i, reduction='batchmean')

    return l_i2t + λ_t2i * l_t2i, l_i2t, l_t2i


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
