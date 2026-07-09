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


def evidence_prototype_loss(
    phrase_embeddings: torch.Tensor,
    phrase_mask: torch.Tensor,
    phrase_to_image: torch.Tensor,
    mask_tokens: torch.Tensor,
    bank,
    evid_valid: torch.Tensor,
    compute_rec: bool = True,
    compute_evid_p: bool = True,
    lambda_mu: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LGDEA diagnostic-evidence prototype loss (prototype-space-only scope).

    Two paper-paired terms (no cross-image negatives, no graph propagation):
      L_rec    (Eq. 5): reconstruct befund phrase embeddings from the prototype
                        space + ‖µ_k‖² shrinkage. Learns µ_k (joint: gradient also
                        flows into the phrase embeddings / cls_proj).
      L_evid_p (Eq. 9): per-pair KL(Q̄_R ‖ Q̄_I) aligning the report-induced prototype
                        distribution (teacher, detached) to the image one.

    Curriculum: caller passes compute_rec=True in stage 1 and additionally
    compute_evid_p=True in stage 2.

    Args:
        phrase_embeddings: [B, J, D]  L2-normalised befund phrase embeddings (padded)
        phrase_mask:       [B, J]     True = valid phrase
        phrase_to_image:   [N_total]  source image index (0..B-1) per valid phrase
        mask_tokens:       [B, N_tok, vit_dim]  raw MaskTokenDecoder tokens
        bank:              PrototypeBank
        evid_valid:        [B] bool   images eligible for L_evid_p (has_mask & has_befund)
        compute_rec:       bool       compute L_rec
        compute_evid_p:    bool       compute L_evid_p
        lambda_mu:         float      weight of the ‖µ_k‖² shrinkage term in L_rec
    Returns:
        (l_rec, l_evid_p, usage_entropy) — usage_entropy is the detached entropy of the
        batch-mean phrase assignment (higher = more prototypes in use; low = collapse).
    """
    B = phrase_mask.shape[0]
    K = bank.n_prototypes
    zero = mask_tokens.sum() * 0.0

    valid_phrases = phrase_embeddings[phrase_mask]          # [N_total, D]
    if valid_phrases.shape[0] == 0:
        return zero, zero, zero

    p_text = bank.assign_text(valid_phrases)               # [N_total, K]

    # L_rec — reconstruct phrase embeddings + prototype-norm shrinkage (Eq. 5)
    l_rec = zero
    if compute_rec:
        recon = bank.reconstruct(p_text)                   # [N_total, D]
        rec_err = ((valid_phrases - recon) ** 2).sum(dim=-1).mean()
        mu_reg  = (bank.prototypes ** 2).sum(dim=-1).mean()
        l_rec   = rec_err + lambda_mu * mu_reg

    # Usage entropy of the batch-mean assignment (collapse monitor)
    with torch.no_grad():
        pbar = p_text.mean(dim=0)                          # [K]
        usage_entropy = -(pbar * (pbar + 1e-8).log()).sum()

    # L_evid_p — per-pair KL(Q̄_R ‖ Q̄_I) over images with both a mask and befund (Eq. 7-9)
    l_evid_p = zero
    if compute_evid_p and evid_valid.any():
        # Q̄_R: mean phrase assignment per image
        qbar_r = torch.zeros(B, K, device=p_text.device, dtype=p_text.dtype)
        qbar_r.index_add_(0, phrase_to_image, p_text)
        n_phrase = torch.bincount(phrase_to_image, minlength=B).clamp(min=1).unsqueeze(1).float()
        qbar_r = qbar_r / n_phrase                         # [B, K]

        # Q̄_I: mean prototype assignment over the image's mask tokens
        q_img  = bank.assign_image(mask_tokens)            # [B, N_tok, K]
        qbar_i = q_img.mean(dim=1)                         # [B, K]

        qr = qbar_r[evid_valid].detach()                   # teacher
        qi = qbar_i[evid_valid]
        l_evid_p = F.kl_div((qi + 1e-8).log(), qr, reduction="batchmean")

    return l_rec, l_evid_p, usage_entropy.detach()


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
                # Aggregate phrase similarities per image: t[n, k] = mean of phrase_n·phrase_j for j in image k
                pairwise = valid_phrases @ valid_phrases.T / τ_s             # [N_total, N_total]
                t_t2i = torch.zeros(valid_phrases.shape[0], B_sim,
                                    device=valid_phrases.device, dtype=valid_phrases.dtype)
                for k in range(B_sim):
                    t_t2i[:, k] = pairwise[:, phrase_to_image == k].mean(dim=-1)
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


def select_fg_token(mask_logits: torch.Tensor) -> torch.Tensor:
    """Select the most spatially-concentrated mask token per sample (no GT needed).

    Uses peak-to-mean ratio: a token focused on a small lesion area will have a
    high max activation relative to its mean, whereas a token that fires uniformly
    across background patches will have a ratio near 1.  Selects the token with
    the highest ratio, consistently at both training time and inference.

    Args:
        mask_logits: [B, N, P]
    Returns:
        fg_idx: [B,] int64
    """
    with torch.no_grad():
        acts = torch.sigmoid(mask_logits.float())           # [B, N, P]
        peak = acts.amax(dim=-1)                            # [B, N]
        mean = acts.mean(dim=-1)                            # [B, N]
        concentration = peak / (mean + 1e-8)               # [B, N]
        return concentration.argmax(dim=1)                  # [B,]


def compute_l_dice_ce(
    mask_logits: torch.Tensor,      # [B, N, P]  P=196 (14×14)
    gt_patch_labels: torch.Tensor,  # [B, P]  float32, soft coverage fractions 0.0–1.0
    hard_neg_k: int = 10,           # top-k background patches to penalise explicitly
    hard_neg_weight: float = 0.5,   # weight of the hard-negative BCE term
) -> torch.Tensor:
    """Dice + global-BCE + hard-negative segmentation loss.

    Token selection: argmax of per-token sigmoid overlap with gt_patch_labels
    (detached). This gives stable gradient signal — the token that partially
    covers the lesion consistently receives gradients to cover it better, rather
    than the selection jumping between tokens across batches.

    gt_patch_labels are soft coverage fractions (fraction of each 16×16 patch
    covered by the segmentation mask). Samples with all-zero GT are excluded.

    Loss structure (all terms on the GT-aligned fg token):
      - Dice: ratio-based overlap, shapes the prediction globally.
      - Global BCE: dense gradient over all 196 patches for stable learning.
      - Hard-negative BCE: additionally penalises the top-hard_neg_k most-activated
        background patches (GT == 0). Targets leakage onto adjacent bone that global
        BCE alone cannot suppress because easy correct patches dilute the gradient.

    Returns (total, dice_detached, bce_detached, hard_neg_detached). Returns 0.0 if B==0.
    """
    B, N, P = mask_logits.shape
    if B == 0:
        return (mask_logits.sum() * 0.0,) * 4

    logits = mask_logits.float()
    gt     = gt_patch_labels.float()

    with torch.no_grad():
        overlap = (torch.sigmoid(logits) * gt.unsqueeze(1)).sum(-1)  # [B, N]
        fg_idx  = overlap.argmax(dim=1)                               # [B,]
    fg_logits = logits[torch.arange(B, device=logits.device), fg_idx]  # [B, P]

    nonempty = gt.sum(dim=-1) > 0
    if nonempty.any():
        fl = fg_logits[nonempty]   # [B_pos, P]
        gp = gt[nonempty]          # [B_pos, P]
        fp = torch.sigmoid(fl)

        # Dice
        inter = (fp * gp).sum(-1)
        dice  = 1.0 - (2.0 * inter / (fp.sum(-1) + gp.sum(-1) + 1e-6)).mean()

        # Global BCE — dense gradient over all patches for stable learning
        bce = F.binary_cross_entropy_with_logits(fl, gp)

        # Hard-negative mining — extra penalty on top-k most-activated background patches
        hard_neg    = logits.sum() * 0.0
        lesion_mask = gp > 0.0
        if hard_neg_k > 0:
            bg_logits = fl.masked_fill(lesion_mask, float("-inf"))
            k = min(hard_neg_k, int((~lesion_mask).sum(dim=-1).min().item()))
            if k > 0:
                topk_logits, _ = bg_logits.topk(k, dim=-1)         # [B_pos, k]
                hard_neg = F.binary_cross_entropy_with_logits(
                    topk_logits, torch.zeros_like(topk_logits),
                )
    else:
        dice     = logits.sum() * 0.0
        bce      = logits.sum() * 0.0
        hard_neg = logits.sum() * 0.0

    total = dice + bce + hard_neg_weight * hard_neg
    return total, dice.detach(), bce.detach(), hard_neg.detach()

def ortho_loss_old(
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


def ortho_loss(
    patch_tokens: torch.Tensor,
    patch_labels: torch.Tensor,
) -> torch.Tensor:
    """Lesion-background orthogonality regularizer with soft patch weighting.

    Each patch contributes to both embeddings proportionally to its coverage
    fraction: a patch that is 20% lesion contributes 20% to the lesion
    embedding and 80% to the background embedding. Samples where all weight
    falls on one side are excluded.

    Operates in the raw 768-dim ViT feature space (not the projected space).

    Args:
        patch_tokens:  [B, M, d_vit] raw patch token embeddings
        patch_labels:  [B, M] float32, soft coverage fractions 0.0–1.0
    """
    w_les = patch_labels.unsqueeze(-1)          # [B, M, 1]
    w_bg  = (1.0 - patch_labels).unsqueeze(-1)  # [B, M, 1]

    sum_les = w_les.sum(dim=1)  # [B, 1]
    sum_bg  = w_bg.sum(dim=1)   # [B, 1]

    valid = (sum_les.squeeze(-1) > 0) & (sum_bg.squeeze(-1) > 0)
    if not valid.any():
        return patch_tokens.sum() * 0.0

    pt       = patch_tokens[valid]
    v_lesion = (pt * w_les[valid]).sum(dim=1) / sum_les[valid].clamp(min=1e-6)
    v_bg     = (pt * w_bg[valid]).sum(dim=1)  / sum_bg[valid].clamp(min=1e-6)

    # +1 shifts range from [-1,1] to [0,2] so loss stays non-negative.
    return (F.cosine_similarity(v_lesion, v_bg, dim=-1) + 1.0).mean()
