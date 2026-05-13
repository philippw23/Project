from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ita_loss(
    z_img: torch.Tensor,
    z_text: torch.Tensor,
    logit_scale: torch.Tensor,
    soft_label_temp: float = 0.07,
) -> torch.Tensor:
    """Global image-text alignment with MedCLIP-style soft labels.

    Instead of one-hot diagonal targets, pairwise text cosine similarities are
    used as soft targets so the model is not penalised for retrieving semantically
    similar reports. Both z_img and z_text must already be L2-normalised.

    Args:
        z_img:          [B, D] L2-normalised image embeddings
        z_text:         [B, D] L2-normalised text (beurteilung) embeddings
        logit_scale:    scalar parameter; exp() gives the contrastive temperature
        soft_label_temp: temperature for the soft label distribution
    """
    scale = logit_scale.exp().clamp(max=100.0)

    logits_i2t = z_img  @ z_text.t() * scale
    logits_t2i = z_text @ z_img.t()  * scale

    with torch.no_grad():
        sim_text  = z_text @ z_text.t() / soft_label_temp
        sim_image = z_img  @ z_img.t()  / soft_label_temp
        targets_i2t = F.softmax(sim_text,  dim=-1)
        targets_t2i = F.softmax(sim_image, dim=-1)

    loss_i2t = -(targets_i2t * F.log_softmax(logits_i2t, dim=-1)).sum(dim=-1).mean()
    loss_t2i = -(targets_t2i * F.log_softmax(logits_t2i, dim=-1)).sum(dim=-1).mean()
    return (loss_i2t + loss_t2i) / 2.0


def sim_loss(
    proj_patches: torch.Tensor,
    proj_words: torch.Tensor,
    z_befund_cls: torch.Tensor,
    logit_scale: torch.Tensor,
    phrase_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Local lesion-phrase alignment using GLoRIA-style attention pooling.

    Steps:
    1. A = softmax(proj_patches @ proj_words^T / sqrt(D), dim=1)   [B, N, L]
       Normalises over patches: each word/phrase gets a prob distribution over patches.
    2. context = A^T @ proj_patches   [B, L, D]
       Word/phrase-conditioned lesion representations.
    3. v_lesion = normalize(context.mean(dim=1))   [B, D]
       If phrase_mask is given, only real (non-padding) slots contribute to the mean.
    4. Symmetric InfoNCE between v_lesion and z_befund_cls.

    All inputs must be L2-normalised (output of ProjectionHead).

    Args:
        proj_patches:   [B, N, D] projected patch embeddings (lesion crop)
        proj_words:     [B, L, D] projected word token or phrase embeddings (befund)
        z_befund_cls:   [B, D]    projected CLS embedding of befund
        logit_scale:    scalar contrastive temperature parameter
        phrase_mask:    [B, L] bool — True = real slot, False = padding (phrase modes only)
    """
    B, N, D = proj_patches.shape

    attn = torch.bmm(proj_patches, proj_words.transpose(1, 2)) / math.sqrt(D)
    attn = F.softmax(attn, dim=1)

    context = torch.bmm(attn.transpose(1, 2), proj_patches)  # [B, L, D]

    if phrase_mask is not None:
        context  = context * phrase_mask.unsqueeze(-1).float()
        n_real   = phrase_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        v_lesion = F.normalize(context.sum(dim=1) / n_real, dim=-1)
    else:
        v_lesion = F.normalize(context.mean(dim=1), dim=-1)

    scale      = logit_scale.exp().clamp(max=100.0)
    logits_il  = v_lesion    @ z_befund_cls.t() * scale
    logits_li  = z_befund_cls @ v_lesion.t()    * scale
    labels     = torch.arange(B, device=v_lesion.device)

    return (F.cross_entropy(logits_il, labels) + F.cross_entropy(logits_li, labels)) / 2.0


def sim_loss_v2(
    proj_patches: torch.Tensor,
    proj_words: torch.Tensor,
    H_soft: torch.Tensor,
    logit_scale: torch.Tensor,
    phrase_temp: float = 0.07,
) -> torch.Tensor:
    """Local lesion-phrase alignment via heatmap pooling and phrase attention.

    Steps (following diagram formulas exactly):
      1. H_mean = H_soft.mean(dim=1)                               [B, m]
         (each H_i is a softmax dist → H_mean also sums to 1)
      2. v_local = normalize(H_mean @ proj_patches)                [B, D]
         lesion-weighted patch pool
      3. α_j = softmax(sim(v_local, t_j) / τ_s)                   [B, J]
         v_local attends over phrase tokens
      4. t̂ = normalize(Σ α_j * t_j)                               [B, D]
         phrase-attended text representation
      5. s_{ik} = sim(v_local_i, t̂_k) * τ                        [B, B]
      6. w_{ik} = softmax(t̂ · t̂^T / τ_s)   (soft semantic targets)
      7. L_sim = ½(L_{i→t} + L_{t→i})  with soft InfoNCE

    Args:
        proj_patches:  [B, m, D]  L2-normalised patch embeddings (vit.patch_proj)
        proj_words:    [B, J, D]  L2-normalised word token embeddings (befund)
        H_soft:        [B, N, m]  softmax heatmaps from MaskTokenModule
        logit_scale:   scalar     contrastive temperature τ
        phrase_temp:   float      τ_s for phrase attention + soft label targets
    """
    # 1–2. lesion-weighted patch pool
    H_mean  = H_soft.mean(dim=1)                                    # [B, m]
    v_local = F.normalize(
        (H_mean.unsqueeze(1) @ proj_patches).squeeze(1), dim=-1
    )                                                                # [B, D]

    # 3–4. v_local-guided phrase attention → t̂
    attn_logits = torch.bmm(
        v_local.unsqueeze(1), proj_words.transpose(1, 2)
    ).squeeze(1) / phrase_temp                                       # [B, J]
    alpha = F.softmax(attn_logits, dim=-1)                           # [B, J]
    t_hat = F.normalize(
        (alpha.unsqueeze(1) @ proj_words).squeeze(1), dim=-1
    )                                                                # [B, D]

    # 5. batch similarity matrix
    tau = logit_scale.exp().clamp(max=100.0)
    s   = v_local @ t_hat.t() * tau                                  # [B, B]

    # 6. soft semantic targets from t̂-t̂ similarity
    with torch.no_grad():
        r = t_hat @ t_hat.t() / phrase_temp                         # [B, B]
        w = F.softmax(r, dim=-1)                                     # [B, B]

    # 7. soft InfoNCE (image→text and text→image)
    L_i2t = -(w * F.log_softmax(s,     dim=-1)).sum(-1).mean()
    L_t2i = -(w * F.log_softmax(s.t(), dim=-1)).sum(-1).mean()
    return (L_i2t + L_t2i) / 2.0


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
