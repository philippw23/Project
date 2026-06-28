from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskTokenModule_old(nn.Module):
    """Mask tokens cross-attend to ViT patches, then self-attend on the concatenated sequence.

    Forward pass:
        1. Cross-attention:  Q = mask_tokens,  K,V = patch_tokens  → updated_mask [B,N,d]
        2. Self-attention:   seq = concat([patch_tokens, updated_mask]) → [B,m+N,d]
        3. Split:            mask_feats = seq[:, m:, :]              → [B,N,d]
        4. Heatmaps (raw vit_dim space):
               M_proj  = L2_norm(mask_proj(mask_feats))             → [B,N,d]
               P_norm  = L2_norm(patch_tokens)                       → [B,m,d]
               H_logits = M_proj @ P_norm^T / τ                      → [B,N,m]
               H_soft   = softmax(H_logits, dim=-1)                  → [B,N,m]

    No FFN blocks — keeps parameter count low for small datasets.
    vit.patch_proj is NOT used here; it is only needed for L_sim in Stage 3.
    """

    def __init__(
        self,
        n_tokens: int = 16,
        vit_dim: int = 768,
        n_heads: int = 8,
        tau_spatial_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.vit_dim  = vit_dim

        self.mask_tokens = nn.Parameter(torch.randn(1, n_tokens, vit_dim) * 0.02)

        # cross-attention: Q=mask, K,V=patches
        self.cross_attn = nn.MultiheadAttention(vit_dim, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(vit_dim)

        # self-attention on concat([patches, updated_mask])
        self.self_attn = nn.MultiheadAttention(vit_dim, n_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(vit_dim)

        # project mask features for cosine similarity with patches (stays in vit_dim)
        self.mask_proj = nn.Linear(vit_dim, vit_dim, bias=False)

        # learnable spatial temperature
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_spatial_init)))

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: [B, m, vit_dim]  raw ViT patch embeddings

        Returns:
            H_soft:     [B, N, m]  softmax heatmaps per mask token (for L_sim pooling)
            H_logits:   [B, N, m]  raw logits (for L_seg BCE)
            mask_feats: [B, N, vit_dim]  final mask token features (for downstream)
        """
        B, m, _ = patch_tokens.shape
        tokens = self.mask_tokens.expand(B, -1, -1)               # [B, N, d]

        # 1. cross-attention: mask tokens (Q) attend to patches (K, V)
        upd, _ = self.cross_attn(tokens, patch_tokens, patch_tokens, need_weights=False)
        tokens = self.cross_norm(tokens + upd)                     # [B, N, d]

        # 2. concat [patches; updated_mask] → self-attention
        seq = torch.cat([patch_tokens, tokens], dim=1)             # [B, m+N, d]
        upd2, _ = self.self_attn(seq, seq, seq, need_weights=False)
        seq = self.self_norm(seq + upd2)                           # [B, m+N, d]

        # 3. split: last N tokens are final mask features
        mask_feats = seq[:, m:, :]                                 # [B, N, d]

        # 4. heatmap logits via cosine similarity in raw vit_dim space
        M_proj   = F.normalize(self.mask_proj(mask_feats), dim=-1) # [B, N, d]
        P_norm   = F.normalize(patch_tokens, dim=-1)                # [B, m, d]
        tau      = self.log_tau.exp().clamp(min=0.01)
        H_logits = torch.bmm(M_proj, P_norm.transpose(1, 2)) / tau  # [B, N, m]
        H_soft   = F.softmax(H_logits, dim=-1)                       # [B, N, m]

        return H_soft, H_logits, mask_feats


class MaskTokenDecoder(nn.Module):
    """N learnable mask tokens cross-attend to ViT patch tokens (with optional
    Gaussian smoothing on attention weights), then self-attend among themselves.

    Cross-attention uses manual Q/K/V projections so Gaussian smoothing can be
    injected between softmax and value-weighting — enforcing spatially compact,
    contiguous focus regions on the 14×14 patch grid.
    Self-attention uses nn.MultiheadAttention (no spatial structure → no smoothing).

    Returns only the refined tokens [B, N, d]. The mask heatmap for dice loss is
    computed by MaskPredictionHead(tokens, patch_tokens). The tokens can also be
    used directly for L_sim (dot product with text embeddings).
    """

    def __init__(
        self,
        n_tokens: int = 4,
        vit_dim: int = 768,
        n_heads: int = 8,
        sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.vit_dim  = vit_dim
        self.n_heads  = n_heads
        self.d_head   = vit_dim // n_heads

        self.mask_tokens = nn.Parameter(torch.randn(1, n_tokens, vit_dim) * 0.02)

        # Manual cross-attention projections so we can inject Gaussian smoothing.
        self.q_proj   = nn.Linear(vit_dim, vit_dim, bias=False)
        self.k_proj   = nn.Linear(vit_dim, vit_dim, bias=False)
        self.v_proj   = nn.Linear(vit_dim, vit_dim, bias=False)
        self.out_proj = nn.Linear(vit_dim, vit_dim, bias=False)
        self.cross_norm = nn.LayerNorm(vit_dim)

        if sigma > 0:
            kernel = _make_gauss_kernel(sigma)
            self.register_buffer("gauss_kernel", kernel)  # [1, 1, k, k]
            self.gauss_pad = kernel.shape[-1] // 2
        else:
            self.gauss_kernel = None
            self.gauss_pad    = 0

        # Self-attention: tokens [B, N, d] have no spatial layout → no smoothing needed.
        self.self_attn = nn.MultiheadAttention(vit_dim, n_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(vit_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, P, vit_dim]  raw ViT patch embeddings (P=196 for 224px)
        Returns:
            tokens: [B, N, vit_dim]  refined mask token features
        """
        B, P, _ = patch_tokens.shape
        N, H, d_h = self.n_tokens, self.n_heads, self.d_head
        tokens = self.mask_tokens.expand(B, -1, -1)  # [B, N, d]

        # Cross-attention: Q=mask_tokens, K=V=patch_tokens
        Q = self.q_proj(tokens).reshape(B, N, H, d_h).permute(0, 2, 1, 3)        # [B,H,N,d_h]
        K = self.k_proj(patch_tokens).reshape(B, P, H, d_h).permute(0, 2, 1, 3)  # [B,H,P,d_h]
        V = self.v_proj(patch_tokens).reshape(B, P, H, d_h).permute(0, 2, 1, 3)  # [B,H,P,d_h]

        weights = F.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h), dim=-1
        )  # [B, H, N, P]

        if self.gauss_kernel is not None:
            # Smooth each head's N attention maps over the 14×14 patch grid.
            w = weights.reshape(B * H * N, 1, 14, 14)
            w = F.conv2d(w, self.gauss_kernel, padding=self.gauss_pad)
            w = w.reshape(B * H, N, P)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)  # renormalize to sum=1
            weights = w.reshape(B, H, N, P)

        attn_out = torch.matmul(weights, V)                               # [B, H, N, d_h]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, self.vit_dim)
        tokens = self.cross_norm(tokens + self.out_proj(attn_out))        # [B, N, d]

        # Self-attention: tokens refine relative to each other (no spatial structure).
        upd2, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.self_norm(tokens + upd2)                            # [B, N, d]

        return tokens

    def __repr__(self) -> str:
        sigma_str = f"{self.gauss_pad}" if self.gauss_kernel is not None else "0"
        return (
            f"MaskTokenDecoder(n_tokens={self.n_tokens}, vit_dim={self.vit_dim}, "
            f"n_heads={self.n_heads}, sigma_pad={sigma_str})"
        )


def _make_gauss_kernel(sigma: float, max_k: int = 13) -> torch.Tensor:
    """2D Gaussian kernel for depthwise conv — shape [1, 1, k, k]."""
    k = min(2 * round(2.5 * sigma) + 1, max_k)
    k = k if k % 2 == 1 else k + 1          # ensure odd
    c = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-c ** 2 / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]


class MaskPredictionHead(nn.Module):
    """Projects mask tokens and computes cosine similarity with patch embeddings.

    Output mask_logits [B, N, P] are raw logits (no sigmoid) for dice+BCE loss.
    Spatial smoothing is handled upstream by MaskTokenDecoder's Gaussian-smoothed
    cross-attention, so no additional smoothing is applied here.
    """

    def __init__(self, vit_dim: int = 768, tau: float = 1.0) -> None:
        super().__init__()
        self.proj = nn.Linear(vit_dim, vit_dim, bias=False)
        self.tau  = tau

    def forward(
        self,
        tokens: torch.Tensor,       # [B, N, vit_dim]  from MaskTokenDecoder
        patch_tokens: torch.Tensor, # [B, P, vit_dim]
    ) -> torch.Tensor:
        """Returns mask_logits [B, N, P]."""
        t_norm = F.normalize(self.proj(tokens), dim=-1)              # [B, N, d]
        scale  = patch_tokens.shape[-1] ** 0.5 * self.tau
        return torch.bmm(t_norm, patch_tokens.transpose(1, 2)) / scale  # [B, N, P]


def get_fg_idx(
    mask_logits: torch.Tensor,     # [B, N, P]
    gt_patch_labels: torch.Tensor, # [B, P]  float32, 1=lesion 0=background
) -> torch.Tensor:
    """Select foreground token index per batch element (detached — no gradient).

    Training: token with highest sigmoid overlap with GT mask.
    Inference (no GT): call with gt_patch_labels = mask_logits.sigmoid().sum(dim=1).
    """
    with torch.no_grad():
        pred_probs = torch.sigmoid(mask_logits)                          # [B, N, P]
        overlap    = (pred_probs * gt_patch_labels.unsqueeze(1)).sum(-1) # [B, N]
        return overlap.argmax(dim=1)                                      # [B,]
