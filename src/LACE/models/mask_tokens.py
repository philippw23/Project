from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskTokenModule(nn.Module):
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
