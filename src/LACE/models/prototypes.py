from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from LACE.models.encoders import ProjectionHead


class PrototypeBank(nn.Module):
    """LGDEA-style diagnostic-evidence prototype space.

    A fixed set of K learnable prototypes µ_k live in the shared embed_dim space.
    Befund phrase embeddings (already in embed_dim) and mask-decoder tokens
    (projected from vit_dim via φ) are both soft-assigned to the prototypes with a
    shared temperature τ_proto, producing K-dim distributions used by the evidence
    losses (see LACE.loss.objectives.evidence_prototype_loss).

    Prototypes are unnormalised and initialised N(0, 0.02); their norm is controlled
    only by the ‖µ_k‖² shrinkage term inside L_rec (LGDEA Eq. 5).
    """

    def __init__(
        self,
        n_prototypes: int = 32,
        dim: int = 512,
        vit_dim: int = 768,
        tau: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_prototypes = n_prototypes
        self.dim          = dim
        self.tau          = tau

        self.prototypes = nn.Parameter(torch.randn(n_prototypes, dim) * 0.02)
        # φ: maps raw mask-token features (vit_dim) into the prototype space and
        # L2-normalises, matching the normalised befund phrase embeddings.
        self.img_proj = ProjectionHead(vit_dim, dim)

    def assign_text(self, z: torch.Tensor) -> torch.Tensor:
        """Soft-assign normalised phrase embeddings to prototypes.

        Args:
            z: [N, dim]  L2-normalised phrase embeddings
        Returns:
            [N, K]  row-stochastic assignment p(k | z_n)  (LGDEA Eq. 4)
        """
        return F.softmax(z @ self.prototypes.T / self.tau, dim=-1)

    def assign_image(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project mask tokens through φ and soft-assign to prototypes.

        Args:
            tokens: [B, N_tok, vit_dim]  raw mask-decoder token features
        Returns:
            [B, N_tok, K]  Q_I(ℓ, ·)  (LGDEA Eq. 6)
        """
        v = self.img_proj(tokens)                                   # [B, N_tok, dim]
        return F.softmax(v @ self.prototypes.T / self.tau, dim=-1)  # [B, N_tok, K]

    def reconstruct(self, p: torch.Tensor) -> torch.Tensor:
        """Reconstruct embeddings from an assignment: Σ_k p(k) µ_k. Returns [N, dim]."""
        return p @ self.prototypes
