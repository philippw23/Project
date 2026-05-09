from __future__ import annotations

import torch
import torch.nn as nn

from LACE.models.encoders import SharedViT
from LACE.models.mask_tokens import MaskTokenModule


class LACEv2Classifier(nn.Module):
    """Downstream malignancy classifier built on frozen pretrained ViT + MaskTokenModule.

    Forward pass:
        1. ViT (frozen) → raw patch tokens [B, 196, 768]
        2. MaskTokenModule (frozen) → mask_feats [B, N, 768]
        3. Mean-pool mask_feats → lesion_repr [B, 768]
        4. Encode age (normalised scalar) and sex (binary) → metric_emb [B, n_meta_dim]
        5. Concat [lesion_repr; metric_emb] → MLP head → classification logit

    Only the metric embedding layers and the MLP head are trained.
    Age should be normalised to [0, 1] before passing.
    Sex should be an integer tensor with 0=female, 1=male.
    """

    def __init__(
        self,
        mask_module: MaskTokenModule,
        vit: SharedViT,
        n_meta_dim: int = 32,
        n_classes: int = 1,
    ) -> None:
        super().__init__()
        self.vit         = vit
        self.mask_module = mask_module

        for p in self.vit.parameters():
            p.requires_grad_(False)
        for p in self.mask_module.parameters():
            p.requires_grad_(False)

        vit_dim = mask_module.vit_dim

        self.age_emb = nn.Linear(1, n_meta_dim // 2)
        self.sex_emb = nn.Embedding(2, n_meta_dim // 2)

        feat_dim = vit_dim + n_meta_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        age: torch.Tensor,
        sex: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224]
            age:    [B, 1]  float, normalised to [0, 1]
            sex:    [B]     long, 0=female / 1=male

        Returns:
            logits: [B, n_classes]
        """
        with torch.no_grad():
            _, patch_feat = self.vit.forward_all(images)        # [B, 196, 768]
            _, _, mask_feats = self.mask_module(patch_feat)      # [B, N, 768]

        lesion_repr = mask_feats.mean(dim=1)                     # [B, 768]

        age_feat = self.age_emb(age.float())                     # [B, n_meta//2]
        sex_feat = self.sex_emb(sex.long())                      # [B, n_meta//2]
        metric_emb = torch.cat([age_feat, sex_feat], dim=-1)     # [B, n_meta]

        x = torch.cat([lesion_repr, metric_emb], dim=-1)         # [B, vit_dim + n_meta]
        return self.head(x)                                       # [B, n_classes]
