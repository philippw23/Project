from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from LACE.models.encoders import SharedViT
from LACE.models.mask_tokens import MaskTokenDecoder, MaskPredictionHead, MaskTokenModule_old as MaskTokenModule
from LACE.loss.objectives import select_fg_token

class LACEv2Classifier(nn.Module):
    """Downstream malignancy classifier: frozen ViT + frozen mask decoder → MLP.

    visual_mode controls the visual representation:
        "cls"    → vit.img_proj(cls_raw)                                  [B, 512]
        "fg"     → heatmap-weighted avg of vit.patch_proj(patches)        [B, 512]
        "cls_fg" → cat([img_proj(cls_raw), fg_repr])                      [B, 1024]

    The fg representation uses the fg mask token's heatmap (from MaskPredictionHead)
    as attention weights over projected patch features — consistent with L_sim
    during pretraining.

    ViT, mask_decoder, and mask_head are frozen. Only age_emb, sex_emb, and the
    MLP head are trained.
    """

    def __init__(
        self,
        vit: SharedViT,
        mask_decoder: MaskTokenDecoder,
        mask_head: MaskPredictionHead,
        n_meta_dim: int = 32,
        n_classes: int = 3,
        use_meta: bool = True,
        linear_head: bool = False,
        visual_mode: str = "cls_fg",   # "cls" | "fg" | "cls_fg"
        sim_attn_tau: float = 0.07,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vit          = vit
        self.mask_decoder = mask_decoder
        self.mask_head    = mask_head
        self.visual_mode  = visual_mode
        self.use_meta     = use_meta
        self.sim_attn_tau = sim_attn_tau

        for p in self.vit.parameters():
            p.requires_grad_(False)
        for p in self.mask_decoder.parameters():
            p.requires_grad_(False)
        for p in self.mask_head.parameters():
            p.requires_grad_(False)

        vit_emb_dim = vit.proj_dim   # 512
        feat_dim    = vit_emb_dim * 2 if visual_mode == "cls_fg" else vit_emb_dim

        if use_meta:
            self.age_emb = nn.Linear(1, n_meta_dim // 2)
            self.sex_emb = nn.Embedding(2, n_meta_dim // 2)
            feat_dim    += n_meta_dim
        else:
            self.age_emb = None
            self.sex_emb = None

        if linear_head:
            self.head = nn.Linear(feat_dim, n_classes)
        else:
            # Previous fixed head (LayerNorm + single 256-unit GELU block):
            # self.head = nn.Sequential(
            #     nn.LayerNorm(feat_dim),
            #     nn.Linear(feat_dim, 256),
            #     nn.GELU(),
            #     nn.Dropout(0.1),
            #     nn.Linear(256, n_classes),
            # )
            # Now matches MalignancyMLP (biomedclip/chexfound/gloria/imagenet_img):
            # a configurable stack of Linear -> ReLU -> Dropout per hidden_dims.
            layers: list[nn.Module] = []
            in_dim = feat_dim
            for h in (hidden_dims or [256]):
                layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = h
            layers.append(nn.Linear(in_dim, n_classes))
            self.head = nn.Sequential(*layers)

    def _get_visual(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cls_raw, patch_feat = self.vit.forward_all(images)      # [B,768], [B,196,768]
            tokens      = self.mask_decoder(patch_feat)              # [B, N, 768]
            mask_logits = self.mask_head(tokens, patch_feat)         # [B, N, 196]
            fg_idx      = select_fg_token(mask_logits.float())       # [B,]
            B           = images.shape[0]
            fg_logits   = mask_logits[torch.arange(B), fg_idx]       # [B, 196]
            w = F.softmax(fg_logits.float() / self.sim_attn_tau, dim=-1)       # [B, 196]
            P_proj = F.normalize(self.vit.patch_proj(patch_feat), dim=-1)      # [B, 196, D]
            m_fg   = F.normalize((w.unsqueeze(-1) * P_proj).sum(1), dim=-1)    # [B, D]

        if self.visual_mode == "cls":
            return self.vit.img_proj(cls_raw)                        # [B, 512]
        elif self.visual_mode == "fg":
            return m_fg                                               # [B, 512]
        else:  # cls_fg
            return torch.cat([self.vit.img_proj(cls_raw), m_fg], dim=-1)  # [B, 1024]

    def forward(
        self,
        images: torch.Tensor,  # [B, 3, 224, 224]
        age: torch.Tensor,     # [B, 1]  float, normalised
        sex: torch.Tensor,     # [B,]    long, 0=female / 1=male
    ) -> torch.Tensor:
        visual = self._get_visual(images)
        if self.use_meta:
            age_feat = self.age_emb(age.float())
            sex_feat = self.sex_emb(sex.long())
            x = torch.cat([visual, age_feat, sex_feat], dim=-1)
        else:
            x = visual
        return self.head(x)
