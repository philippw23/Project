from __future__ import annotations

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from biomedclip.utils.misc import MODEL_TAG
from LACE.models.lora import inject_lora_vit, count_trainable_params

VIT_DIM   = 768
BERT_DIM  = 768
EMBED_DIM = 256


class ProjectionHead(nn.Module):
    """Linear(in_dim, embed_dim) followed by L2-normalisation."""

    def __init__(self, in_dim: int, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class SharedViT(nn.Module):
    """BiomedCLIP ViT-B/16 trunk with LoRA, used for both full images and lesion crops.

    Exposes three forward modes:
        forward_all(images)     -> (cls [B,768], patches [B,196,768])  single trunk pass
        forward_cls(images)     -> z_img [B,256]                       for L_ITA
        forward_patches(images) -> patches [B,196,768]                 for L_sim / L_ortho
    """

    def __init__(self, lora_layers: int, r: int, alpha: float) -> None:
        super().__init__()
        _model, _, self.preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)
        self.trunk = _model.visual.trunk
        del _model

        inject_lora_vit(self.trunk, lora_layers, r, alpha)

        self.img_proj   = ProjectionHead(VIT_DIM, EMBED_DIM)
        self.patch_proj = ProjectionHead(VIT_DIM, EMBED_DIM)

        n_train = count_trainable_params(self)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"SharedViT: LoRA in last {lora_layers} blocks (r={r}, alpha={alpha}) | "
            f"trainable: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)"
        )

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        """trunk.forward_features -> [B, 197, 768] (CLS at index 0, patches at 1:)."""
        return self.trunk.forward_features(images)

    def forward_all(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single trunk pass returning raw CLS and patch tokens."""
        tokens = self._features(images)
        return tokens[:, 0, :], tokens[:, 1:, :]

    def forward_cls(self, images: torch.Tensor) -> torch.Tensor:
        """Projected + normalised CLS embedding for L_ITA. Returns [B, 256]."""
        cls = self._features(images)[:, 0, :]
        return self.img_proj(cls)

    def forward_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Raw patch tokens (768-dim). Returns [B, 196, 768]."""
        return self._features(images)[:, 1:, :]


class BiomedCLIPTextEncoder(nn.Module):
    """BiomedCLIP PubMedBERT text encoder with trainable projection heads.

    Transformer weights are frozen; cls_proj and word_proj are trained.
    self.tokenizer is exposed for data loading.

    encode_beurteilung  →  CLS projection for L_ITA
    encode_befund       →  (CLS, word projections) for L_sim phrase attention
    """

    def __init__(self) -> None:
        super().__init__()
        _model, _, _ = open_clip.create_model_and_transforms(MODEL_TAG)
        self.transformer = _model.text.transformer
        self.tokenizer   = open_clip.get_tokenizer(MODEL_TAG)
        del _model
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        self.cls_proj  = ProjectionHead(BERT_DIM, EMBED_DIM)
        self.word_proj = ProjectionHead(BERT_DIM, EMBED_DIM)

        n_bert = sum(p.numel() for p in self.transformer.parameters())
        n_proj = (sum(p.numel() for p in self.cls_proj.parameters()) +
                  sum(p.numel() for p in self.word_proj.parameters()))
        print(
            f"BiomedCLIPTextEncoder: frozen transformer ({n_bert:,} params) + "
            f"trainable projections ({n_proj:,} params)"
        )

    @torch.no_grad()
    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # [B, L, 768]

    def encode_beurteilung(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Projected CLS of Beurteilung text for L_ITA. Returns [B, 256]."""
        seq = self._forward(input_ids, attention_mask)
        return self.cls_proj(seq[:, 0, :])

    def encode_befund(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CLS + word-token projections of Befund text for L_sim.

        Returns (z_cls [B, 256], proj_words [B, L, 256]).
        z_cls is kept for API compatibility; sim_loss_v2 uses proj_words only.
        """
        seq = self._forward(input_ids, attention_mask)
        B, L, D = seq.shape
        proj_words = self.word_proj(seq.reshape(-1, D)).reshape(B, L, EMBED_DIM)
        return self.cls_proj(seq[:, 0, :]), proj_words
