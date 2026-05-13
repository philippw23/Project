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

    def __init__(self, lora_layers: int, r: int, alpha: float, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        _model, _, self.preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)
        self.trunk = _model.visual.trunk
        del _model

        inject_lora_vit(self.trunk, lora_layers, r, alpha)

        self.proj_dim   = embed_dim
        self.img_proj   = ProjectionHead(VIT_DIM, embed_dim)
        self.patch_proj = ProjectionHead(VIT_DIM, embed_dim)

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


class PhraseAttentionPool(nn.Module):
    """Learned-query attention pooling over N phrase CLS embeddings.

    A single trainable query vector scores each phrase; softmax weights are
    used to compute a weighted sum. Padding slots are masked to -inf before
    softmax so they contribute zero to the output.
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(embed_dim))
        nn.init.normal_(self.query, std=0.02)

    def forward(
        self,
        phrase_embs: torch.Tensor,
        phrase_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            phrase_embs: [B, N, D] L2-normalised phrase embeddings
            phrase_mask: [B, N]    True = real phrase, False = padding
        Returns:
            [B, D] L2-normalised pooled embedding
        """
        attn = phrase_embs @ self.query                        # [B, N]
        attn = attn.masked_fill(~phrase_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)                         # [B, N]
        out  = (attn.unsqueeze(-1) * phrase_embs).sum(dim=1)   # [B, D]
        return F.normalize(out, dim=-1)


class BiomedCLIPTextEncoder(nn.Module):
    """BiomedCLIP PubMedBERT text encoder with trainable projection heads.

    Transformer weights are frozen; cls_proj, word_proj, and phrase_pool are trained.
    self.tokenizer is exposed for data loading.

    encode_beurteilung        →  CLS projection for L_ITA  (full / concat modes)
    encode_beurteilung_phrases→  pooled phrase CLS for L_ITA  (phrase modes)
    encode_befund             →  (CLS, word projections) for L_sim  (full / concat)
    encode_befund_phrases     →  (mean CLS, phrase embeddings) for L_sim  (phrase modes)
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        _model, _, _ = open_clip.create_model_and_transforms(MODEL_TAG)
        self.transformer = _model.text.transformer
        self.tokenizer   = open_clip.get_tokenizer(MODEL_TAG)
        del _model
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        self.cls_proj   = ProjectionHead(BERT_DIM, embed_dim)
        self.word_proj  = ProjectionHead(BERT_DIM, embed_dim)
        self.phrase_pool = PhraseAttentionPool(embed_dim)

        n_bert = sum(p.numel() for p in self.transformer.parameters())
        n_proj = (sum(p.numel() for p in self.cls_proj.parameters()) +
                  sum(p.numel() for p in self.word_proj.parameters()) +
                  sum(p.numel() for p in self.phrase_pool.parameters()))
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

    def _encode_phrase_batch(
        self,
        phrase_ids: torch.Tensor,
        phrase_attn: torch.Tensor,
        phrase_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run all phrases through the frozen transformer and project CLS tokens.

        Args:
            phrase_ids:  [B, N, T] token ids (padding slots have all-zero ids)
            phrase_attn: [B, N, T] attention masks
            phrase_mask: [B, N]    True = real phrase, False = padding slot
        Returns:
            [B, N, embed_dim] L2-normalised phrase embeddings; padding slots zeroed.
        """
        B, N, T = phrase_ids.shape
        hidden = self._forward(
            phrase_ids.reshape(B * N, T),
            phrase_attn.reshape(B * N, T),
        )                                                    # [B*N, T, 768]
        cls  = hidden[:, 0, :]                               # [B*N, 768]
        embs = self.cls_proj(cls).reshape(B, N, -1)          # [B, N, embed_dim]
        embs = embs * phrase_mask.float().unsqueeze(-1)       # zero padding slots
        return embs

    def encode_beurteilung(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Projected CLS of Beurteilung text for L_ITA. Returns [B, 256]."""
        seq = self._forward(input_ids, attention_mask)
        return self.cls_proj(seq[:, 0, :])

    def encode_beurteilung_phrases(
        self,
        phrase_ids: torch.Tensor,
        phrase_attn: torch.Tensor,
        phrase_mask: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Pool beurteilung phrase CLS embeddings into a single vector for L_ITA.

        Args:
            phrase_ids:  [B, N, T]
            phrase_attn: [B, N, T]
            phrase_mask: [B, N]    True = real phrase
            mode:        "phrase_mean" or "phrase_attn"
        Returns:
            [B, embed_dim] L2-normalised embedding
        """
        embs = self._encode_phrase_batch(phrase_ids, phrase_attn, phrase_mask)
        if mode == "phrase_attn":
            return self.phrase_pool(embs, phrase_mask)
        # phrase_mean
        n_real = phrase_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        return F.normalize(embs.sum(dim=1) / n_real, dim=-1)

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

    def encode_befund_phrases(
        self,
        phrase_ids: torch.Tensor,
        phrase_attn: torch.Tensor,
        phrase_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode befund phrases for L_sim.

        Returns:
            z_cls  [B, embed_dim]    mean-pooled phrase CLS (InfoNCE anchor)
            embs   [B, N, embed_dim] per-phrase embeddings (feeds patch attention)
        """
        embs  = self._encode_phrase_batch(phrase_ids, phrase_attn, phrase_mask)
        n_real = phrase_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        z_cls  = F.normalize(embs.sum(dim=1) / n_real, dim=-1)
        return z_cls, embs
