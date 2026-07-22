from __future__ import annotations

import math
import types

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from biomedclip.utils.misc import MODEL_TAG
from LACE.models.lora import inject_lora_vit, unfreeze_last_n_vit, count_trainable_params


def enable_dynamic_img_size(trunk: nn.Module) -> None:
    """Allow a timm ViT trunk to accept images at any resolution.

    Disables PatchEmbed's strict size assertion and replaces _pos_embed with a
    version that bicubic-interpolates positional embeddings when the patch count
    differs from the pretrained size.  At native resolution the behaviour is
    identical to the original — no overhead or accuracy difference.
    """
    trunk.patch_embed.strict_img_size = False

    def _pos_embed_dynamic(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        pos_embed = self.pos_embed                           # [1, n_prefix+src_N, D]
        n_prefix  = 0 if self.no_embed_class else 1         # CLS occupies a pos slot
        src_N     = pos_embed.shape[1] - n_prefix

        if N != src_N:
            src_grid  = int(math.isqrt(src_N))
            dst_grid  = int(math.isqrt(N))
            prefix_pe = pos_embed[:, :n_prefix]
            patch_pe  = pos_embed[:, n_prefix:].reshape(1, src_grid, src_grid, D).permute(0, 3, 1, 2)
            patch_pe  = F.interpolate(patch_pe, (dst_grid, dst_grid), mode="bicubic", align_corners=False)
            patch_pe  = patch_pe.permute(0, 2, 3, 1).reshape(1, N, D)
            pos_embed = torch.cat([prefix_pe, patch_pe], dim=1)

        cls_token = self.cls_token.expand(B, -1, -1)
        if self.no_embed_class:
            x = x + pos_embed
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    trunk._pos_embed = types.MethodType(_pos_embed_dynamic, trunk)

VIT_DIM            = 768   # BiomedCLIP ViT-B/16
CHEXFOUND_VIT_DIM  = 1024  # CheXFound  ViT-L/16
BERT_DIM  = 768
EMBED_DIM = 512


class ProjectionHead(nn.Module):
    """Projection head followed by L2-normalisation.

    hidden_dim=None → single Linear; hidden_dim=k → Linear→GELU→Linear (2-layer MLP).
    """

    def __init__(
        self, in_dim: int, embed_dim: int = EMBED_DIM, hidden_dim: int | None = None
    ) -> None:
        super().__init__()
        if hidden_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim, bias=False),
            )
        else:
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

    def __init__(
        self,
        lora_layers: int,
        r: int,
        alpha: float,
        embed_dim: int = EMBED_DIM,
        unfreeze_layers: int = 0,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        _model, _, self.preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)
        self.trunk = _model.visual.trunk
        del _model

        enable_dynamic_img_size(self.trunk)
        if unfreeze_layers > 0:
            unfreeze_last_n_vit(self.trunk, unfreeze_layers)
            mode_str = f"full fine-tune of last {unfreeze_layers} blocks (+ final norm)"
        else:
            inject_lora_vit(self.trunk, lora_layers, r, alpha)
            mode_str = f"LoRA in last {lora_layers} blocks (r={r}, alpha={alpha})"

        self.proj_dim   = embed_dim
        self.img_proj   = ProjectionHead(VIT_DIM, embed_dim)
        self.patch_proj = ProjectionHead(VIT_DIM, embed_dim)

        # Describes the adapter topology only; callers that freeze the trunk after
        # construction must report their own trainable count (see verbose).
        self.adapter_mode = mode_str

        if verbose:
            print(f"SharedViT: {mode_str} | {self.param_summary()}")

    def param_summary(self) -> str:
        """Trainable/total parameter counts, evaluated at call time."""
        n_train = count_trainable_params(self)
        n_total = sum(p.numel() for p in self.parameters())
        return f"trainable: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)"

    def load_pretrained_projections(self) -> None:
        """Copy img_proj and patch_proj weights from BiomedCLIP's pretrained visual.head.proj."""
        _model, _, _ = open_clip.create_model_and_transforms(MODEL_TAG)
        src = _model.visual.head.proj
        self.img_proj.proj.weight.data.copy_(src.weight.data)
        self.patch_proj.proj.weight.data.copy_(src.weight.data)
        del _model
        print(f"SharedViT: loaded pretrained img_proj and patch_proj from BiomedCLIP visual.head.proj {tuple(src.weight.shape)}")

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

    @property
    def vit_dim(self) -> int:
        return VIT_DIM


class CheXFoundSharedViT(nn.Module):
    """CheXFound ViT-L/16 trunk with LoRA, matching SharedViT's interface.

    Wraps CheXFoundViT and exposes the same three forward modes so it can be
    used as a drop-in replacement for SharedViT in pretrain_v2.py:
        forward_all(images)     -> (cls [B,1024], patches [B,196,1024])
        forward_cls(images)     -> z_img [B,embed_dim]   for L_ITA
        forward_patches(images) -> patches [B,196,1024]  for L_sim / L_ortho

    _features(images) returns [B, 197, 1024] (CLS at 0, patches at 1:) to match
    the slicing pattern used by pretrain_v2.py: vit._features(img)[:, 0, :].
    """

    def __init__(
        self,
        config_path: str,
        weights_path: str,
        lora_layers: int,
        r: int,
        alpha: float,
        embed_dim: int = EMBED_DIM,
    ) -> None:
        super().__init__()
        from chexfound.models.encoders import CheXFoundViT as _CheXFoundViT
        from chexfound.data.transforms import build_preprocess_val_chexfound

        self._encoder = _CheXFoundViT(
            config_path, weights_path, lora_layers, r, alpha, load_pretrained=True
        )
        self.preprocess_val = build_preprocess_val_chexfound()
        self.img_proj   = ProjectionHead(CHEXFOUND_VIT_DIM, embed_dim)
        self.patch_proj = ProjectionHead(CHEXFOUND_VIT_DIM, embed_dim)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"CheXFoundSharedViT: LoRA in last {lora_layers} blocks (r={r}, alpha={alpha}) | "
            f"trainable: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)"
        )

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        """Return [B, 197, 1024] with CLS at index 0, patches at 1:."""
        cls, patches = self._encoder._features(images)
        return torch.cat([cls.unsqueeze(1), patches], dim=1)

    def forward_all(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls, patches = self._encoder._features(images)
        return cls, patches

    def forward_cls(self, images: torch.Tensor) -> torch.Tensor:
        cls, _ = self._encoder._features(images)
        return self.img_proj(cls)

    def forward_patches(self, images: torch.Tensor) -> torch.Tensor:
        _, patches = self._encoder._features(images)
        return patches

    @property
    def vit_dim(self) -> int:
        return CHEXFOUND_VIT_DIM


class BiomedCLIPTextEncoder(nn.Module):
    """BiomedCLIP PubMedBERT text encoder with trainable projection heads.

    Transformer weights are frozen; cls_proj and word_proj are trained.
    self.tokenizer is exposed for data loading.

    encode_beurteilung    →  CLS projection for L_ITA  (full / concat modes)
    encode_befund         →  (CLS, word projections) for L_sim  (full / concat)
    encode_befund_phrases →  (mean CLS, phrase embeddings) for L_sim  (phrase modes)
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        _model, _, _ = open_clip.create_model_and_transforms(MODEL_TAG)
        self.transformer = _model.text.transformer
        self.tokenizer   = open_clip.get_tokenizer(MODEL_TAG)
        del _model
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        self.embed_dim = embed_dim
        self.cls_proj  = ProjectionHead(BERT_DIM, embed_dim, hidden_dim=640)
        self.word_proj = ProjectionHead(BERT_DIM, embed_dim, hidden_dim=640)

        n_bert = sum(p.numel() for p in self.transformer.parameters())
        n_proj = (sum(p.numel() for p in self.cls_proj.parameters()) +
                  sum(p.numel() for p in self.word_proj.parameters()))
        print(
            f"BiomedCLIPTextEncoder: frozen transformer ({n_bert:,} params) + "
            f"trainable projections ({n_proj:,} params)"
        )

    def load_pretrained_projections(self) -> None:
        """Copy cls_proj weights from BiomedCLIP's pretrained text.proj (768→640→512)."""
        _model, _, _ = open_clip.create_model_and_transforms(MODEL_TAG)
        src = _model.text.proj
        self.cls_proj.proj[0].weight.data.copy_(src[0].weight.data)
        self.cls_proj.proj[2].weight.data.copy_(src[2].weight.data)
        del _model
        print("BiomedCLIPTextEncoder: loaded pretrained cls_proj from BiomedCLIP text.proj")

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
        """Projected CLS of Beurteilung text for L_ITA. Returns [B, 512]."""
        seq = self._forward(input_ids, attention_mask)
        return self.cls_proj(seq[:, 0, :])

    def encode_befund(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CLS + word-token projections of Befund text for L_sim (full-text mode).

        Returns (z_cls [B, 512], proj_phrases [B, L, 512]).
        z_cls is kept for API compatibility; sim_loss_v2 uses proj_phrases only.
        """
        seq = self._forward(input_ids, attention_mask)
        B, L, D = seq.shape
        proj_phrases = self.word_proj(seq.reshape(-1, D)).reshape(B, L, self.embed_dim)
        return self.cls_proj(seq[:, 0, :]), proj_phrases

    def encode_befund_phrases(
        self,
        phrase_ids: torch.Tensor,
        phrase_attn: torch.Tensor,
        phrase_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode befund phrases for L_sim (phrase mode).

        Returns:
            z_cls        [B, embed_dim]    mean-pooled phrase CLS (InfoNCE anchor in v1)
            proj_phrases [B, N, embed_dim] per-phrase CLS embeddings (feeds sim_loss_v2)
        """
        proj_phrases = self._encode_phrase_batch(phrase_ids, phrase_attn, phrase_mask)
        n_real = phrase_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        z_cls  = F.normalize(proj_phrases.sum(dim=1) / n_real, dim=-1)
        return z_cls, proj_phrases
