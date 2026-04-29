"""DINOv2-style Vision Transformer (ViT-L/16) — self-contained implementation.

Architecture follows the CheXFound / DINOv2 specification:
  - SwiGLU FFN with fused gate+value linear (fc1) and output (fc2)
  - Register tokens (no positional embedding applied to them)
  - LayerScale initialised at 1e-5
  - uniform drop-path rate across all blocks
  - forward_features() returns a dict with "x_norm_clstoken" / "x_norm_patchtokens"
    so the existing encoders.py interface stays unchanged.
  - mask_token for iBOT student-side masked prediction
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Stochastic depth ──────────────────────────────────────────────────────────

def _drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    noise = x.new_empty(shape).bernoulli_(keep).div_(keep)
    return x * noise


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return _drop_path(x, self.drop_prob, self.training)


# ── LayerScale ────────────────────────────────────────────────────────────────

class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


# ── Feed-forward networks ─────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU with a single fused linear for gate+value (fc1) and output (fc2).

    hidden_features is the bottleneck dimension *after* chunking fc1's output.
    fc1 produces 2 * hidden_features so the two halves serve as gate and value.

    Naming (fc1 / fc2) is intentional — it matches the LoRA injection targets in
    lora.py and the weight keys written by the CheXFound training code.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        gate, val = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * val)


# ── Multi-head self-attention ─────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# ── Transformer block ─────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale: float = 1e-5,
        ffn_layer: str = "swiglufused",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.ls1 = LayerScale(dim, layerscale)
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        # hidden_features sized so that SwiGLU has ~same params as standard MLP:
        # 3 * D * H  =  2 * D * (4D)  →  H = 8D/3
        if ffn_layer == "swiglufused":
            hidden = int(dim * mlp_ratio * 2 / 3)
            self.mlp = SwiGLUFFN(dim, hidden, bias=ffn_bias)
        else:
            hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden, bias=ffn_bias),
                nn.GELU(),
                nn.Linear(hidden, dim, bias=ffn_bias),
            )
        self.ls2 = LayerScale(dim, layerscale)
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ── Patch embedding ───────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, tuple] = 518,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


# ── Vision Transformer ────────────────────────────────────────────────────────

class VisionTransformer(nn.Module):
    """DINOv2-style ViT with optional register tokens and mask token for iBOT.

    forward_features() returns:
        {
            "x_norm_clstoken":   (B, D)    — normalised CLS token
            "x_norm_patchtokens": (B, N, D) — normalised patch tokens (registers excluded)
            "x_prenorm":          (B, 1+R+N, D) — pre-norm sequence (for debugging)
        }
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        ffn_layer: str = "swiglufused",
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = True,
        layerscale: float = 1e-5,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos_embed covers CLS + patches; register tokens get no positional signal.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, embed_dim))

        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        else:
            self.register_tokens = None

        # Learnable mask token used during iBOT student training.
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        dpr_val = drop_path_rate
        dpr = [dpr_val] * depth if drop_path_uniform else [
            x.item() for x in torch.linspace(0, dpr_val, depth)
        ]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path_rate=dpr[i],
                layerscale=layerscale,
                ffn_layer=ffn_layer,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Positional encoding interpolation ─────────────────────────────────────

    def _interpolate_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
        """Bicubic interpolation of pos_embed for images larger/smaller than default."""
        npatch = x.shape[1]          # current patch count
        N = self.pos_embed.shape[1] - 1  # stored patch count

        if npatch == N and w == h:
            return self.pos_embed

        cls_pe = self.pos_embed[:, :1]
        patch_pe = self.pos_embed[:, 1:]  # (1, N, D)

        M = int(math.sqrt(N))
        assert M * M == N, f"pos_embed has non-square grid ({N} patches)"
        D = self.embed_dim
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        kwargs: dict = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / (M + self.interpolate_offset)
            sy = float(h0 + self.interpolate_offset) / (M + self.interpolate_offset)
            kwargs["scale_factor"] = (sy, sx)
        else:
            kwargs["size"] = (h0, w0)

        patch_pe = (
            patch_pe.reshape(1, M, M, D)
            .permute(0, 3, 1, 2)
        )
        patch_pe = F.interpolate(
            patch_pe,
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, -1, D)
        return torch.cat([cls_pe, patch_pe], dim=1)

    # ── Token preparation ─────────────────────────────────────────────────────

    def _prepare_tokens(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Embed patches, apply mask token, prepend CLS + registers, add pos_embed."""
        B, C, H, W = x.shape
        patches = self.patch_embed(x)  # (B, N, D)

        if mask is not None:
            # mask: (B, N) bool — True marks positions to be replaced.
            mt = self.mask_token.unsqueeze(0).expand(B, patches.shape[1], -1)
            patches = patches * (~mask).unsqueeze(-1) + mt * mask.unsqueeze(-1)

        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        pos = self._interpolate_pos_encoding(patches, W, H)  # (1, 1+N, D)

        # Apply positional encoding to CLS and patches separately.
        cls = cls + pos[:, :1]
        patches = patches + pos[:, 1:]

        if self.register_tokens is not None:
            reg = self.register_tokens.expand(B, -1, -1)  # (B, R, D) — no pos embed
            return torch.cat([cls, reg, patches], dim=1)
        return torch.cat([cls, patches], dim=1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward_features(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> dict:
        x = self._prepare_tokens(x, mask=mask)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        R = self.num_register_tokens
        return {
            "x_norm_clstoken":    x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1 + R:],
            "x_prenorm":          x,
        }

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.forward_features(x, mask=mask)["x_norm_clstoken"]


# ── Architecture presets ──────────────────────────────────────────────────────

_ARCH: dict[str, dict] = {
    "vit_small":  {"embed_dim": 384,  "depth": 12, "num_heads": 6},
    "vit_base":   {"embed_dim": 768,  "depth": 12, "num_heads": 12},
    "vit_large":  {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "vit_huge":   {"embed_dim": 1280, "depth": 32, "num_heads": 16},
}


def build_vit(arch: str = "vit_large", **kwargs) -> VisionTransformer:
    """Construct a VisionTransformer from an architecture name + optional overrides."""
    cfg = _ARCH[arch].copy()
    cfg.update(kwargs)
    return VisionTransformer(**cfg)
