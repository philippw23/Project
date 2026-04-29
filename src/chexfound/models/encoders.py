from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from chexfound.models.lora import count_trainable_params, inject_lora_chexfound

VIT_DIM = 1024  # CheXFound ViT-L embedding dimension

# ── CheXFound repo dependency ─────────────────────────────────────────────────
_ROOT = os.environ.get("CHEXFOUND_ROOT", "")
if not _ROOT:
    raise ImportError(
        "Environment variable CHEXFOUND_ROOT is not set. "
        "Set it to the root of the cloned RPIDIAL/CheXFound repository:\n"
        "  export CHEXFOUND_ROOT=/path/to/CheXFound"
    )
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from chexfound.models import build_model_from_cfg  # type: ignore[import]
    from chexfound import utils as dinov2_utils          # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        f"Cannot import chexfound from CHEXFOUND_ROOT={_ROOT!r}. "
        "Ensure the path points to a valid CheXFound repo checkout."
    ) from exc


class CheXFoundViT(nn.Module):
    """CheXFound ViT-L/16 (1024-dim) with optional LoRA adaptation.

    Exposes:
        forward(images)     -> cls [B, 1024]   for downstream classification
        _features(images)   -> (cls [B, 1024], patches [B, 1024, 1024])

    CheXFound's forward_features() returns a dict (DINOv2-style), unlike
    BiomedCLIP/timm which return a tensor with CLS at index 0.

    Args:
        config_path:     Path to the CheXFound model config YAML.
        weights_path:    Path to the pretrained CheXFound .pth checkpoint.
        lora_layers:     Number of last ViT blocks to apply LoRA to (0 = none).
        r:               LoRA rank.
        alpha:           LoRA alpha scaling.
        load_pretrained: If False, skip loading weights_path (caller will
                         overwrite via load_state_dict from a fine-tuned ckpt).
    """

    def __init__(
        self,
        config_path: str,
        weights_path: str,
        lora_layers: int,
        r: int,
        alpha: float,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.trunk, embed_dim = build_model_from_cfg(config_path, only_teacher=True)
        if embed_dim != VIT_DIM:
            raise ValueError(f"Expected embed_dim={VIT_DIM}, got {embed_dim}")

        if load_pretrained:
            dinov2_utils.load_pretrained_weights(self.trunk, weights_path, "teacher")

        inject_lora_chexfound(self.trunk, lora_layers, r, alpha)

        n_train = count_trainable_params(self)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"CheXFoundViT: LoRA in last {lora_layers} blocks (r={r}, alpha={alpha}) | "
            f"trainable: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)"
        )

    def _features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.trunk.forward_features(images)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls, _ = self._features(images)
        return cls  # [B, 1024]


def load_continued_pretrain_weights(model: CheXFoundViT, checkpoint_path: str) -> None:
    """Load teacher weights from a checkpoint saved by CheXFound's train.py."""
    dinov2_utils.load_pretrained_weights(model.trunk, checkpoint_path, "teacher")
