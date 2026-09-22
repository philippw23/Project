from __future__ import annotations

import torch
import torch.nn as nn

from chexfound.models.lora import count_trainable_params, inject_lora_chexfound
from chexfound.models.model_factory import build_model_from_cfg
from chexfound.models.weight_utils import load_pretrained_weights

VIT_DIM = 1024  # CheXFound ViT-L embedding dimension


class CheXFoundViT(nn.Module):
    """CheXFound ViT-L/16 (1024-dim) with optional LoRA adaptation.

    Exposes:
        forward(images)     -> cls [B, 1024]   for downstream classification
        _features(images)   -> (cls [B, 1024], patches [B, N, 1024])

    forward_features() returns a DINOv2-style dict so the call site in
    chexfound/train/downstream.py is unchanged.

    Args:
        config_path:     Path to the CheXFound model config YAML.
        weights_path:    Path to the pretrained CheXFound .pth checkpoint.
                         Required only when load_pretrained=True.
        lora_layers:     Number of last ViT blocks to apply LoRA to (0 = none).
        r:               LoRA rank.
        alpha:           LoRA alpha scaling.
        load_pretrained: If False, skip loading weights_path (caller will
                         overwrite via load_continued_pretrain_weights).
    """

    def __init__(
        self,
        config_path: str,
        weights_path: str | None,
        lora_layers: int,
        r: int,
        alpha: float,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.trunk, _, embed_dim = build_model_from_cfg(config_path, only_teacher=True)
        if embed_dim != VIT_DIM:
            raise ValueError(f"Expected embed_dim={VIT_DIM}, got {embed_dim}")

        if load_pretrained:
            if weights_path is None:
                raise ValueError("weights_path must be set when load_pretrained=True")
            load_pretrained_weights(self.trunk, weights_path, checkpoint_key="teacher")

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
    """Load teacher weights from a checkpoint saved by pretrain.py."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora_config") if isinstance(ckpt, dict) else None
    lora_layers = 0
    if lora_cfg and lora_cfg.get("lora_layers", 0) > 0:
        lora_layers = lora_cfg["lora_layers"]
        inject_lora_chexfound(
            model.trunk,
            lora_layers=lora_layers,
            r=lora_cfg["lora_r"],
            alpha=lora_cfg["lora_alpha"],
        )
    load_pretrained_weights(model.trunk, checkpoint_path, checkpoint_key="teacher")
    n_train = count_trainable_params(model)
    n_total = sum(p.numel() for p in model.parameters())
    r     = lora_cfg["lora_r"]     if lora_cfg else 0
    alpha = lora_cfg["lora_alpha"] if lora_cfg else 0.0
    print(
        f"CheXFoundViT (after loading checkpoint): LoRA in last {lora_layers} blocks "
        f"(r={r}, alpha={alpha}) | trainable: {n_train:,} / {n_total:,} "
        f"({100 * n_train / n_total:.2f}%)"
    )
