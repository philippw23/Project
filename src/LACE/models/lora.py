from __future__ import annotations

import warnings

import torch.nn as nn

from biomedclip.models.lora import LoRALinear


def inject_lora_vit(
    vit_trunk: nn.Module,
    lora_layers: int,
    r: int,
    alpha: float,
) -> None:
    """Freeze all ViT trunk params; inject LoRA into the last lora_layers blocks.

    Operates on model.visual.trunk (timm VisionTransformer) directly.
    Targets per block: attn.qkv, attn.proj, mlp.fc1, mlp.fc2.
    """
    for p in vit_trunk.parameters():
        p.requires_grad_(False)

    blocks = vit_trunk.blocks
    n_blocks = len(blocks)
    effective = min(lora_layers, n_blocks)
    if effective < lora_layers:
        warnings.warn(
            f"lora_layers={lora_layers} exceeds total ViT blocks ({n_blocks}). "
            f"Applying LoRA to all {n_blocks} blocks.",
            stacklevel=2,
        )

    for i in range(n_blocks - effective, n_blocks):
        block = blocks[i]
        block.attn.qkv  = LoRALinear(block.attn.qkv,  r, alpha)
        block.attn.proj = LoRALinear(block.attn.proj,  r, alpha)
        block.mlp.fc1   = LoRALinear(block.mlp.fc1,    r, alpha)
        block.mlp.fc2   = LoRALinear(block.mlp.fc2,    r, alpha)


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
