from __future__ import annotations

import warnings
from typing import Iterable

import torch.nn as nn

from biomedclip.models.lora import LoRALinear


def _iter_real_vit_blocks(model: nn.Module) -> list[nn.Module]:
    """Return the real transformer blocks, flattening upstream BlockChunks.

    Upstream CheXFound/DINOv2 stores blocks either as a flat ModuleList or as
    chunked ModuleLists padded with Identity modules. LoRA layer counts should
    refer to real transformer blocks, not FSDP chunks or Identity placeholders.
    """
    real_blocks: list[nn.Module] = []
    for item in model.blocks:
        children: Iterable[nn.Module]
        if isinstance(item, nn.ModuleList):
            children = item
        else:
            children = (item,)

        for block in children:
            if isinstance(block, nn.Identity):
                continue
            if hasattr(block, "attn") and hasattr(block, "mlp"):
                real_blocks.append(block)

    return real_blocks


def _wrap_linear(parent: nn.Module, attr: str, r: int, alpha: float) -> bool:
    layer = getattr(parent, attr, None)
    if layer is None:
        return False
    if isinstance(layer, LoRALinear):
        return True
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected {parent.__class__.__name__}.{attr} to be nn.Linear, got {type(layer)!r}")
    setattr(parent, attr, LoRALinear(layer, r, alpha))
    return True


def _wrap_mlp_linears(mlp: nn.Module, r: int, alpha: float) -> None:
    # timm-style MLP uses fc1/fc2; upstream CheXFound SwiGLU uses w12/w3.
    if _wrap_linear(mlp, "fc1", r, alpha) | _wrap_linear(mlp, "fc2", r, alpha):
        return
    if _wrap_linear(mlp, "w12", r, alpha) | _wrap_linear(mlp, "w3", r, alpha):
        return
    raise AttributeError(
        "Could not find supported MLP linear names. Expected fc1/fc2 or w12/w3."
    )


def inject_lora_chexfound(
    model: nn.Module,
    lora_layers: int,
    r: int,
    alpha: float,
) -> None:
    """Freeze all CheXFound ViT params; inject LoRA into the last lora_layers blocks.

    CheXFound/DINOv2 exposes blocks at model.blocks. Depending on block_chunks,
    this can be a flat list or a list of BlockChunks padded with Identity
    modules. This function always counts real transformer blocks, so
    lora_layers=6 means blocks 18-23 in ViT-L/16.

    Targets per block:
      - attn.qkv
      - attn.proj
      - mlp.w12 and mlp.w3 for upstream SwiGLU, or mlp.fc1/fc2 for local/timm MLP
    """
    for p in model.parameters():
        p.requires_grad_(False)

    if lora_layers == 0:
        return

    blocks = _iter_real_vit_blocks(model)
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
        _wrap_linear(block.attn, "qkv", r, alpha)
        _wrap_linear(block.attn, "proj", r, alpha)
        _wrap_mlp_linears(block.mlp, r, alpha)


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
