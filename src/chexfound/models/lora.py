from __future__ import annotations

import warnings
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LoRASwiGLUFFN(nn.Module):
    """LoRA adapter for xFormers/local SwiGLU FFNs.

    xFormers' fused SwiGLU kernel reads the wrapped linears directly, so plain
    LoRALinear replacement can be skipped by the fused forward. This wrapper
    keeps the frozen w12/w3 weights and runs an explicit SwiGLU forward with
    trainable low-rank branches.
    """

    def __init__(self, swiglu: nn.Module, r: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(getattr(swiglu, "w12", None), nn.Linear):
            raise TypeError(f"Expected {swiglu.__class__.__name__}.w12 to be nn.Linear")
        if not isinstance(getattr(swiglu, "w3", None), nn.Linear):
            raise TypeError(f"Expected {swiglu.__class__.__name__}.w3 to be nn.Linear")
        self.w12 = LoRALinear(swiglu.w12, r, alpha)
        self.w3 = LoRALinear(swiglu.w3, r, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _wrap_mlp_linears(mlp: nn.Module, r: int, alpha: float) -> nn.Module:
    if isinstance(mlp, LoRASwiGLUFFN):
        return mlp

    # xFormers fused SwiGLU accesses .weight directly inside its CUDA kernel,
    # bypassing LoRALinear.forward entirely. Wrap the full FFN and run the
    # SwiGLU math explicitly so the low-rank branch participates.
    try:
        from xformers.ops.swiglu_op import SwiGLU
        if isinstance(mlp, SwiGLU):
            return LoRASwiGLUFFN(mlp, r, alpha)
    except ImportError:
        pass

    # timm-style MLP uses fc1/fc2; non-fused CheXFound SwiGLU uses w12/w3.
    if _wrap_linear(mlp, "fc1", r, alpha) | _wrap_linear(mlp, "fc2", r, alpha):
        return mlp
    if _wrap_linear(mlp, "w12", r, alpha) | _wrap_linear(mlp, "w3", r, alpha):
        return mlp
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
        block.mlp = _wrap_mlp_linears(block.mlp, r, alpha)


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
