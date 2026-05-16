from __future__ import annotations

import warnings

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank branch.

    The effective weight becomes:
        W_eff = W_frozen + (B @ A) * (alpha / r)

    where A has shape (r, in_features) and B has shape (out_features, r).
    B is initialised to zero so the branch contributes nothing at step 0,
    preserving the pretrained behaviour at the start of fine-tuning.
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.linear = linear
        self.r      = r
        self.scale  = alpha / r  # constant scaling factor applied to the LoRA output

        in_features  = linear.in_features
        out_features = linear.out_features
        device = linear.weight.device
        dtype  = linear.weight.dtype

        # A: down-projection matrix, initialised with small Gaussian noise.
        self.lora_A = nn.Parameter(
            torch.randn(r, in_features, device=device, dtype=dtype) * 0.01
        )
        # B: up-projection matrix, initialised to zero (LoRA starts as identity).
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, r, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen linear output plus the low-rank adaptation.
        return self.linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def inject_lora(model: nn.Module, lora_layers: int, r: int, alpha: float) -> None:
    """Freeze the whole model, then inject LoRA into the last N ViT blocks.

    Freezes all parameters first, then replaces the four linear layers in each
    of the last lora_layers transformer blocks with LoRALinear wrappers that
    add a trainable low-rank branch alongside the frozen original weights.

    Additionally unfreezes:
      - model.logit_scale       : learnable contrastive temperature
      - model.visual.proj       : projection head from ViT space to CLIP space

    BiomedCLIP uses a TimmModel wrapper, so the block path is:
        model.visual.trunk.blocks[i]

    Targets per block (timm ViT naming):
        block.attn.qkv   (fused QKV Linear: dim → 3 × dim)
        block.attn.proj  (attention output projection: dim → dim)
        block.mlp.fc1    (MLP first layer: dim → mlp_width)
        block.mlp.fc2    (MLP second layer: mlp_width → dim)
    """
    # Freeze the entire model as a baseline; only the LoRA branches and the
    # explicitly listed parameters below will be trainable.
    for p in model.parameters():
        p.requires_grad_(False)

    blocks = model.visual.trunk.blocks
    n_blocks = len(blocks)
    effective_layers = min(lora_layers, n_blocks)
    if effective_layers < lora_layers:
        warnings.warn(
            f"--lora_layers={lora_layers} exceeds total ViT blocks ({n_blocks}). "
            f"Applying LoRA to all {n_blocks} blocks.",
            stacklevel=2,
        )

    # Select only the last effective_layers blocks (highest-level representations).
    target_indices = range(n_blocks - effective_layers, n_blocks)

    for i in target_indices:
        block = blocks[i]
        # Wrap each of the four linear layers with a LoRALinear.
        # The original weights remain frozen; only lora_A and lora_B are trained.
        block.attn.qkv  = LoRALinear(block.attn.qkv,  r, alpha)
        block.attn.proj = LoRALinear(block.attn.proj, r, alpha)
        block.mlp.fc1   = LoRALinear(block.mlp.fc1,   r, alpha)
        block.mlp.fc2   = LoRALinear(block.mlp.fc2,   r, alpha)

    # Unfreeze the contrastive temperature so it adapts to the training batch size.
    model.logit_scale.requires_grad_(True)

    if hasattr(model.visual, "head") and model.visual.head is not None:
        for p in model.visual.head.parameters():
            p.requires_grad_(True)
    if hasattr(model.text, "proj") and model.text.proj is not None:
        for p in model.text.proj.parameters():
            p.requires_grad_(True)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
