from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank branch.

    W_eff = W_frozen + (B @ A) * (alpha / r)
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.linear = linear
        self.r = r
        self.scale = alpha / r

        device = linear.weight.device
        dtype = linear.weight.dtype

        self.lora_A = nn.Parameter(
            torch.randn(r, linear.in_features, device=device, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(linear.out_features, r, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


class LoRAConv2d(nn.Module):
    """Wraps a frozen 1x1 nn.Conv2d with a trainable low-rank branch.

    Only supports kernel_size=1 (equivalent to a linear map over channels).
    W_eff = W_frozen + (B @ A).view(out_ch, in_ch, 1, 1) * (alpha / r)
    """

    def __init__(self, conv: nn.Conv2d, r: int, alpha: float) -> None:
        super().__init__()
        if conv.kernel_size != (1, 1):
            raise ValueError(
                f"LoRAConv2d only supports kernel_size=1, got {conv.kernel_size}"
            )
        self.conv = conv
        self.r = r
        self.scale = alpha / r

        in_ch = conv.in_channels
        out_ch = conv.out_channels
        device = conv.weight.device
        dtype = conv.weight.dtype

        self.lora_A = nn.Parameter(
            torch.randn(r, in_ch, device=device, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_ch, r, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_out = self.conv(x)
        delta_w = (self.lora_B @ self.lora_A).view(
            self.conv.out_channels, self.conv.in_channels, 1, 1
        )
        lora_out = F.conv2d(
            x, delta_w, None,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )
        return frozen_out + lora_out * self.scale


def inject_lora_gloria(
    img_encoder: nn.Module,
    lora_layers: int,
    r: int,
    alpha: float,
) -> None:
    """Freeze img_encoder, then inject LoRA into ResNet50 1x1-Conv and projection heads.

    Targets:
      - All 3 blocks of model.layer4: conv1 + conv3 (both 1x1) + downsample[0] if present
      - Last `lora_layers` blocks of model.layer3: conv1 + conv3 (1x1)
      - global_embedder (nn.Linear)
      - local_embedder  (nn.Conv2d 1x1)
    """
    for p in img_encoder.parameters():
        p.requires_grad_(False)

    backbone = img_encoder.model

    # layer4: all 3 Bottleneck blocks
    for block in backbone.layer4:
        block.conv1 = LoRAConv2d(block.conv1, r, alpha)
        block.conv3 = LoRAConv2d(block.conv3, r, alpha)
        if block.downsample is not None:
            ds_conv = block.downsample[0]
            if isinstance(ds_conv, nn.Conv2d) and ds_conv.kernel_size == (1, 1):
                block.downsample[0] = LoRAConv2d(ds_conv, r, alpha)

    # layer3: last lora_layers blocks
    if lora_layers > 0:
        layer3_blocks = list(backbone.layer3)
        n = len(layer3_blocks)
        effective = min(lora_layers, n)
        if effective < lora_layers:
            warnings.warn(
                f"lora_layers={lora_layers} exceeds layer3 block count ({n}). "
                f"Applying to all {n} blocks.",
                stacklevel=2,
            )
        for block in layer3_blocks[-effective:]:
            block.conv1 = LoRAConv2d(block.conv1, r, alpha)
            block.conv3 = LoRAConv2d(block.conv3, r, alpha)

    # projection heads
    img_encoder.global_embedder = LoRALinear(img_encoder.global_embedder, r, alpha)
    img_encoder.local_embedder = LoRAConv2d(img_encoder.local_embedder, r, alpha)


def unfreeze_gloria(
    img_encoder: nn.Module,
    n_layers: int,
) -> None:
    """Freeze img_encoder, then unfreeze last N ResNet layer3 blocks + all of layer4 + projection heads.

    Symmetric semantics to inject_lora_gloria:
      n_layers == 0 → only projection heads (global_embedder, local_embedder) trainable
      n_layers  > 0 → additionally unfreezes all layer4 blocks and last n_layers layer3 blocks
    """
    for p in img_encoder.parameters():
        p.requires_grad_(False)

    backbone = img_encoder.model

    if n_layers > 0:
        for block in backbone.layer4:
            for p in block.parameters():
                p.requires_grad_(True)

        layer3_blocks = list(backbone.layer3)
        n = len(layer3_blocks)
        effective = min(n_layers, n)
        if effective < n_layers:
            warnings.warn(
                f"n_layers={n_layers} exceeds layer3 block count ({n}). "
                f"Unfreezing all {n} blocks.",
                stacklevel=2,
            )
        for block in layer3_blocks[-effective:]:
            for p in block.parameters():
                p.requires_grad_(True)

    for p in img_encoder.global_embedder.parameters():
        p.requires_grad_(True)
    for p in img_encoder.local_embedder.parameters():
        p.requires_grad_(True)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
