from __future__ import annotations

import timm
import torch.nn as nn


def build_encoder(encoder_name: str) -> tuple[nn.Module, int]:
    """Return (encoder, embed_dim) for the requested architecture.

    Both models are initialised with random weights (pretrained=False).
    num_classes=0 makes timm return the pooled feature vector directly.
    """
    if encoder_name == "resnet18":
        model = timm.create_model("resnet18", pretrained=False, num_classes=0)
        return model, 512
    if encoder_name == "vit_tiny":
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
        return model, 192
    raise ValueError(f"Unknown encoder: {encoder_name!r}. Choose 'resnet18' or 'vit_tiny'.")
