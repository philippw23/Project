from __future__ import annotations

import timm
import torch.nn as nn

EMBED_DIM = 768


def build_encoder() -> tuple[nn.Module, int]:
    """Return (encoder, embed_dim) for ViT-B/16 with ImageNet pretrained weights.

    num_classes=0 makes timm return the pooled [CLS] feature vector directly.
    """
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    return model, EMBED_DIM
