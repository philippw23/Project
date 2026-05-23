"""YOLO backbone encoder for downstream classification.

Loads a pretrained Ultralytics YOLO model, strips the Detect head, and
exposes the multi-scale neck features (P3/P4/P5) as a single embedding
via global average pooling + concatenation.

The Detect head's input feature maps are used so we capture all three
resolution scales the model was trained to attend to.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class YOLOBackbone(nn.Module):
    """YOLO backbone + neck without the Detect head.

    Forward: full image → concatenated global-pooled P3/P4/P5 features.
    The embedding dimension is the sum of channels across all three scales
    and is set automatically on construction via a dummy forward pass.
    """

    def __init__(self, weights: str) -> None:
        super().__init__()
        from ultralytics import YOLO

        yolo = YOLO(weights)
        inner = yolo.model

        self.layers      = inner.model
        self.save_set    = set(inner.save)
        self.detect_idx  = len(inner.model) - 1
        # Detect layer's source indices — the three neck output layers
        self.detect_srcs = inner.model[-1].f
        self.pool        = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: list = []
        for i, m in enumerate(self.layers):
            if i == self.detect_idx:
                break
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            x = m(x)
            y.append(x if i in self.save_set else None)

        feat_maps = [x if j == -1 else y[j] for j in self.detect_srcs]
        pooled    = [self.pool(f).flatten(1) for f in feat_maps]
        return torch.cat(pooled, dim=1)


class BBoxHead(nn.Module):
    """Predicts one normalised bounding box (cx, cy, w, h) ∈ [0, 1]."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


def build_encoder(weights: str, imgsz: int = 640) -> tuple[nn.Module, int]:
    """Return (backbone, embed_dim) for a pretrained YOLO model.

    embed_dim is determined by a single dummy forward pass and equals the
    sum of channels at all neck output scales (P3 + P4 + P5).
    """
    backbone = YOLOBackbone(weights)
    backbone.eval()
    with torch.no_grad():
        dummy     = torch.zeros(1, 3, imgsz, imgsz)
        embed_dim = backbone(dummy).shape[1]
    return backbone, embed_dim
