"""YOLO architecture inspection utilities."""
from __future__ import annotations

from pathlib import Path


def _count_params(module) -> tuple[int, int]:
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_yolo_architecture(weights: str, imgsz: int = 640) -> None:
    """Print a structured summary of a pretrained YOLO model.

    Shows every layer with its index, type, source index, and parameter count.
    Splits the model into backbone (up to and including SPPF), neck, and
    Detect head, and prints parameter totals for each section.

    Args:
        weights: Ultralytics model weights identifier or path (e.g. "yolov5l6u.pt").
        imgsz:   Image size used to compute the backbone embedding dimension.
    """
    import torch
    from ultralytics import YOLO

    print(f"Loading: {weights}")
    yolo  = YOLO(weights)
    inner = yolo.model
    layers = inner.model

    total_p, train_p = _count_params(inner)

    print("\n" + "=" * 70)
    print(f"YOLO Architecture  —  {weights}")
    print("=" * 70)
    print(f"Total parameters    : {total_p:,}")
    print(f"Trainable parameters: {train_p:,}  ({100 * train_p / max(total_p, 1):.1f} %)")
    print(f"Number of layers    : {len(layers)}")

    detect_idx = len(layers) - 1

    # Find SPPF to split backbone / neck
    sppf_idx = None
    for i, m in enumerate(layers):
        if type(m).__name__ == "SPPF":
            sppf_idx = i

    print(f"\n{'Idx':>4}  {'Type':<32}  {'From':>8}  {'Params':>14}  {'Section'}")
    print("-" * 75)

    section_params: dict[str, int] = {"backbone": 0, "neck": 0, "detect": 0}

    for i, m in enumerate(layers):
        p, _ = _count_params(m)
        f    = getattr(m, "f", -1)
        name = type(m).__name__

        if i == detect_idx:
            section = "detect"
        elif sppf_idx is not None and i <= sppf_idx:
            section = "backbone"
        else:
            section = "neck"

        section_params[section] += p
        print(f"{i:>4}  {name:<32}  {str(f):>8}  {p:>14,}  {section}")

    print("-" * 75)
    backbone_end = sppf_idx if sppf_idx is not None else "?"
    neck_end     = detect_idx - 1
    print(f"\nBackbone (0 – {backbone_end})   : {section_params['backbone']:>14,} params")
    print(f"Neck     ({backbone_end + 1 if sppf_idx else '?'} – {neck_end})  : {section_params['neck']:>14,} params")
    print(f"Detect head            : {section_params['detect']:>14,} params")

    # Detect head summary
    detect_layer = layers[detect_idx]
    nc = getattr(detect_layer, "nc", "?")
    print(f"\nDetect head: nc={nc} classes")
    if hasattr(detect_layer, "cv3"):
        print(f"  Output scales (P3/P4/P5): {len(detect_layer.cv3)}")

    # Embedding dim via YOLOBackbone (pooled P3+P4+P5)
    print("\n" + "=" * 70)
    print("Backbone embedding (pooled P3 + P4 + P5)")
    print("=" * 70)
    from yolo.models.encoders import build_encoder
    _, embed_dim = build_encoder(weights, imgsz)
    detect_srcs = layers[detect_idx].f
    print(f"Detect head sources : {detect_srcs}")
    print(f"Embedding dimension : {embed_dim:,}")
