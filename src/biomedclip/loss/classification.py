from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    return weights / weights.mean().clamp_min(1e-12)


def compute_class_weights(
    label_counts: torch.Tensor,
    num_classes: int,
    mode: str,
    beta: float,
    device: torch.device,
) -> torch.Tensor | None:
    counts = label_counts.to(dtype=torch.float32).clamp_min(1.0)
    if mode == "none":
        return None
    if mode == "inverse":
        weights = counts.sum() / (num_classes * counts)
    elif mode == "sqrt":
        weights = (counts.sum() / (num_classes * counts)).sqrt()
    elif mode == "effective":
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(counts, beta), counts))
    else:
        raise ValueError(f"Unknown class weighting mode: {mode}")
    return _normalize_weights(weights).to(device)


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_unweighted = F.cross_entropy(
            logits,
            target,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        ce_weighted = F.cross_entropy(
            logits,
            target,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_unweighted)
        return ((1.0 - pt).pow(self.gamma) * ce_weighted).mean()


class LDAMLoss(nn.Module):
    def __init__(
        self,
        class_counts: torch.Tensor,
        max_margin: float = 0.5,
        weight: torch.Tensor | None = None,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        counts = class_counts.to(dtype=torch.float32).clamp_min(1.0)
        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        margins = margins * (max_margin / margins.max())
        self.register_buffer("margins", margins)
        self.scale = scale
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        margins = self.margins[target].unsqueeze(1)
        adjusted = logits.scatter_add(1, target.unsqueeze(1), -margins)
        return F.cross_entropy(self.scale * adjusted, target, weight=self.weight)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: torch.Tensor) -> None:
        super().__init__()
        log_counts = class_counts.to(dtype=torch.float32).clamp_min(1.0).log()
        self.register_buffer("log_counts", log_counts)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.log_counts.unsqueeze(0), target)


def build_classification_loss(
    args: argparse.Namespace,
    label_counts: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None,
    device: torch.device,
) -> nn.Module:
    counts = label_counts.to(device=device, dtype=torch.float32)
    if args.loss == "ce":
        return nn.CrossEntropyLoss()
    if args.loss == "wce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if args.loss == "ce_smooth":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    if args.loss == "focal":
        return FocalLoss(alpha=class_weights, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    if args.loss == "cb_focal":
        alpha = compute_class_weights(label_counts, num_classes, "effective", args.cb_beta, device)
        return FocalLoss(alpha=alpha, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    if args.loss == "ldam":
        return LDAMLoss(counts, max_margin=args.ldam_max_margin, weight=class_weights, scale=args.ldam_scale)
    if args.loss == "balanced_softmax":
        return BalancedSoftmaxLoss(counts)
    raise ValueError(f"Unknown loss: {args.loss}")
