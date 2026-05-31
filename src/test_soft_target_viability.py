#!/usr/bin/env python3
"""Soft-target viability test for LACE v1 — first training batch.

Loads one batch through the exact LACE v1 pipeline (SharedViT + BiomedCLIPTextEncoder
with pretrained projections, epoch-0 state) and produces similarity heatmaps for:

  1. Full image ↔ full image
  2. Crop ↔ crop
  3. Beurteilung phrase ↔ phrase  (grouped by source image)
  4. Befund phrase ↔ phrase       (grouped by source image)
  5. Beurteilung full text ↔ full text  (mean-pooled phrases, one vec per image)
  6. Befund full text ↔ full text       (mean-pooled phrases, one vec per image)

The image heatmaps show whether image embeddings are peaked enough to serve
as soft-target anchors for a symmetric loss alongside the existing text-text targets.
Heatmaps 5 & 6 show whether the full-section representations are discriminative
across images (B×B, one entry per batch sample).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from LACE.data.splits import build_pretrain_datasets_lace
from LACE.data.transforms import build_train_transform_lace
from LACE.loss.objectives import phrase_soft_targets
from LACE.models.encoders import BiomedCLIPTextEncoder, SharedViT
from biomedclip.utils.misc import DEFAULT_DATASET_JSON, DEFAULT_SPLITS


def soft_target_entropy(sim: np.ndarray, tau: float | None) -> np.ndarray:
    """Row-wise entropy of softmax(sim/tau), diagonal excluded."""
    logits = sim.copy()
    np.fill_diagonal(logits, -np.inf)
    if tau is not None:
        logits /= tau
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return -(probs * np.log(probs + 1e-12)).sum(axis=1)


def similarity_heatmap(
    emb: torch.Tensor,
    tau_s: float,
    title: str,
    group_boundaries: list[int] | None = None,
    same_image_boost: float = 0.0,
    src_indices: torch.Tensor | None = None,
) -> tuple:
    """2-panel figure: raw cosine similarity + per-row normalised soft-target heatmap.

    Uses phrase_soft_targets from LACE objectives for the soft-target panel so the
    visualised distribution is identical to what the training loss computes.
    """
    sim = (emb @ emb.T).numpy()

    with torch.no_grad():
        soft_probs = phrase_soft_targets(
            emb, tau_s, src_indices, same_image_boost
        ).numpy()
    # Mask diagonal before computing per-row scale so the self-similarity entry
    # (which always dominates) does not collapse all off-diagonal variation to zero.
    mask = soft_probs.copy()
    np.fill_diagonal(mask, np.nan)
    row_min = np.nanmin(mask, axis=1, keepdims=True)
    row_max = np.nanmax(mask, axis=1, keepdims=True)
    soft_norm = (mask - row_min) / np.where(row_max > row_min, row_max - row_min, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(sim, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title(f"{title} — raw cosine  (N={sim.shape[0]})")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Index")

    im1 = axes[1].imshow(soft_norm, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04,
                 label="per-row normalised probability")
    boost_tag = f"  gamma={same_image_boost}" if same_image_boost > 0.0 else ""
    axes[1].set_title(f"{title} — soft-targets  tau_s={tau_s}{boost_tag}")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Index")

    if group_boundaries:
        for b in group_boundaries[1:-1]:
            for ax in axes:
                ax.axhline(b - 0.5, color="black", linewidth=0.8)
                ax.axvline(b - 0.5, color="black", linewidth=0.8)

    fig.tight_layout()
    return fig, axes


def _phrase_group_boundaries(src: np.ndarray) -> list[int]:
    """Boundary indices where the source image index changes."""
    bounds = [0]
    for i in range(1, len(src)):
        if src[i] != src[i - 1]:
            bounds.append(i)
    bounds.append(len(src))
    return bounds


def print_entropy(sim: np.ndarray, tau_s: float, label: str) -> None:
    N = sim.shape[0]
    h_raw   = soft_target_entropy(sim, tau=None)
    h_sharp = soft_target_entropy(sim, tau=tau_s)
    h_unif  = np.log(max(N - 1, 1))
    print(f"  {label}: N={N}")
    print(f"    raw cosine — H={h_raw.mean():.3f}  ({h_raw.mean()/h_unif:.1%} of uniform)")
    print(f"    τ_s={tau_s}   — H={h_sharp.mean():.3f}  ({h_sharp.mean()/h_unif:.1%} of uniform)")


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── LACE v1 models — epoch 0 state ───────────────────────────────────────
    print("Initializing LACE v1 models (epoch 0, pretrained projections)…")
    vit      = SharedViT(args.lora_layers, args.lora_r, args.lora_alpha, args.embed_dim)
    text_enc = BiomedCLIPTextEncoder(embed_dim=args.embed_dim)
    vit.load_pretrained_projections()
    text_enc.load_pretrained_projections()
    vit      = vit.to(device).eval()
    text_enc = text_enc.to(device).eval()

    preprocess_train = build_train_transform_lace(vit.preprocess_val)
    tokenizer        = text_enc.tokenizer

    # ── Train dataset ─────────────────────────────────────────────────────────
    splits_path = Path(args.splits)
    if splits_path.exists():
        with open(splits_path, encoding="utf-8") as fh:
            pretrain_samples = json.load(fh)["train"]
        print(f"Loaded {len(pretrain_samples)} train samples from {splits_path}")
    else:
        with open(args.dataset, encoding="utf-8") as fh:
            pretrain_samples = json.load(fh)
        print(f"Splits not found — using all {len(pretrain_samples)} samples from dataset")

    train_ds, _ = build_pretrain_datasets_lace(
        pretrain_samples, preprocess_train, vit.preprocess_val,
        tokenizer, args.seed,
        text_mode="phrase",
        max_bef_phrases=16,
        max_beur_phrases=16,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, generator=g,
    )
    batch = next(iter(loader))
    B = batch["full_image"].shape[0]
    print(f"First batch: {B} samples")

    # ── Encode ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        full_embs = vit.forward_cls(batch["full_image"].to(device)).cpu()
        crop_embs = vit.forward_cls(batch["crop_image"].to(device)).cpu()

        beur_pmask = batch["beur_phrase_mask"]
        bef_pmask  = batch["bef_phrase_mask"]

        beur_embs = text_enc._encode_phrase_batch(
            batch["beur_phrase_ids"].to(device),
            batch["beur_phrase_attn"].to(device),
            beur_pmask.to(device),
        ).cpu()
        bef_embs = text_enc._encode_phrase_batch(
            batch["bef_phrase_ids"].to(device),
            batch["bef_phrase_attn"].to(device),
            bef_pmask.to(device),
        ).cpu()

    # ── Full-text encoding (befund_en / beurteilung_en) ───────────────────────
    # Reconstruct which samples landed in the phrase batch by replaying the same
    # randperm: DataLoader(shuffle=True, generator=g) calls randperm(N, generator)
    # internally, so the same seed gives the same order.
    g2 = torch.Generator()
    g2.manual_seed(args.seed)
    perm = torch.randperm(len(train_ds), generator=g2)
    batch_samples = [train_ds.samples[i.item()] for i in perm[:B]]

    def _tok_full(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        hf_tok = getattr(tokenizer, "tokenizer", tokenizer)
        enc = hf_tok(
            texts,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    beur_texts = [str(s.get("beurteilung_en") or "").strip() or "[PAD]" for s in batch_samples]
    bef_texts  = [str(s.get("befund_en") or "").strip() or "[PAD]"  for s in batch_samples]

    beur_ids, beur_attn_full = _tok_full(beur_texts)
    bef_ids,  bef_attn_full  = _tok_full(bef_texts)

    with torch.no_grad():
        beur_full_embs = text_enc.encode_beurteilung(
            beur_ids.to(device), beur_attn_full.to(device)
        ).cpu()
        bef_full_embs = text_enc.encode_befund(
            bef_ids.to(device), bef_attn_full.to(device)
        )[0].cpu()   # encode_befund returns (z_cls, proj_phrases)

    # ── flatten valid phrases + track source image index
    beur_valid = beur_embs[beur_pmask]
    bef_valid  = bef_embs[bef_pmask]

    img_idx  = torch.arange(B).unsqueeze(1).expand(-1, beur_pmask.shape[1])
    beur_src = img_idx[beur_pmask]
    img_idx  = torch.arange(B).unsqueeze(1).expand(-1, bef_pmask.shape[1])
    bef_src  = img_idx[bef_pmask]

    tau_img  = args.tau_s_img
    tau_beur = args.tau_s_beur
    tau_bef  = args.tau_s_bef

    # ── Entropy stats ─────────────────────────────────────────────────────────
    print("\nEntropy:")
    print_entropy((full_embs      @ full_embs.T).numpy(),      tau_img,  "Full image")
    print_entropy((crop_embs      @ crop_embs.T).numpy(),      tau_img,  "Crop image")
    print_entropy((beur_valid     @ beur_valid.T).numpy(),     tau_beur, "Beurteilung phrases")
    print_entropy((bef_valid      @ bef_valid.T).numpy(),      tau_bef,  "Befund phrases")
    print_entropy((beur_full_embs @ beur_full_embs.T).numpy(), tau_beur, "Beurteilung full text")
    print_entropy((bef_full_embs  @ bef_full_embs.T).numpy(),  tau_bef,  "Befund full text")

    # ── Figures ───────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boost = args.same_image_boost
    specs = [
        (full_embs,      "Full image",            "full_image_heatmap.png",    tau_img,  None, None),
        (crop_embs,      "Crop image",            "crop_image_heatmap.png",    tau_img,  None, None),
        (beur_valid,     "Beurteilung phrases",   "beur_phrase_heatmap.png",   tau_beur,
         _phrase_group_boundaries(beur_src.numpy()), beur_src),
        (bef_valid,      "Befund phrases",        "bef_phrase_heatmap.png",    tau_bef,
         _phrase_group_boundaries(bef_src.numpy()),  bef_src),
        (beur_full_embs, "Beurteilung full text", "beur_fulltext_heatmap.png", tau_beur, None, None),
        (bef_full_embs,  "Befund full text",      "bef_fulltext_heatmap.png",  tau_bef,  None, None),
    ]
    for emb, title, fname, tau_s, bounds, src in specs:
        fig, _ = similarity_heatmap(
            emb, tau_s, title, group_boundaries=bounds,
            same_image_boost=boost, src_indices=src,
        )
        p = out_dir / fname
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"Saved → {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LACE v1 soft-target viability — first training batch"
    )
    parser.add_argument("--splits",      default=str(DEFAULT_SPLITS),
                        help="Path to split.json (default: data/internal_dataset/split.json)")
    parser.add_argument("--dataset",     default=str(DEFAULT_DATASET_JSON),
                        help="Fallback dataset path if splits not found")
    parser.add_argument("--out_dir",     default=str(ROOT_DIR / "results" / "soft_target_viability"))
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--tau_s_img",  type=float, default=0.015,
                        help="Soft-target temperature for image heatmaps (default: 0.015)")
    parser.add_argument("--tau_s_beur", type=float, default=0.015,
                        help="Soft-target temperature for Beurteilung phrases (default: 0.015)")
    parser.add_argument("--tau_s_bef",  type=float, default=0.07,
                        help="Soft-target temperature for Befund phrases (default: 0.07)")
    parser.add_argument("--lora_layers", type=int,   default=4)
    parser.add_argument("--lora_r",      type=int,   default=8)
    parser.add_argument("--lora_alpha",  type=float, default=16.0)
    parser.add_argument("--embed_dim",   type=int,   default=512)
    parser.add_argument("--same_image_boost", type=float, default=0.0,
                        help="Logit boost added to same-image phrase pairs before softmax (0.0 = no boost)")
    parser.add_argument("--seed",        type=int,   default=42)
    main(parser.parse_args())
