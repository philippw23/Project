"""t-SNE embedding visualization for bone-tumor malignancy classification.

Supports BiomedCLIP, CheXFound, LACE (v2), and ImageNet baseline encoders.

Usage examples:
    # BiomedCLIP checkpoint
    python src/data/visualize_tsne.py \\
        --encoder_type biomedclip \\
        --checkpoint results/biomedclip_pretrain/.../best.pt \\
        --splits data/internal_dataset/split.json

    # BiomedCLIP base weights (no checkpoint)
    python src/data/visualize_tsne.py \\
        --encoder_type biomedclip \\
        --splits data/internal_dataset/split.json

    # CheXFound baseline (no continued-pretrain checkpoint)
    python src/data/visualize_tsne.py \\
        --encoder_type chexfound \\
        --checkpoint none \\
        --chexfound_config src/chexfound/configs/vit_large.yaml \\
        --chexfound_weights data/chexfound_weights.pth \\
        --splits data/internal_dataset/split.json

    # CheXFound from continued-pretrain checkpoint
    python src/data/visualize_tsne.py \\
        --encoder_type chexfound \\
        --checkpoint results/chexfound_pretrain/.../best.pt \\
        --chexfound_config src/chexfound/configs/vit_large.yaml \\
        --splits data/internal_dataset/split.json

    # LACE v2
    python src/data/visualize_tsne.py \\
        --encoder_type lace \\
        --version v2 \\
        --checkpoint results/lace_v2_pretrain/.../best.pt \\
        --splits results/lace_v2_pretrain/.../splits.json

    # ImageNet ViT-B/16 baseline
    python src/data/visualize_tsne.py \\
        --encoder_type imagenet \\
        --splits data/internal_dataset/split.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biomedclip.data.datasets import DownstreamDataset, IDX_TO_LABEL
from biomedclip.utils.misc import DEFAULT_OUT_DIR, DEFAULT_SPLITS, MODEL_TAG


CLASS_COLORS = {
    "benign":       "#2ca02c",
    "intermediate": "#ff7f0e",
    "malignant":    "#d62728",
}


# ── Encoder loading ───────────────────────────────────────────────────────────

def load_biomedclip_encoder(args, device):
    import open_clip
    from biomedclip.models.lora import inject_lora

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_TAG)
    if args.checkpoint and args.checkpoint.lower() != "none":
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        lora_cfg = ckpt.get("lora_config") or {}
        if lora_cfg.get("lora_layers"):
            inject_lora(model, lora_cfg["lora_layers"], lora_cfg["lora_r"], lora_cfg["lora_alpha"])
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded BiomedCLIP checkpoint: {args.checkpoint}")
    else:
        print("Using BiomedCLIP base pretrained weights (no checkpoint)")

    encoder = model.visual.trunk.to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder, preprocess


def load_chexfound_encoder(args, device):
    from chexfound.data.transforms import build_preprocess_val_chexfound
    from chexfound.models.encoders import CheXFoundViT, load_continued_pretrain_weights

    preprocess = build_preprocess_val_chexfound()
    checkpoint = None if (args.checkpoint or "none").lower() == "none" else args.checkpoint

    if checkpoint is None:
        if not args.chexfound_weights:
            raise ValueError(
                "--chexfound_weights is required when --checkpoint none. "
                "Provide the path to the original CheXFound .pth file."
            )
        encoder = CheXFoundViT(
            args.chexfound_config, args.chexfound_weights,
            lora_layers=0, r=8, alpha=16.0, load_pretrained=True,
        )
        print("Loaded CheXFound baseline (original pretrained weights)")
    else:
        encoder = CheXFoundViT(
            args.chexfound_config, weights_path=None,
            lora_layers=0, r=8, alpha=16.0, load_pretrained=False,
        )
        load_continued_pretrain_weights(encoder, checkpoint)
        print(f"Loaded CheXFound from checkpoint: {checkpoint}")

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder, preprocess


def load_lace_encoder(args, device):
    from LACE.models.encoders import SharedViT
    from LACE.models.mask_tokens import MaskTokenDecoder, MaskPredictionHead
    from LACE.models.downstream import LACEv2Classifier

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora_config", {})
    if not lora_cfg:
        raise RuntimeError("Checkpoint has no 'lora_config'. Was it produced by LACE pretraining?")

    vit = SharedViT(
        lora_layers=lora_cfg["lora_layers"],
        r=lora_cfg["lora_r"],
        alpha=lora_cfg["lora_alpha"],
        embed_dim=lora_cfg.get("embed_dim", 512),
    )
    vit.load_state_dict(ckpt["vit_state"])
    vit = vit.to(device)

    mask_decoder = MaskTokenDecoder(
        n_tokens=ckpt.get("n_mask_tokens", 4),
        n_heads=ckpt.get("n_mask_heads", 8),
        sigma=ckpt.get("gauss_sigma", 1.5),
    )
    mask_decoder.load_state_dict(ckpt["mask_decoder_state"])
    mask_decoder = mask_decoder.to(device)

    mask_head = MaskPredictionHead()
    mask_head.load_state_dict(ckpt["mask_head_state"])
    mask_head = mask_head.to(device)

    # use_meta/n_classes/linear_head only shape the (unused-here) classification
    # head; only _get_visual() is called, so their values don't matter.
    classifier = LACEv2Classifier(
        vit=vit, mask_decoder=mask_decoder, mask_head=mask_head, use_meta=False,
    ).to(device)
    classifier.eval()

    print(f"Loaded LACE v2 checkpoint: {args.checkpoint}")
    return classifier, vit.preprocess_val


def load_imagenet_encoder(device):
    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    model.eval()

    config = resolve_data_config({}, model=model)
    preprocess = create_transform(**config)
    print("Loaded ImageNet ViT-B/16 (timm pretrained)")
    return model, preprocess


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_biomedclip_embeddings(encoder, loader, device):
    all_emb, all_lbl = [], []
    for batch in loader:
        feat = encoder.forward_features(batch["image"].to(device))  # [B, N+1, 768]
        cls = feat[:, 0]
        all_emb.append(F.normalize(cls, dim=-1).cpu())
        all_lbl.append(batch["label"])
    return torch.cat(all_emb).numpy(), torch.cat(all_lbl).numpy()


@torch.no_grad()
def extract_chexfound_embeddings(encoder, loader, device):
    all_emb, all_lbl = [], []
    for batch in loader:
        cls = encoder(batch["image"].to(device))  # [B, 1024]
        all_emb.append(F.normalize(cls, dim=-1).cpu())
        all_lbl.append(batch["label"])
    return torch.cat(all_emb).numpy(), torch.cat(all_lbl).numpy()


@torch.no_grad()
def extract_lace_embeddings(classifier, loader, device):
    all_emb, all_lbl = [], []
    for batch in loader:
        images = batch["image"].to(device)
        emb = classifier._get_visual(images)  # [B, 512] or [B, 1024] per visual_mode
        all_emb.append(F.normalize(emb, dim=-1).cpu())
        all_lbl.append(batch["label"])
    return torch.cat(all_emb).numpy(), torch.cat(all_lbl).numpy()


@torch.no_grad()
def extract_imagenet_embeddings(encoder, loader, device):
    all_emb, all_lbl = [], []
    for batch in loader:
        feat = encoder(batch["image"].to(device))  # [B, 768]
        all_emb.append(F.normalize(feat, dim=-1).cpu())
        all_lbl.append(batch["label"])
    return torch.cat(all_emb).numpy(), torch.cat(all_lbl).numpy()


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset(splits_path, split_name, preprocess, use_mask):
    with open(splits_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    all_samples    = raw["train"] + raw["val"] + raw["test"]
    age_sex_lookup = {Path(s["image"]).stem: (float(s["age"]), float(s["sex"])) for s in all_samples}
    train_ages     = [float(s["age"]) for s in raw["train"]]
    age_mean       = float(np.mean(train_ages))
    age_std        = float(np.std(train_ages))

    split_map = {"train": raw["train"], "val": raw["val"], "test": raw["test"], "all": all_samples}
    if split_name not in split_map:
        raise ValueError(f"Unknown split '{split_name}'. Choose from: train, val, test, all.")

    ds = DownstreamDataset(split_map[split_name], age_sex_lookup, age_mean, age_std, preprocess, use_mask)
    print(f"Dataset ({split_name}): {len(ds)} samples")
    return ds


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_tsne(coords, labels, title, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor("white")

    for idx, label_name in IDX_TO_LABEL.items():
        mask = labels == idx
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=CLASS_COLORS[label_name],
            label=label_name.capitalize(),
            alpha=0.6,
            s=20,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", framealpha=0.9)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="t-SNE embedding visualization")
    p.add_argument("--encoder_type", required=True,
                   choices=["biomedclip", "chexfound", "lace", "imagenet"])
    p.add_argument("--checkpoint", default=None,
                   help="Path to encoder checkpoint. Use 'none' for baseline (no fine-tuning).")
    p.add_argument("--splits", default=str(DEFAULT_SPLITS),
                   help="Path to splits.json (default: %(default)s)")
    p.add_argument("--split", default="all",
                   choices=["train", "val", "test", "all"],
                   help="Which data split(s) to visualize (default: all)")
    p.add_argument("--use_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="Crop images to lesion bounding box (default: True)")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--n_iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None,
                   help="Output PNG path (auto-generated from encoder_type + checkpoint if not set)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--chexfound_config", default=None,
                   help="Path to CheXFound model config YAML (required for chexfound encoder)")
    p.add_argument("--chexfound_weights", default=None,
                   help="Path to original CheXFound .pth weights (required when --checkpoint none)")
    p.add_argument("--version", default="v2", choices=["v2"],
                   help="LACE version (v1 was retired; only v2 remains).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | encoder: {args.encoder_type} | split: {args.split}")

    # ── Load encoder ──────────────────────────────────────────────────────────
    if args.encoder_type == "biomedclip":
        encoder, preprocess = load_biomedclip_encoder(args, device)
    elif args.encoder_type == "chexfound":
        encoder, preprocess = load_chexfound_encoder(args, device)
    elif args.encoder_type == "lace":
        encoder, preprocess = load_lace_encoder(args, device)
    else:  # imagenet
        encoder, preprocess = load_imagenet_encoder(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = load_dataset(args.splits, args.split, preprocess, args.use_mask)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )

    # ── Extract embeddings ────────────────────────────────────────────────────
    print("Extracting embeddings...")
    if args.encoder_type == "biomedclip":
        emb, labels = extract_biomedclip_embeddings(encoder, loader, device)
    elif args.encoder_type == "chexfound":
        emb, labels = extract_chexfound_embeddings(encoder, loader, device)
    elif args.encoder_type == "lace":
        emb, labels = extract_lace_embeddings(encoder, loader, device)
    else:  # imagenet
        emb, labels = extract_imagenet_embeddings(encoder, loader, device)
    print(f"Embeddings shape: {emb.shape}")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print(f"Running t-SNE (perplexity={args.perplexity}, n_iter={args.n_iter}, seed={args.seed})...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        max_iter=args.n_iter,
        random_state=args.seed,
        n_jobs=-1,
    )
    coords = tsne.fit_transform(emb)

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output is None:
        if args.checkpoint and args.checkpoint.lower() != "none":
            ckpt_stem = Path(args.checkpoint).stem
        else:
            ckpt_stem = "baseline"
        version_tag = f"_{args.version}" if args.encoder_type == "lace" else ""
        out_name = f"tsne_{args.encoder_type}{version_tag}_{ckpt_stem}.png"
        output_path = DEFAULT_OUT_DIR / out_name
    else:
        output_path = Path(args.output)

    # ── Plot ──────────────────────────────────────────────────────────────────
    ckpt_label = Path(args.checkpoint).name if (args.checkpoint and args.checkpoint.lower() != "none") else "baseline"
    encoder_label = args.encoder_type.upper() + (f" {args.version}" if args.encoder_type == "lace" else "")
    title = f"{encoder_label} — {ckpt_label} ({args.split} split)"
    plot_tsne(coords, labels, title, output_path)


if __name__ == "__main__":
    main()
