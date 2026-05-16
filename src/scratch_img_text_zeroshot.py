"""Zero-shot malignancy classification using TinyTextTransformer.

Loads a checkpoint produced by scratch_img_text_downstream.py (which saves the
text encoder and both projection heads alongside the image encoder).  Zero-shot
predictions compare L2-normalised image projections against class prototype text
projections via cosine similarity.

Expected near-random performance — included for consistency with the BiomedCLIP
zero-shot evaluation.

Usage:
    python src/scratch_img_text_zeroshot.py \\
        --checkpoint results/.../best_checkpoint.pt \\
        --splits     results/.../splits.json \\
        --excel      data/internal_dataset/metadata.xlsx
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.utils.misc import DEFAULT_SPLITS
from biomedclip.data.datasets import DownstreamDataset
from biomedclip.eval.zeroshot import build_class_embeddings, zeroshot_eval
from scratch_img.data.transforms import build_val_transform
from scratch_img.models.encoders import build_encoder
from scratch_img_text.models.text_encoder import (
    TinyTextTransformer, build_tokenizer, get_vocab_size,
)


def extract_projected_embeddings(
    encoder: nn.Module,
    img_proj: nn.Linear,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return (N, proj_dim) L2-normalised image projections and integer labels."""
    encoder.eval()
    img_proj.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            proj   = F.normalize(img_proj(encoder(images)), dim=-1)
            all_emb.append(proj.cpu())
            all_lbl.append(batch["label"])
    return torch.cat(all_emb), torch.cat(all_lbl).numpy()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation with TinyTextTransformer from scratch_img_text checkpoint."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_checkpoint.pt from scratch_img_text_downstream.py.")
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS), help="Path to split.json.")
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--wandb",        action="store_true")
    parser.add_argument("--wandb_project", default="scratch-img-text-zeroshot")
    parser.add_argument("--wandb_run",    default=None)
    parser.add_argument("--wandb_entity", default=None)
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        warnings.warn("--wandb set but wandb is not installed. Skipping.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            config=vars(args),
        )

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for key in ("text_enc_state_dict", "img_proj_state_dict", "txt_proj_state_dict"):
        if key not in ckpt:
            raise RuntimeError(
                f"Checkpoint '{args.checkpoint}' is missing '{key}'. "
                "Re-run scratch_img_text_downstream.py to generate a checkpoint that "
                "includes the text encoder and projection heads."
            )

    saved_args   = ckpt.get("args", {})
    encoder_name = saved_args.get("encoder", "resnet18")
    proj_dim     = saved_args.get("proj_dim", 256)
    text_hidden  = saved_args.get("text_hidden_dim", 256)
    text_layers  = saved_args.get("text_n_layers",   4)
    text_heads   = saved_args.get("text_n_heads",    4)
    max_text_len = saved_args.get("max_text_len",    128)

    # ── Image encoder + projection head ──────────────────────────────────────
    encoder, embed_dim = build_encoder(encoder_name)
    encoder = encoder.to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    img_proj = nn.Linear(embed_dim, proj_dim, bias=False).to(device)
    img_proj.load_state_dict(ckpt["img_proj_state_dict"])
    for p in img_proj.parameters():
        p.requires_grad_(False)
    img_proj.eval()

    print(f"Image encoder: {encoder_name} ({embed_dim}-dim) → proj {proj_dim}-dim")

    # ── Text encoder + projection head ────────────────────────────────────────
    tokenizer  = build_tokenizer()
    vocab_size = get_vocab_size(tokenizer)
    text_enc   = TinyTextTransformer(
        vocab_size=vocab_size, hidden_dim=text_hidden,
        n_layers=text_layers, n_heads=text_heads, max_len=max_text_len,
    ).to(device)
    text_enc.load_state_dict(ckpt["text_enc_state_dict"])
    for p in text_enc.parameters():
        p.requires_grad_(False)
    text_enc.eval()

    txt_proj = nn.Linear(text_hidden, proj_dim, bias=False).to(device)
    txt_proj.load_state_dict(ckpt["txt_proj_state_dict"])
    for p in txt_proj.parameters():
        p.requires_grad_(False)
    txt_proj.eval()

    print(f"Loaded checkpoint: {args.checkpoint} "
          f"(epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', float('nan')):.4f})")

    # ── Build class prototype text embeddings ─────────────────────────────────
    def tokenize_fn(prompts: list[str]) -> torch.Tensor:
        enc = tokenizer(
            prompts,
            max_length=max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Stack into (B, 2, L) so build_class_embeddings receives a single tensor.
        return torch.stack([enc["input_ids"], enc["attention_mask"]], dim=1)

    def encode_text_fn(tokens: torch.Tensor) -> torch.Tensor:
        input_ids      = tokens[:, 0, :].to(device)
        attention_mask = tokens[:, 1, :].to(device)
        feat = text_enc(input_ids, attention_mask)   # (B, text_hidden)
        return txt_proj(feat)                         # (B, proj_dim)

    print("Building class embeddings from prompt templates...")
    class_matrix = build_class_embeddings(encode_text_fn, tokenize_fn, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    with open(args.splits, encoding="utf-8") as fh:
        raw = json.load(fh)
    splits = {
        "train": raw["train"],
        "val":   raw["val"],
        "test":  raw["test"],
    }

    all_samples    = splits["train"] + splits["val"] + splits["test"]
    age_sex_lookup = {Path(s["image"]).stem: (float(s["age"]), float(s["sex"])) for s in all_samples}
    train_ages     = [s["age"] for s in splits["train"]]
    age_mean = float(np.mean(train_ages))
    age_std  = float(np.std(train_ages))

    preprocess_val = build_val_transform()
    val_ds  = DownstreamDataset(splits["val"],  age_sex_lookup, age_mean, age_std,
                                 preprocess_val, args.use_mask)
    test_ds = DownstreamDataset(splits["test"], age_sex_lookup, age_mean, age_std,
                                 preprocess_val, args.use_mask)
    print(f"Samples — val: {len(val_ds)}, test: {len(test_ds)}")

    use_pin       = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}
    val_loader    = DataLoader(val_ds,  shuffle=False, **loader_kwargs)
    test_loader   = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # ── Zero-shot evaluation ──────────────────────────────────────────────────
    print("\nEmbedding val split...")
    val_emb, val_lbl = extract_projected_embeddings(encoder, img_proj, val_loader, device)
    val_metrics = zeroshot_eval(val_emb, val_lbl, class_matrix, split_name="val")

    print("\nEmbedding test split...")
    test_emb, test_lbl = extract_projected_embeddings(encoder, img_proj, test_loader, device)
    test_metrics = zeroshot_eval(test_emb, test_lbl, class_matrix, split_name="test")

    if use_wandb:
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()})
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
