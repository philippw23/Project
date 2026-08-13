"""Visualize the LACE v2 fg mask-token attention against a frozen pretrain checkpoint.

The downstream head (LACEv2Classifier) freezes mask_decoder/mask_head verbatim
from the pretrain checkpoint (see LACE/train/downstream.py::build_v2_model) —
so whatever the fg token attends to at pretrain-checkpoint time is exactly what
the downstream classifier sees. This script reuses LACE.train.pretrain_v2's
visualize_heatmaps() against a chosen split (e.g. a CV fold's val/test set) to
inspect that directly, without needing a downstream run at all.

Usage:
    python src/visualize_fg_attention.py \\
        --checkpoint results/lace_v2_pretrain/run_.../fold0/best_retrieval_checkpoint.pt \\
        --split_file results/lace_v2_pretrain/run_.../fold0/split.json \\
        --split_key test \\
        --out_dir results/fg_attention_viz/fold0_test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from PIL import Image

from biomedclip.data.transforms import build_preprocess_val
from LACE.data.transforms import crop_around_mask_pair, pad_to_square
from LACE.models.mask_tokens import MaskPredictionHead, MaskTokenDecoder
from LACE.train.downstream import _load_vit
from LACE.train.pretrain_v2 import visualize_heatmaps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Pretrain checkpoint (best_checkpoint.pt / best_retrieval_checkpoint.pt) "
                        "— the same one a downstream/CV run for this fold would load.")
    p.add_argument("--split_file", required=True,
                   help="Split JSON with train/val/test keys (a fold's split.json, or split.json).")
    p.add_argument("--split_key", default="test", choices=["train", "val", "test"])
    p.add_argument("--n_samples", type=int, default=20,
                   help="Max number of samples with a non-empty GT mask to visualize.")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--context_fraction", type=float, default=0.15,
                   help="Must match the crop used at pretrain time for a fair look "
                        "(pretrain_v2 default is 0.15).")
    p.add_argument("--context_mode", default="image", choices=["image", "lesion"])
    p.add_argument("--min_crop_size", type=int, default=224)
    p.add_argument("--out_dir", required=True,
                   help="Heatmap PNGs are written to <out_dir>/heatmaps/epoch_000/.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vit = _load_vit(ckpt, device)
    vit.eval()

    mask_decoder = MaskTokenDecoder(
        n_tokens=ckpt.get("n_mask_tokens", 4),
        n_heads=ckpt.get("n_mask_heads", 8),
        sigma=ckpt.get("gauss_sigma", 1.5),
    )
    mask_decoder.load_state_dict(ckpt["mask_decoder_state"])
    mask_decoder.to(device).eval()

    mask_head = MaskPredictionHead()
    mask_head.load_state_dict(ckpt["mask_head_state"])
    mask_head.to(device).eval()

    print(
        f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
        f"n_mask_tokens={ckpt.get('n_mask_tokens', '?')}) — "
        f"same weights a downstream run for this fold would freeze."
    )

    preprocess_val = build_preprocess_val(vit.preprocess_val, args.image_size)

    with open(args.split_file, encoding="utf-8") as fh:
        samples = json.load(fh)[args.split_key]
    print(f"{len(samples)} samples in split '{args.split_key}'")

    vis_samples: list[dict] = []
    for s in samples:
        if len(vis_samples) >= args.n_samples:
            break
        mask_path = Path(s["mask"])
        if not mask_path.exists():
            continue
        mask_arr = np.array(Image.open(mask_path).convert("L"), dtype=float)
        if not np.any(mask_arr > 0):
            continue
        img_path = Path(s["image"])
        img_arr = np.array(Image.open(img_path).convert("RGB"))
        if args.context_fraction > 0:
            img_crop_arr, crop_mask_arr = crop_around_mask_pair(
                img_arr, mask_arr, context_fraction=args.context_fraction,
                context_mode=args.context_mode, min_crop_size=args.min_crop_size,
            )
        else:
            img_crop_arr = pad_to_square(img_arr)
            crop_mask_arr = pad_to_square(mask_arr)
        vis_samples.append({
            "img_path":   img_path,
            "img_crop":   img_crop_arr,
            "img_tensor": preprocess_val(Image.fromarray(img_crop_arr)),
            "mask_arr":   (crop_mask_arr > 0),
        })
    print(f"Visualizing {len(vis_samples)} samples with a non-empty GT mask")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visualize_heatmaps(vit, mask_decoder, mask_head, vis_samples, epoch=0, run_dir=out_dir, device=device)
    print(f"Saved to {out_dir / 'heatmaps' / 'epoch_000'}")


if __name__ == "__main__":
    main()
