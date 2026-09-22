"""BiomedCLIP zero-shot malignancy classification.

Evaluates cosine-similarity zero-shot classification using a 5-template prompt
ensemble for each class (benign / intermediate / malignant).  Supports:
  - Vanilla BiomedCLIP (no checkpoint)
  - LoRA fine-tuned checkpoint
  - Partial fine-tune checkpoint (--no_lora variant from biomedclip/train/pretrain.py)

Usage (vanilla encoder):
    python src/biomedclip_zeroshot.py \\
        --freezed_biomedclip \\
        --splits results/.../splits.json \\
        --excel  data/internal_dataset/metadata.xlsx

Usage (fine-tuned checkpoint):
    python src/biomedclip_zeroshot.py \\
        --checkpoint results/biomedclip_pretrain/.../best_r1_checkpoint.pt \\
        --splits     results/.../splits.json \\
        --excel      data/internal_dataset/metadata.xlsx
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.utils.misc import ROOT_DIR, MODEL_TAG, DEFAULT_OUT_DIR, DEFAULT_SPLITS
from biomedclip.data.datasets import DownstreamDataset
from biomedclip.models.lora import inject_lora
from biomedclip.eval.zeroshot import build_class_embeddings, zeroshot_eval
DEFAULT_CHECKPOINT = ROOT_DIR / "results" / "biomedclip_pretrain" / "best_r1_checkpoint.pt"


def extract_image_embeddings(
    visual: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Run visual encoder over a split, return (N, dim) L2-normalised embeddings and labels."""
    visual.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            emb    = F.normalize(visual(images), dim=-1)
            all_emb.append(emb.cpu())
            all_lbl.append(batch["label"])
    return torch.cat(all_emb), torch.cat(all_lbl).numpy()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BiomedCLIP zero-shot malignancy classification with prompt ensemble."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="Path to BiomedCLIP pretrain checkpoint.")
    parser.add_argument("--freezed_biomedclip", action="store_true",
                        help="Use vanilla BiomedCLIP without a fine-tuned checkpoint.")
    parser.add_argument("--splits",   default=str(DEFAULT_SPLITS))
    parser.add_argument("--out_dir",  default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--wandb",       action="store_true")
    parser.add_argument("--wandb_project", default="biomedclip-zeroshot")
    parser.add_argument("--wandb_run",   default=None)
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

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_TAG}")
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)
    tokenizer = open_clip.get_tokenizer(MODEL_TAG)

    if args.freezed_biomedclip:
        print("Using vanilla BiomedCLIP (no checkpoint).")
    else:
        ckpt     = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        lora_cfg = ckpt.get("lora_config") or {}
        if lora_cfg.get("no_lora"):
            print(f"Partial fine-tune checkpoint (unfreeze_blocks={lora_cfg['unfreeze_blocks']}, no LoRA).")
        elif lora_cfg:
            inject_lora(model, lora_cfg["lora_layers"], lora_cfg["lora_r"], lora_cfg["lora_alpha"])
            print(f"LoRA injected: layers={lora_cfg['lora_layers']}, r={lora_cfg['lora_r']}")
        else:
            print("No lora_config in checkpoint — loading weights directly.")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Build class prototype embeddings ──────────────────────────────────────
    print("Building class embeddings from prompt templates...")
    class_matrix = build_class_embeddings(
        encode_text_fn=model.encode_text,
        tokenize_fn=tokenizer,
        device=device,
    )

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
    # Use the full visual tower (model.visual) so the projected 512-dim CLIP
    # embedding is in the same space as the text embeddings.
    visual = model.visual

    print("\nEmbedding val split...")
    val_emb, val_lbl = extract_image_embeddings(visual, val_loader, device)
    val_metrics = zeroshot_eval(val_emb, val_lbl, class_matrix, split_name="val")

    print("\nEmbedding test split...")
    test_emb, test_lbl = extract_image_embeddings(visual, test_loader, device)
    test_metrics = zeroshot_eval(test_emb, test_lbl, class_matrix, split_name="test")

    if use_wandb:
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()})
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
