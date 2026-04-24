"""BiomedCLIP contrastive pretraining with LoRA on the ViT image encoder.

Domain-adapts microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 on
(X-ray image, German radiology report) pairs from the bone-tumor dataset.
LoRA is injected into the last N transformer blocks of the ViT; the text
encoder (PubMedBERT) is kept fully frozen throughout.

Usage example:
    python src/biomedclip_pretrain.py \\
        --excel data/metadata.xlsx \\
        --reports data/text/sanitized_reports.json \\
        --use_mask --lora_layers 4 --epochs 50
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.utils.misc import (
    MODEL_TAG,
    DEFAULT_IMAGES_DIR,
    DEFAULT_MASKS_DIR,
    DEFAULT_EXCEL,
    DEFAULT_REPORTS,
    DEFAULT_OUT_DIR,
    save_checkpoint,
    print_biomedclip_architecture,
)
from biomedclip.data.transforms import build_train_transform
from biomedclip.data.splits import build_stratified_splits, build_pretrain_datasets
from biomedclip.models.lora import inject_lora, count_trainable_params
from biomedclip.loss.contrastive import clip_loss
from biomedclip.eval.retrieval import evaluate, evaluate_retrieval


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up followed by cosine annealing to zero.

    During the first warmup_epochs epochs the lr rises linearly from 0 to peak.
    Afterwards it follows a cosine curve that reaches 0 at epoch total_epochs.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linearly ramp up: fraction of warmup complete.
            return float(epoch + 1) / max(1, warmup_epochs)
        # Cosine decay from 1.0 → 0.0 over the remaining epochs.
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    trainable_params: list,
    max_grad_norm: float = 1.0,
) -> float:
    """Run one training epoch and return the average batch loss.

    open_clip exposes encode_image() and encode_text() as named methods rather
    than a single forward(images, texts), so the training loop calls those
    methods directly on the model.
    """
    model.train()
    total_loss = 0.0

    # Disable progress bar when stdout is not a terminal (e.g. SLURM log file).
    pbar = tqdm(loader, desc="  train", leave=False, disable=not sys.stdout.isatty())
    for batch in pbar:
        images = batch["image"].to(device)
        texts  = batch["text"].to(device)

        optimizer.zero_grad()

        # Mixed-precision forward: compute activations in float16 to save memory.
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            image_feat = model.encode_image(images)   # (B, embed_dim)
            text_feat  = model.encode_text(texts)     # (B, embed_dim)
            loss       = clip_loss(image_feat, text_feat, model.logit_scale)

        # GradScaler multiplies the loss before backward to prevent float16 underflow,
        # then divides the gradients back before the optimizer step.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Gradient clipping prevents occasional large updates from destabilising training.
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BiomedCLIP contrastive pretraining with LoRA on the ViT encoder."
    )
    parser.add_argument("--excel",   default=str(DEFAULT_EXCEL),
                        help="Path to metadata.xlsx (default: %(default)s)")
    parser.add_argument("--reports", default=str(DEFAULT_REPORTS),
                        help="Path to reports JSON (default: %(default)s)")
    parser.add_argument("--english", action="store_true",
                        help="Read befund_en/beurteilung_en instead of befund/beurteilung "
                             "(use with translated_reports.json).")
    parser.add_argument("--images",  default=str(DEFAULT_IMAGES_DIR),
                        help="Directory containing image PNGs (default: %(default)s)")
    parser.add_argument("--masks",   default=str(DEFAULT_MASKS_DIR),
                        help="Directory containing segmentation mask PNGs (default: %(default)s)")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR),
                        help="Output directory for checkpoints (default: %(default)s)")
    parser.add_argument("--print_architecture", action="store_true",
                        help="Load BioMedCLIP, print image/text encoder architecture, then exit")
    parser.add_argument("--print_full_model", action="store_true",
                        help="With --print_architecture, also print the full model wrapper")
    parser.add_argument("--lora_layers", type=int, default=4,
                        help="Number of last ViT transformer blocks to apply LoRA to (default: %(default)s)")
    parser.add_argument("--lora_r",     type=int,   default=8,
                        help="LoRA rank r (default: %(default)s)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha scaling factor (default: %(default)s)")
    parser.add_argument("--use_mask",  action="store_true",
                        help="Crop images around the lesion using segmentation masks")
    parser.add_argument("--batch_size", type=int,   default=32,
                        help="Training batch size (default: %(default)s)")
    parser.add_argument("--epochs",     type=int,   default=50,
                        help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--lr",          type=float, default=5e-4,
                        help="Peak learning rate for AdamW (default: %(default)s)")
    parser.add_argument("--weight_decay", type=float, default=0.2,
                        help="AdamW weight decay for non-norm parameters (default: %(default)s)")
    parser.add_argument("--patience",    type=int,   default=20,
                        help="Early stopping patience in epochs based on mean R@1 (0 to disable, default: %(default)s)")
    parser.add_argument("--downstream_train_frac", type=float, default=0.8,
                        help="Fraction of downstream data for classifier training (default: %(default)s)")
    parser.add_argument("--downstream_val_frac",   type=float, default=0.1,
                        help="Fraction of downstream data for classifier validation (default: %(default)s)")
    parser.add_argument("--test_frac",             type=float, default=0.1,
                        help="Fraction of downstream data held out for final evaluation (default: %(default)s)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for reproducibility (default: %(default)s)")
    parser.add_argument("--wandb",       action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="biomedclip-pretrain",
                        help="W&B project name (default: %(default)s)")
    parser.add_argument("--wandb_run",   default=None,
                        help="W&B run name (default: auto-generated)")
    parser.add_argument("--wandb_entity", default=None,
                        help="W&B team/entity name (default: personal account)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run as wandb sweep agent (hyperparams come from wandb.config)")
    return parser.parse_args(argv)


def _apply_sweep_config(args: argparse.Namespace) -> None:
    """Overwrite args with values from wandb.config when running as sweep agent."""
    cfg = wandb.config
    for key in ("lora_layers", "lora_r", "lr", "weight_decay", "batch_size"):
        if key in cfg:
            setattr(args, key, cfg[key])
    # Keep alpha proportional to rank (a common LoRA convention).
    args.lora_alpha = 2.0 * args.lora_r


def main(args: argparse.Namespace) -> None:
    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Optional architecture inspection ──────────────────────────────────────
    if args.print_architecture:
        print_biomedclip_architecture(
            model_tag=MODEL_TAG,
            device=device,
            include_full_model=args.print_full_model,
        )
        return

    # ── Model loading ─────────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_TAG}")
    # open_clip returns (model, train_preprocess, val_preprocess).
    # We override train_preprocess with a custom augmentation pipeline below.
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)
    tokenizer = open_clip.get_tokenizer(MODEL_TAG)
    model = model.to(device)
    # Adds random horizontal flip, colour jitter, and random resized crop on top of val_preprocess.
    preprocess_train = build_train_transform(preprocess_val)

    # Freeze all parameters, then inject trainable LoRA branches into the last
    # lora_layers ViT blocks and unfreeze logit_scale and visual.proj.
    inject_lora(model, args.lora_layers, args.lora_r, args.lora_alpha)
    n_trainable = count_trainable_params(model)
    n_total     = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA injected into last {args.lora_layers} ViT blocks "
        f"(r={args.lora_r}, alpha={args.lora_alpha})"
    )
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f} %)")

    # ── Output directory ──────────────────────────────────────────────────────
    run_dir = Path(args.out_dir) / "biomedclip_pretrain" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ── Data splits and datasets ───────────────────────────────────────────────
    # Stratified split by malignancy label; splits.json is written to run_dir.
    pretrain_samples, _, _, _ = build_stratified_splits(args, run_dir=run_dir)

    # train_ds applies preprocess_train (with augmentations); val_ds uses preprocess_val.
    train_ds, val_ds = build_pretrain_datasets(
        pretrain_samples, preprocess_train, preprocess_val, tokenizer, args.use_mask, args.seed
    )

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory,
        drop_last=len(train_ds) > args.batch_size,  # discard a tiny last batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=use_pin_memory,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Norm and bias parameters are excluded from weight decay because they
    # act as scale/shift terms and should not be penalised toward zero.
    no_decay_suffixes = ("bias", "norm.weight", "norm.bias", "ln_1.weight", "ln_1.bias",
                         "ln_2.weight", "ln_2.bias", "ln_pre.weight", "ln_pre.bias",
                         "ln_post.weight", "ln_post.bias")
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(no_decay_suffixes):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    trainable_params = decay_params + no_decay_params
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.98),  # standard CLIP betas
        eps=1e-6,
    )

    # Warm-up for the first 20 % of epochs, then cosine decay to 0.
    warmup_epochs = max(1, args.epochs // 5)
    scheduler     = make_scheduler(optimizer, warmup_epochs, args.epochs)
    # GradScaler prevents float16 gradient underflow during mixed-precision training.
    scaler        = torch.amp.GradScaler("cuda") if device.type == "cuda" else torch.amp.GradScaler("cpu")

    # ── W&B initialisation ────────────────────────────────────────────────────
    use_wandb = (args.wandb or args.sweep) and WANDB_AVAILABLE
    if (args.wandb or args.sweep) and not WANDB_AVAILABLE:
        warnings.warn("--wandb/--sweep set but wandb is not installed. Skipping.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            config={
                "lora_layers": args.lora_layers,
                "lora_r":      args.lora_r,
                "lora_alpha":  args.lora_alpha,
                "epochs":      args.epochs,
                "batch_size":  args.batch_size,
                "lr":          args.lr,
                "weight_decay": args.weight_decay,
                "seed":        args.seed,
                "use_mask":    args.use_mask,
            },
        )
        if args.sweep:
            # Overwrite CLI args with sweep-agent-supplied hyperparameters.
            _apply_sweep_config(args)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss     = float("inf")
    best_mean_r1      = 0.0          # average of I2T R@1 and T2I R@1
    epochs_no_improve = 0            # consecutive epochs without R@1 improvement
    lora_config = {"lora_layers": args.lora_layers, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha}

    print(f"\nStarting training for {args.epochs} epochs (warmup: {warmup_epochs})\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, trainable_params)
        # evaluate() computes the symmetric CLIP val loss without gradient updates.
        val_loss   = evaluate(model, val_loader, device)
        scheduler.step()

        lr_current  = scheduler.get_last_lr()[0]
        logit_scale = model.logit_scale.item()
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={lr_current:.2e} | logit_scale={logit_scale:.3f}"
        )

        # Retrieval evaluation: R@1, R@5, median rank for I→T and T→I directions.
        retrieval = evaluate_retrieval(
            model, val_ds.samples, preprocess_val, tokenizer, device,
            args.use_mask, batch_size=args.batch_size,
        ) or {}
        if retrieval:
            # mean_r1 averages both retrieval directions as a single summary metric.
            retrieval["retrieval/mean_r1"] = (
                retrieval["retrieval/i2t_r1"] + retrieval["retrieval/t2i_r1"]
            ) / 2
            print(
                f"           | I2T R@1={retrieval['retrieval/i2t_r1']:.1%}"
                f"  R@5={retrieval['retrieval/i2t_r5']:.1%}"
                f"  med={retrieval['retrieval/i2t_median_rank']:.0f}"
                f" | T2I R@1={retrieval['retrieval/t2i_r1']:.1%}"
                f"  R@5={retrieval['retrieval/t2i_r5']:.1%}"
                f"  med={retrieval['retrieval/t2i_median_rank']:.0f}"
                f" | avg/bs={int(retrieval['retrieval/batch_size'])} n={int(retrieval['retrieval/n_pairs'])}"
            )
        if use_wandb:
            log_dict = {
                "train/loss": train_loss, "val/loss": val_loss,
                "train/lr": lr_current,   "train/logit_scale": logit_scale,
            }
            log_dict.update(retrieval)
            wandb.log(log_dict, step=epoch)

        # Save a checkpoint whenever validation loss reaches a new minimum.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            run_dir / "best_val_checkpoint.pt", lora_config=lora_config)

        # Save the best-R@1 checkpoint and track early stopping progress.
        mean_r1 = retrieval.get("retrieval/mean_r1", 0.0)
        if mean_r1 > best_mean_r1:
            best_mean_r1      = mean_r1
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_loss,
                            run_dir / "best_r1_checkpoint.pt", lora_config=lora_config)
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered (no R@1 improvement for {args.patience} epochs).")
                break

    # ── Final checkpoint and summary ──────────────────────────────────────────
    save_checkpoint(model, optimizer, args.epochs, val_loss,
                    run_dir / "final_checkpoint.pt", lora_config=lora_config)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} | Best mean R@1: {best_mean_r1:.1%}")
    print(f"Checkpoints saved to: {run_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
