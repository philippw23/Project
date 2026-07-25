"""BiomedCLIP contrastive pretraining with LoRA on the ViT image encoder.

Domain-adapts microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 on
(X-ray image, German radiology report) pairs from the bone-tumor dataset.
LoRA is injected into the last N transformer blocks of the ViT; the text
encoder (PubMedBERT) is kept fully frozen throughout.

Usage example:
    python src/biomedclip_pretrain.py \\
        --excel data/internal_dataset/metadata.xlsx \\
        --reports data/internal_dataset/text/sanitized_reports.json \\
        --use_mask --lora_layers 4 --epochs 50

Pass --cv_dir <folder> (e.g. data/internal_dataset/cv_binary) to run one full
pretraining pass per fold file matching --cv_pattern instead of a single run:
results land under run_<name>/fold0/, fold1/, ... alongside the usual
checkpoints/split.json. Mutually exclusive with --splits and --sweep.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import shutil
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
    DEFAULT_DATASET_JSON,
    DEFAULT_OUT_DIR,
    save_checkpoint,
    print_biomedclip_architecture,
)
from biomedclip.data.transforms import build_train_transform
from biomedclip.data.splits import build_stratified_splits
from biomedclip.data.datasets import BoneTumorPairDataset
from biomedclip.models.lora import inject_lora, count_trainable_params
from biomedclip.loss.contrastive import clip_loss
from biomedclip.eval.retrieval import evaluate, evaluate_retrieval

def _unfreeze_last_blocks(model: nn.Module, unfreeze_blocks: int) -> None:
    """Freeze the whole model then unfreeze the last N ViT blocks plus projection layers.

    Used for partial fine-tuning without LoRA.  The text encoder body stays frozen.
    Unfrozen also: model.logit_scale, model.visual.head, and model.text.proj.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    blocks   = model.visual.trunk.blocks
    n_blocks = len(blocks)
    effective = min(unfreeze_blocks, n_blocks)
    for block in blocks[n_blocks - effective:]:
        for p in block.parameters():
            p.requires_grad_(True)

    model.logit_scale.requires_grad_(True)
    if hasattr(model.visual, "head") and model.visual.head is not None:
        for p in model.visual.head.parameters():
            p.requires_grad_(True)
    if hasattr(model.text, "proj") and model.text.proj is not None:
        for p in model.text.proj.parameters():
            p.requires_grad_(True)

def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_frac: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up followed by cosine annealing to min_lr_frac * peak_lr.

    During the first warmup_epochs epochs the lr rises linearly from 0 to peak.
    Afterwards it follows a cosine curve that reaches min_lr_frac at epoch total_epochs.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))

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
    # ── Data / splits ────────────────────────────────────────────────────────
    parser.add_argument("--splits", default=None,
                        help="Path to a pre-existing split.json (from create_split.py). "
                             "If omitted, a new split is generated from --dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_JSON),
                        help="Path to dataset_full.json — used only when --splits is omitted "
                             "(default: %(default)s)")
    parser.add_argument("--cv_dir", default=None,
                        help="Directory of fold split files (e.g. data/internal_dataset/cv_binary). "
                             "If set, runs one pretraining pass per fold, writing each fold's "
                             "checkpoints to run_dir/foldN/. Mutually exclusive with --splits and --sweep.")
    parser.add_argument("--cv_pattern", default="split_binary_fold*.json",
                        help="Glob for fold files inside --cv_dir (sorted; fold index parsed "
                             "from each filename, e.g. '...fold3...' -> fold3).")
    parser.add_argument("--use_mask",  action="store_true",
                        help="Crop images around the lesion using segmentation masks")
    parser.add_argument("--use_phrases", action="store_true",
                        help="Replace full report text with concatenated LLM-extracted phrases "
                             "(befund_phrases + beurteilung_phrases, separated by '. '). "
                             "Samples missing either phrase list are dropped.")

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR),
                        help="Output directory for checkpoints (default: %(default)s)")

    # ── Debug / inspection ───────────────────────────────────────────────────
    parser.add_argument("--print_architecture", action="store_true",
                        help="Load BioMedCLIP, print image/text encoder architecture, then exit")
    parser.add_argument("--print_full_model", action="store_true",
                        help="With --print_architecture, also print the full model wrapper")

    # ── Encoder adaptation (LoRA vs. partial fine-tune) ─────────────────────
    parser.add_argument("--lora_layers", type=int, default=4,
                        help="Number of last ViT transformer blocks to apply LoRA to (default: %(default)s)")
    parser.add_argument("--lora_r",     type=int,   default=8,
                        help="LoRA rank r (default: %(default)s)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha scaling factor (default: %(default)s)")
    parser.add_argument("--no_lora", action="store_true",
                        help="Skip LoRA; instead fully unfreeze the last --unfreeze_blocks ViT blocks.")
    parser.add_argument("--unfreeze_blocks", type=int, default=4,
                        help="For --no_lora: number of last ViT blocks to unfreeze (default: %(default)s)")

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int,   default=32,
                        help="Training batch size (default: %(default)s)")
    parser.add_argument("--epochs",     type=int,   default=50,
                        help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--lr_blocks", type=float, default=1e-5,
                        help="LR for unfrozen ViT blocks in partial fine-tune mode (default: %(default)s)")
    parser.add_argument("--lr_proj", type=float, default=5e-4,
                        help="Peak LR for projection heads and logit_scale (default: %(default)s)")
    parser.add_argument("--lr_lora",     type=float, default=1e-4,
                        help="Peak LR for LoRA adapter weights (default: %(default)s)")
    parser.add_argument("--weight_decay", type=float, default=0.2,
                        help="AdamW weight decay for non-norm parameters (default: %(default)s)")
    parser.add_argument("--patience",    type=int,   default=20,
                        help="Early stopping patience in epochs based on mean R@1 (0 to disable, default: %(default)s)")

    # ── Downstream split fractions (used when --splits is omitted) ─────────
    parser.add_argument("--downstream_train_frac", type=float, default=0.8,
                        help="Fraction of downstream data for classifier training (default: %(default)s)")
    parser.add_argument("--downstream_val_frac",   type=float, default=0.1,
                        help="Fraction of downstream data for classifier validation (default: %(default)s)")
    parser.add_argument("--test_frac",             type=float, default=0.1,
                        help="Fraction of downstream data held out for final evaluation (default: %(default)s)")

    # ── Reproducibility ──────────────────────────────────────────────────────
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for reproducibility (default: %(default)s)")

    # ── Experiment tracking ──────────────────────────────────────────────────
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
    for key in ("lora_layers", "lora_r", "lr_lora", "lr_proj", "weight_decay", "batch_size", "unfreeze_blocks", "lr_blocks"):
        if key in cfg:
            setattr(args, key, cfg[key])
    if not args.no_lora:
        # Keep alpha proportional to rank (a common LoRA convention).
        args.lora_alpha = 2.0 * args.lora_r


def main(args: argparse.Namespace) -> None:
    if args.cv_dir and args.splits is not None:
        raise SystemExit("--cv_dir and --splits are mutually exclusive — --splits is managed per fold.")
    if args.cv_dir and args.sweep:
        raise SystemExit("--cv_dir and --sweep are mutually exclusive.")

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

    # ── Output directory ──────────────────────────────────────────────────────
    if args.no_lora:
        tune_tag = f"unfreeze{args.unfreeze_blocks}"
    else:
        tune_tag = f"lora{args.lora_layers}"
    text_tag = "_phrases" if args.use_phrases else ""
    run_name = f"run_bs{args.batch_size}_{tune_tag}{text_tag}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / "biomedclip_pretrain" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ── Fold targets: 1 entry for a normal run, N for --cv_dir ────────────────
    if args.cv_dir:
        fold_files = sorted(Path(args.cv_dir).glob(args.cv_pattern))
        if not fold_files:
            raise SystemExit(f"No fold files matching {args.cv_pattern!r} in {args.cv_dir}")
        fold_targets = []
        for fp in fold_files:
            m = re.search(r"fold(\d+)", fp.stem)
            if not m:
                raise SystemExit(f"Could not parse a fold index out of {fp.name!r} (expected '...foldN...')")
            fold_targets.append((fp, run_dir / f"fold{int(m.group(1))}", int(m.group(1))))
        print(f"CV mode: {len(fold_targets)} folds from {args.cv_dir}")
    else:
        split_path = Path(args.splits) if args.splits else None
        fold_targets = [(split_path, run_dir, None)]

    for split_path, this_run_dir, fold_idx in fold_targets:
        if args.cv_dir:
            this_run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'#' * 70}\n# FOLD {fold_idx} — {split_path.name}\n{'#' * 70}")

        # ── Model loading ───────────────────────────────────────────────────
        print(f"Loading model: {MODEL_TAG}")
        # open_clip returns (model, train_preprocess, val_preprocess).
        # We override train_preprocess with a custom augmentation pipeline below.
        model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)
        tokenizer = open_clip.get_tokenizer(MODEL_TAG)
        model = model.to(device)
        # Adds random horizontal flip, colour jitter, and random resized crop on top of val_preprocess.
        augment_transform = build_train_transform(preprocess_val)

        if args.no_lora:
            _unfreeze_last_blocks(model, args.unfreeze_blocks)
            print(f"Partial fine-tune: last {args.unfreeze_blocks} ViT blocks unfrozen (no LoRA)")
        else:
            args.lora_alpha = 2.0 * args.lora_r  # enforce alpha = 2r regardless of CLI default
            inject_lora(model, args.lora_layers, args.lora_r, args.lora_alpha)
            print(
                f"LoRA injected into last {args.lora_layers} ViT blocks "
                f"(r={args.lora_r}, alpha={args.lora_alpha})"
            )
        n_trainable = count_trainable_params(model)
        n_total     = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f} %)")

        # ── Data splits and datasets ─────────────────────────────────────────
        if split_path is not None:
            with open(split_path, encoding="utf-8") as fh:
                split_data = json.load(fh)
            shutil.copy(split_path, this_run_dir / "split.json")
            train_samples = split_data["train"]
            val_samples   = split_data["val"]
            print(f"Loaded split from {split_path} ({len(train_samples)} train, {len(val_samples)} val samples)")
        else:
            train_samples, val_samples, _ = build_stratified_splits(args, run_dir=this_run_dir)

        def _to_tuples(samples: list[dict]) -> list[tuple]:
            if args.use_phrases:
                result = []
                dropped = 0
                for s in samples:
                    bp  = s.get("befund_phrases") or []
                    bep = s.get("beurteilung_phrases") or []
                    if not bp or not bep:
                        dropped += 1
                        continue
                    result.append((Path(s["image"]), Path(s["mask"]), ". ".join(bp + bep)))
                if dropped:
                    print(f"  use_phrases: dropped {dropped} samples missing phrase lists.")
                return result
            return [(Path(s["image"]), Path(s["mask"]), s["report"]) for s in samples]

        train_ds = BoneTumorPairDataset(_to_tuples(train_samples), augment_transform, tokenizer, args.use_mask)
        val_ds   = BoneTumorPairDataset(_to_tuples(val_samples),   preprocess_val,   tokenizer, args.use_mask)
        print(f"Pretrain datasets: {len(train_ds)} train / {len(val_ds)} monitor-val")

        use_pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_pin_memory,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_pin_memory,
            drop_last=False,
        )

        # ── Optimizer ─────────────────────────────────────────────────────────
        # Norm and bias parameters are excluded from weight decay because they
        # act as scale/shift terms and should not be penalised toward zero.
        no_decay_suffixes = ("bias", "norm.weight", "norm.bias", "ln_1.weight", "ln_1.bias",
                             "ln_2.weight", "ln_2.bias", "ln_pre.weight", "ln_pre.bias",
                             "ln_post.weight", "ln_post.bias")

        if args.no_lora:
            # Differential LR: unfrozen ViT block params use a low LR to avoid
            # destabilising pretrained representations; projection and logit_scale
            # use the full (higher) LR to adapt the contrastive head quickly.
            proj_param_names = {"visual.head", "text.proj", "logit_scale"}
            block_decay, block_no_decay, proj_params = [], [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(name == pn or name.startswith(pn + ".") for pn in proj_param_names):
                    proj_params.append(param)
                elif name.endswith(no_decay_suffixes):
                    block_no_decay.append(param)
                else:
                    block_decay.append(param)
            trainable_params = block_decay + block_no_decay + proj_params
            optimizer = torch.optim.AdamW(
                [
                    {"params": block_decay,    "lr": args.lr_blocks, "weight_decay": args.weight_decay},
                    {"params": block_no_decay, "lr": args.lr_blocks, "weight_decay": 0.0},
                    {"params": proj_params,    "lr": args.lr_proj,   "weight_decay": 0.0},
                ],
                betas=(0.9, 0.98),
                eps=1e-6,
            )
        else:
            lr_lora = args.lr_lora
            lr_proj = args.lr_proj
            proj_param_names = {"visual.head", "text.proj", "logit_scale"}
            lora_decay, lora_no_decay, proj_params = [], [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(name == pn or name.startswith(pn + ".") for pn in proj_param_names):
                    proj_params.append(param)
                elif name.endswith(no_decay_suffixes):
                    lora_no_decay.append(param)
                else:
                    lora_decay.append(param)
            trainable_params = lora_decay + lora_no_decay + proj_params
            optimizer = torch.optim.AdamW(
                [
                    {"params": lora_decay,    "lr": lr_lora, "weight_decay": args.weight_decay},
                    {"params": lora_no_decay, "lr": lr_lora, "weight_decay": 0.0},
                    {"params": proj_params,   "lr": lr_proj, "weight_decay": 0.0},
                ],
                betas=(0.9, 0.98),
                eps=1e-6,
            )

        # Warm-up for the first 20 % of epochs, then cosine decay to 0.
        warmup_epochs = max(1, args.epochs // 5)
        scheduler     = make_scheduler(optimizer, warmup_epochs, args.epochs)
        # GradScaler prevents float16 gradient underflow during mixed-precision training.
        scaler        = torch.amp.GradScaler("cuda") if device.type == "cuda" else torch.amp.GradScaler("cpu")

        # ── W&B initialisation ────────────────────────────────────────────────
        use_wandb = (args.wandb or args.sweep) and WANDB_AVAILABLE
        if (args.wandb or args.sweep) and not WANDB_AVAILABLE:
            warnings.warn("--wandb/--sweep set but wandb is not installed. Skipping.")
        if use_wandb:
            wandb_config = {
                "lora_layers": args.lora_layers,
                "lora_r":      args.lora_r,
                "lora_alpha":  args.lora_alpha,
                "epochs":      args.epochs,
                "batch_size":  args.batch_size,
                "lr_lora":     args.lr_lora,
                "lr_proj":     args.lr_proj,
                "weight_decay": args.weight_decay,
                "seed":        args.seed,
                "use_mask":    args.use_mask,
                "text_mode":   "phrases" if args.use_phrases else "full",
            }
            if args.cv_dir:
                wb_base = args.wandb_run or run_name
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=f"{wb_base}_fold{fold_idx}",
                    group=wb_base,
                    config=wandb_config,
                )
            else:
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run,
                    config=wandb_config,
                )
            if args.sweep:
                # Overwrite CLI args with sweep-agent-supplied hyperparameters.
                _apply_sweep_config(args)

        # ── Training loop ────────────────────────────────────────────────────
        best_val_loss     = float("inf")
        best_mean_r1      = 0.0          # average of I2T R@1 and T2I R@1
        epochs_no_improve = 0            # consecutive epochs without R@1 improvement
        lora_config = (
            {"no_lora": True, "unfreeze_blocks": args.unfreeze_blocks}
            if args.no_lora
            else {"lora_layers": args.lora_layers, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha}
        )

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
                    f" | n={int(retrieval['retrieval/n_pairs'])}"
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
                                this_run_dir / "best_val_checkpoint.pt", lora_config=lora_config)

            # Save the best-R@1 checkpoint and track early stopping progress.
            mean_r1 = retrieval.get("retrieval/mean_r1", 0.0)
            if mean_r1 > best_mean_r1:
                best_mean_r1      = mean_r1
                epochs_no_improve = 0
                save_checkpoint(model, optimizer, epoch, val_loss,
                                this_run_dir / "best_r1_checkpoint.pt", lora_config=lora_config)
            else:
                epochs_no_improve += 1
                if args.patience > 0 and epochs_no_improve >= args.patience:
                    print(f"\nEarly stopping triggered (no R@1 improvement for {args.patience} epochs).")
                    break

        # ── Final checkpoint and summary ─────────────────────────────────────
        save_checkpoint(model, optimizer, args.epochs, val_loss,
                        this_run_dir / "final_checkpoint.pt", lora_config=lora_config)
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} | Best mean R@1: {best_mean_r1:.1%}")
        print(f"Checkpoints saved to: {this_run_dir}")
        if use_wandb:
            wandb.finish()

        # Fold's model/optimizer must not leak into the next fold's GPU memory.
        del model, optimizer, scheduler, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main(parse_args())
