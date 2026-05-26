"""LACE v1 pretraining: L_ITA + L_sim + L_ortho active from epoch 1."""
from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import json
import shutil

from biomedclip.data.splits import build_stratified_splits
from biomedclip.utils.misc import (
    DEFAULT_DATASET_JSON,
    DEFAULT_OUT_DIR,
    DEFAULT_SPLITS,
    ROOT_DIR,
)
from LACE.data.datasets import BTXRDOrthoDataset
from LACE.data.splits import build_pretrain_datasets_lace
from LACE.data.transforms import build_train_transform_lace
from LACE.eval.retrieval import evaluate_retrieval_lace
from LACE.loss.objectives import multi_positive_soft_semantic_loss, ortho_loss
from LACE.models.encoders import BiomedCLIPTextEncoder, SharedViT

DEFAULT_BTXRD_IMAGES = ROOT_DIR / "data" / "BTXRD" / "images"
DEFAULT_BTXRD_ANNOTS = ROOT_DIR / "data" / "BTXRD" / "Annotations"


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    schedule: str = "constant",
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        if schedule == "cosine":
            progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _phrase_to_image(phrase_mask: torch.Tensor) -> torch.Tensor:
    """Map each valid phrase to its source image index.

    Args:
        phrase_mask: [B, J] bool — True = valid phrase
        B: batch size
        J: max phrases per sample
    Returns:
        [N_total] int64 — image index for each valid phrase
    """
    B, J = phrase_mask.shape
    idx = torch.arange(B, device=phrase_mask.device).unsqueeze(1).expand(-1, J)
    return idx[phrase_mask]


def _encode_text(batch, text_enc, device, text_mode):
    """Encode beurteilung and befund according to text_mode.

    Returns (z_beur_phrases, beur_pmask, z_bef_phrases, bef_pmask) where:
        z_beur_phrases [B, J, D]   — per-phrase beurteilung embeddings for L_ITA
        beur_pmask     [B, J] bool — True = real phrase
        z_bef_phrases  [B, J, D]   — per-phrase befund embeddings for L_sim
        bef_pmask      [B, J] bool — True = real phrase
    """
    if text_mode in ("phrase_mean", "phrase_attn"):
        beur_pids  = batch["beur_phrase_ids"].to(device)
        beur_pattn = batch["beur_phrase_attn"].to(device)
        beur_pmask = batch["beur_phrase_mask"].to(device)
        bef_pids   = batch["bef_phrase_ids"].to(device)
        bef_pattn  = batch["bef_phrase_attn"].to(device)
        bef_pmask  = batch["bef_phrase_mask"].to(device)
        z_beur_phrases = text_enc._encode_phrase_batch(beur_pids, beur_pattn, beur_pmask)
        z_bef_phrases  = text_enc._encode_phrase_batch(bef_pids,  bef_pattn,  bef_pmask)
    else:
        beur_ids  = batch["beurteilung_ids"].to(device)
        beur_mask = batch["beurteilung_mask"].to(device)
        z_text = text_enc.encode_beurteilung(beur_ids, beur_mask)
        z_beur_phrases = z_text.unsqueeze(1)                                      # [B, 1, D]
        beur_pmask = torch.ones(z_beur_phrases.shape[:2], dtype=torch.bool, device=device)
        bef_ids   = batch["befund_ids"].to(device)
        bef_mask  = batch["befund_mask"].to(device)
        _, z_bef_phrases = text_enc.encode_befund(bef_ids, bef_mask)
        bef_pmask = bef_mask.bool()

    return z_beur_phrases, beur_pmask, z_bef_phrases, bef_pmask


def train_one_epoch(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    internal_loader: DataLoader,
    btxrd_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    τ: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    trainable_params: list,
    device: torch.device,
    lambda_sim: float,
    lambda_reg: float,
    text_mode: str = "phrase_attn",
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    vit.train()
    text_enc.train()

    total_ita = total_sim = total_ortho = total_total = 0.0
    n_batches = 0

    btxrd_cycle = itertools.cycle(btxrd_loader) if btxrd_loader is not None else None

    pbar = tqdm(internal_loader, desc="  train", leave=False,
                disable=not sys.stdout.isatty())

    for batch in pbar:
        full_img   = batch["full_image"].to(device)
        crop_img   = batch["crop_image"].to(device)
        plabels    = batch["patch_labels"].to(device)
        has_mask   = batch["has_mask"].to(device)
        has_befund = batch["has_befund"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):

            # Single ViT pass on full image: CLS for L_ITA, patches for L_ortho
            cls_feat, patch_feat = vit.forward_all(full_img)
            z_img = vit.img_proj(cls_feat)                                # [B, D]

            z_beur_phrases, beur_pmask, z_bef_phrases, bef_pmask = _encode_text(
                batch, text_enc, device, text_mode
            )

            # L_ITA: full-image CLS vs beurteilung phrases
            p2i_ita = _phrase_to_image(beur_pmask)                        # [N_total_ita]
            l_ita = multi_positive_soft_semantic_loss(
                z_img[p2i_ita], z_beur_phrases, beur_pmask, τ
            )

            # L_sim: crop CLS vs befund phrases
            l_sim = torch.zeros(1, device=device)[0]
            sim_valid = has_mask & has_befund
            if sim_valid.sum() >= 2:
                crop_cls      = vit.forward_cls(crop_img[sim_valid])  # [B_sim, D]
                bef_phr_sub   = z_bef_phrases[sim_valid]              # [B_sim, J, D]
                bef_pmask_sub = bef_pmask[sim_valid]                  # [B_sim, J]
                p2i_sim       = _phrase_to_image(bef_pmask_sub)       # [N_total_sim]
                l_sim = multi_positive_soft_semantic_loss(
                    crop_cls[p2i_sim], bef_phr_sub, bef_pmask_sub, τ
                )

            # L_ortho: lesion-background patch orthogonality
            l_ortho = torch.zeros(1, device=device)[0]
            ortho_patches = patch_feat
            ortho_labels  = plabels
            if btxrd_cycle is not None:
                btxrd_batch   = next(btxrd_cycle)
                btxrd_img     = btxrd_batch["image"].to(device)
                btxrd_labels  = btxrd_batch["patch_labels"].to(device)
                btxrd_patches = vit.forward_patches(btxrd_img)
                ortho_patches = torch.cat([ortho_patches, btxrd_patches], dim=0)
                ortho_labels  = torch.cat([ortho_labels,  btxrd_labels],  dim=0)

                l_ortho = ortho_loss(ortho_patches, ortho_labels)

            loss = log_lambda_ita.exp() * l_ita + lambda_sim * l_sim + lambda_reg * l_ortho

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_ita   += l_ita.item()
        total_sim   += l_sim.item()
        total_ortho += l_ortho.item()
        total_total += loss.item()
        n_batches   += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", ita=f"{l_ita.item():.4f}")

    d = max(1, n_batches)
    return {
        "train/loss":       total_total / d,
        "train/l_ita":      total_ita   / d,
        "train/l_sim":      total_sim   / d,
        "train/l_ortho":    total_ortho / d,
        "train/lambda_ita": log_lambda_ita.exp().item(),
        "train/tau":        τ.item(),
    }


@torch.no_grad()
def evaluate_lace(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    val_loader: DataLoader,
    τ: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    device: torch.device,
    lambda_sim: float,
    lambda_reg: float,
    text_mode: str = "phrase_attn",
) -> dict[str, float]:
    vit.eval()
    text_enc.eval()

    total_ita = total_sim = total_ortho = total_total = 0.0
    n_batches = 0

    for batch in val_loader:
        full_img   = batch["full_image"].to(device)
        crop_img   = batch["crop_image"].to(device)
        plabels    = batch["patch_labels"].to(device)
        has_mask   = batch["has_mask"].to(device)
        has_befund = batch["has_befund"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            cls_feat, patch_feat = vit.forward_all(full_img)
            z_img  = vit.img_proj(cls_feat)

            z_beur_phrases, beur_pmask, z_bef_phrases, bef_pmask = _encode_text(
                batch, text_enc, device, text_mode
            )

            p2i_ita = _phrase_to_image(beur_pmask)
            l_ita = multi_positive_soft_semantic_loss(
                z_img[p2i_ita], z_beur_phrases, beur_pmask, τ
            )

            l_sim = torch.zeros(1, device=device)[0]
            sim_valid = has_mask & has_befund
            if sim_valid.sum() >= 2:
                crop_cls      = vit.forward_cls(crop_img[sim_valid])
                bef_phr_sub   = z_bef_phrases[sim_valid]
                bef_pmask_sub = bef_pmask[sim_valid]
                p2i_sim       = _phrase_to_image(bef_pmask_sub)
                l_sim = multi_positive_soft_semantic_loss(
                    crop_cls[p2i_sim], bef_phr_sub, bef_pmask_sub, τ
                )

            l_ortho = ortho_loss(patch_feat, plabels)

            loss = log_lambda_ita.exp() * l_ita + lambda_sim * l_sim + lambda_reg * l_ortho

        total_ita   += l_ita.item()
        total_sim   += l_sim.item()
        total_ortho += l_ortho.item()
        total_total += loss.item()
        n_batches   += 1

    d = max(1, n_batches)
    return {
        "val/loss":    total_total / d,
        "val/l_ita":   total_ita   / d,
        "val/l_sim":   total_sim   / d,
        "val/l_ortho": total_ortho / d,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LACE v1 pretraining")

    parser.add_argument("--splits", default=str(DEFAULT_SPLITS),
                        help="Path to split.json (default: %(default)s). "
                             "Pass --splits '' to generate a fresh split from --dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_JSON),
                        help="Path to dataset_full.json — used only when --splits is omitted "
                             "(default: %(default)s)")
    parser.add_argument("--btxrd_images", default=str(DEFAULT_BTXRD_IMAGES))
    parser.add_argument("--btxrd_annots", default=str(DEFAULT_BTXRD_ANNOTS))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))

    parser.add_argument("--lora_layers", type=int,   default=4)
    parser.add_argument("--lora_r",      type=int,   default=8)
    parser.add_argument("--lora_alpha",  type=float, default=16.0)
    parser.add_argument("--embed_dim",   type=int,   default=256,
                        help="Projection head output dimension (default: 256)")

    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--btxrd_batch_size", type=int,   default=16)
    parser.add_argument("--max_text_len",     type=int,   default=128)

    parser.add_argument(
        "--text_mode", default="phrase_attn",
        choices=["full", "concat", "phrase_mean", "phrase_attn"],
        help="Text encoding strategy for pretraining (default: phrase_attn).",
    )
    parser.add_argument("--max_bef_phrases",  type=int, default=16,
                        help="Max befund phrases per sample (phrase modes only).")
    parser.add_argument("--max_beur_phrases", type=int, default=16,
                        help="Max beurteilung phrases per sample (phrase modes only).")

    parser.add_argument("--epochs",        type=int, default=100)
    parser.add_argument("--patience",      type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--scheduler",    default="constant", choices=["constant", "cosine"])
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--lambda_sim",   type=float, default=1.0)
    parser.add_argument("--lambda_reg",   type=float, default=0.1)

    parser.add_argument("--downstream_train_frac", type=float, default=0.8)
    parser.add_argument("--downstream_val_frac",   type=float, default=0.1)
    parser.add_argument("--test_frac",             type=float, default=0.1)
    parser.add_argument("--seed",                  type=int,   default=42)

    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="lace-pretrain")
    parser.add_argument("--wandb_entity",  default=None)
    parser.add_argument("--wandb_run",     default=None)

    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Models ────────────────────────────────────────────────────────────────
    vit      = SharedViT(args.lora_layers, args.lora_r, args.lora_alpha, args.embed_dim).to(device)
    text_enc = BiomedCLIPTextEncoder(embed_dim=args.embed_dim).to(device)
    τ              = nn.Parameter(torch.tensor(0.07, device=device))
    log_lambda_ita = nn.Parameter(torch.zeros([], device=device))

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = (
        Path(args.out_dir) / "lace_pretrain" /
        datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.splits is not None:
        with open(args.splits, encoding="utf-8") as fh:
            split_data = json.load(fh)
        shutil.copy(args.splits, run_dir / "split.json")
        pretrain_samples = split_data["train"]
        print(f"Loaded split from {args.splits} ({len(pretrain_samples)} train samples)")
    else:
        train, _val, _test = build_stratified_splits(args, run_dir=run_dir)
        pretrain_samples = train

    preprocess_val   = vit.preprocess_val
    preprocess_train = build_train_transform_lace(preprocess_val)
    tokenizer        = text_enc.tokenizer

    train_ds, val_ds = build_pretrain_datasets_lace(
        pretrain_samples, preprocess_train, preprocess_val,
        tokenizer, args.seed, max_text_len=args.max_text_len,
        text_mode=args.text_mode,
        max_bef_phrases=args.max_bef_phrases,
        max_beur_phrases=args.max_beur_phrases,
    )

    use_pin = device.type == "cuda"
    internal_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=use_pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=use_pin,
    )
    btxrd_ds = BTXRDOrthoDataset(
        Path(args.btxrd_images), Path(args.btxrd_annots), preprocess_val,
    )
    btxrd_loader = DataLoader(
        btxrd_ds, batch_size=args.btxrd_batch_size, shuffle=True,
        num_workers=4, pin_memory=use_pin, drop_last=True,
    ) if len(btxrd_ds) > 0 else None

    # ── Optimizer ─────────────────────────────────────────────────────────────
    no_decay_keys = ("bias", "norm.weight", "norm.bias", "ln_1.weight", "ln_1.bias",
                     "ln_2.weight", "ln_2.bias")

    def _split_params(module: nn.Module):
        decay, no_decay = [], []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if any(name.endswith(k) for k in no_decay_keys):
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay

    vit_decay, vit_no_decay       = _split_params(vit)
    txt_decay, txt_no_decay       = _split_params(text_enc)
    decay_params   = vit_decay   + txt_decay   + [τ, log_lambda_ita]
    no_decay_params= vit_no_decay+ txt_no_decay

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.98), eps=1e-6,
    )
    trainable_params = decay_params + no_decay_params

    scheduler = make_scheduler(optimizer, args.warmup_epochs, args.epochs, args.scheduler)
    scaler    = (
        torch.amp.GradScaler("cuda") if device.type == "cuda"
        else torch.amp.GradScaler("cpu")
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
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

    # ── Training loop ─────────────────────────────────────────────────────────
    lora_cfg = {
        "lora_layers": args.lora_layers,
        "lora_r":      args.lora_r,
        "lora_alpha":  args.lora_alpha,
        "embed_dim":   args.embed_dim,
    }
    best_val_loss    = float("inf")
    patience_counter = 0

    print(f"\nStarting LACE v1 training: {args.epochs} epochs, patience={args.patience}\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            vit, text_enc, internal_loader, btxrd_loader,
            optimizer, scaler, τ, log_lambda_ita, trainable_params,
            device=device,
            lambda_sim=args.lambda_sim, lambda_reg=args.lambda_reg,
            text_mode=args.text_mode,
        )
        val_metrics = evaluate_lace(
            vit, text_enc, val_loader, τ, log_lambda_ita,
            device=device,
            lambda_sim=args.lambda_sim, lambda_reg=args.lambda_reg,
            text_mode=args.text_mode,
        )
        scheduler.step()

        val_loss   = val_metrics["val/loss"]
        lr_current = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_metrics['train/loss']:.4f} | val={val_loss:.4f} | "
            f"ita={val_metrics['val/l_ita']:.4f} | "
            f"sim={val_metrics['val/l_sim']:.4f} | "
            f"ortho={val_metrics['val/l_ortho']:.4f} | "
            f"lr={lr_current:.2e}"
        )

        retrieval = evaluate_retrieval_lace(
            vit, text_enc, val_loader, device, text_mode=args.text_mode,
        )
        if retrieval:
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
            wandb.log(
                {**train_metrics, **val_metrics, **retrieval,
                 "epoch": epoch, "train/lr": lr_current},
                step=epoch,
            )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch":           epoch,
                    "val_loss":        val_loss,
                    "lora_config":     lora_cfg,
                    "vit_state":       vit.state_dict(),
                    "text_enc_state":  text_enc.state_dict(),
                    "tau":             τ.data,
                    "log_lambda_ita":  log_lambda_ita.data,
                    "optimizer_state": optimizer.state_dict(),
                },
                run_dir / "best_checkpoint.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs).\n")
                break

    torch.save(
        {
            "epoch":          epoch,
            "val_loss":       val_loss,
            "lora_config":    lora_cfg,
            "vit_state":      vit.state_dict(),
            "text_enc_state": text_enc.state_dict(),
            "tau":            τ.data,
            "log_lambda_ita": log_lambda_ita.data,
        },
        run_dir / "final_checkpoint.pt",
    )
    print(f"\nDone. Best val loss: {best_val_loss:.4f}. Checkpoints: {run_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
