"""LACE pretraining: three-stage curriculum with L_ITA, L_sim, and L_ortho."""
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
from LACE.loss.objectives import ita_loss, ortho_loss, sim_loss
from LACE.models.encoders import BiomedCLIPTextEncoder, SharedViT

DEFAULT_BTXRD_IMAGES = ROOT_DIR / "data" / "BTXRD" / "images"
DEFAULT_BTXRD_ANNOTS = ROOT_DIR / "data" / "BTXRD" / "Annotations"


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _encode_text(batch, text_enc, device, text_mode, stage):
    """Encode beurteilung and befund according to text_mode.

    Returns (z_text, z_bef_cls, proj_words, bef_pmask) where:
        z_text      [B, D]    — beurteilung embedding for L_ITA
        z_bef_cls   [B, D]    — befund CLS for InfoNCE anchor in L_sim
        proj_words  [B, L, D] — word/phrase embeddings for L_sim attention
        bef_pmask   [B, L] bool | None — phrase padding mask (phrase modes only)
    z_bef_cls and proj_words are None when stage < 2.
    """
    if text_mode in ("phrase_mean", "phrase_attn"):
        beur_pids  = batch["beur_phrase_ids"].to(device)
        beur_pattn = batch["beur_phrase_attn"].to(device)
        beur_pmask = batch["beur_phrase_mask"].to(device)
        z_text = text_enc.encode_beurteilung_phrases(
            beur_pids, beur_pattn, beur_pmask, text_mode
        )
        if stage >= 2:
            bef_pids  = batch["bef_phrase_ids"].to(device)
            bef_pattn = batch["bef_phrase_attn"].to(device)
            bef_pmask = batch["bef_phrase_mask"].to(device)
            z_bef_cls, proj_words = text_enc.encode_befund_phrases(
                bef_pids, bef_pattn, bef_pmask
            )
        else:
            z_bef_cls, proj_words, bef_pmask = None, None, None
    else:
        beur_ids  = batch["beurteilung_ids"].to(device)
        beur_mask = batch["beurteilung_mask"].to(device)
        z_text = text_enc.encode_beurteilung(beur_ids, beur_mask)
        if stage >= 2:
            bef_ids   = batch["befund_ids"].to(device)
            bef_mask  = batch["befund_mask"].to(device)
            z_bef_cls, proj_words = text_enc.encode_befund(bef_ids, bef_mask)
            bef_pmask = None
        else:
            z_bef_cls, proj_words, bef_pmask = None, None, None

    return z_text, z_bef_cls, proj_words, bef_pmask


def train_one_epoch(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    internal_loader: DataLoader,
    btxrd_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    logit_scale: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    trainable_params: list,
    device: torch.device,
    stage: int,
    lambda_sim: float,
    lambda_reg: float,
    text_mode: str = "full",
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    vit.train()
    text_enc.train()

    total_ita = total_sim = total_ortho = total_total = 0.0
    n_batches = 0

    btxrd_cycle = itertools.cycle(btxrd_loader) if (stage >= 3 and btxrd_loader is not None) else None

    pbar = tqdm(internal_loader, desc=f"  stage{stage}", leave=False,
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
            z_img = vit.img_proj(cls_feat)

            z_text, z_bef_cls, proj_words, bef_pmask = _encode_text(
                batch, text_enc, device, text_mode, stage
            )

            # L_ITA (all stages)
            l_ita = ita_loss(z_img, z_text, logit_scale)

            # L_sim (stages 2+)
            l_sim = torch.zeros(1, device=device)[0]
            if stage >= 2:
                crop_patches = vit.forward_patches(crop_img)
                B, N, _ = crop_patches.shape
                proj_patches = vit.patch_proj(
                    crop_patches.reshape(-1, 768)
                ).reshape(B, N, vit.proj_dim)

                sim_valid = has_mask & has_befund
                if sim_valid.sum() >= 2:
                    pmask_valid = bef_pmask[sim_valid] if bef_pmask is not None else None
                    l_sim = sim_loss(
                        proj_patches[sim_valid],
                        proj_words[sim_valid],
                        z_bef_cls[sim_valid],
                        logit_scale,
                        phrase_mask=pmask_valid,
                    )

            # L_ortho (stage 3+)
            l_ortho = torch.zeros(1, device=device)[0]
            if stage >= 3:
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
    }


@torch.no_grad()
def evaluate_lace(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    val_loader: DataLoader,
    logit_scale: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    device: torch.device,
    stage: int,
    lambda_sim: float,
    lambda_reg: float,
    text_mode: str = "full",
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

            z_text, z_bef_cls, proj_words, bef_pmask = _encode_text(
                batch, text_enc, device, text_mode, stage
            )
            l_ita = ita_loss(z_img, z_text, logit_scale)

            l_sim = torch.zeros(1, device=device)[0]
            if stage >= 2:
                crop_patches = vit.forward_patches(crop_img)
                B, N, _ = crop_patches.shape
                proj_patches = vit.patch_proj(
                    crop_patches.reshape(-1, 768)
                ).reshape(B, N, vit.proj_dim)
                sim_valid = has_mask & has_befund
                if sim_valid.sum() >= 2:
                    pmask_valid = bef_pmask[sim_valid] if bef_pmask is not None else None
                    l_sim = sim_loss(
                        proj_patches[sim_valid], proj_words[sim_valid],
                        z_bef_cls[sim_valid], logit_scale,
                        phrase_mask=pmask_valid,
                    )

            l_ortho = torch.zeros(1, device=device)[0]
            if stage >= 3:
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
    parser = argparse.ArgumentParser(description="LACE pretraining (three-stage curriculum)")

    parser.add_argument("--splits", default=None,
                        help="Path to a pre-existing split.json (from create_split.py). "
                             "If omitted, a new split is generated from --dataset.")
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
        "--text_mode", default="full",
        choices=["full", "concat", "phrase_mean", "phrase_attn"],
        help="Text encoding strategy for pretraining (default: full).",
    )
    parser.add_argument("--max_bef_phrases",  type=int, default=16,
                        help="Max befund phrases per sample (phrase modes only).")
    parser.add_argument("--max_beur_phrases", type=int, default=16,
                        help="Max beurteilung phrases per sample (phrase modes only).")

    parser.add_argument("--stage1_epochs", type=int,   default=10)
    parser.add_argument("--stage2_epochs", type=int,   default=15)
    parser.add_argument("--stage3_epochs", type=int,   default=15)

    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--lambda_sim",   type=float, default=1.0)
    parser.add_argument("--lambda_reg",   type=float, default=0.1)
    parser.add_argument("--patience",     type=int,   default=20,
                        help="Early stopping patience (0 to disable)")

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
    logit_scale = nn.Parameter(
        torch.ones([], device=device) * math.log(1.0 / 0.07)
    )
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
    decay_params   = vit_decay   + txt_decay   + [logit_scale, log_lambda_ita]
    no_decay_params= vit_no_decay+ txt_no_decay

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.98), eps=1e-6,
    )
    trainable_params = decay_params + no_decay_params

    total_epochs  = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs
    warmup_epochs = max(1, total_epochs // 5)
    scheduler     = make_scheduler(optimizer, warmup_epochs, total_epochs)
    scaler        = (
        torch.amp.GradScaler("cuda") if device.type == "cuda"
        else torch.amp.GradScaler("cpu")
    )

    s1_end = args.stage1_epochs
    s2_end = s1_end + args.stage2_epochs

    def _get_stage(epoch: int) -> int:
        if epoch <= s1_end: return 1
        if epoch <= s2_end: return 2
        return 3

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
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    val_loss          = float("inf")

    print(f"\nStarting LACE training: {total_epochs} epochs "
          f"(stage1={args.stage1_epochs}, stage2={args.stage2_epochs}, "
          f"stage3={args.stage3_epochs})\n")

    for epoch in range(1, total_epochs + 1):
        stage        = _get_stage(epoch)
        active_btxrd = btxrd_loader if stage >= 3 else None

        train_metrics = train_one_epoch(
            vit, text_enc, internal_loader, active_btxrd,
            optimizer, scaler, logit_scale, log_lambda_ita, trainable_params,
            device=device, stage=stage,
            lambda_sim=args.lambda_sim, lambda_reg=args.lambda_reg,
            text_mode=args.text_mode,
        )
        val_metrics = evaluate_lace(
            vit, text_enc, val_loader, logit_scale, log_lambda_ita,
            device=device, stage=stage,
            lambda_sim=args.lambda_sim, lambda_reg=args.lambda_reg,
            text_mode=args.text_mode,
        )
        scheduler.step()

        val_loss   = val_metrics["val/loss"]
        lr_current = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d}/{total_epochs} [stage {stage}] | "
            f"train={train_metrics['train/loss']:.4f} | val={val_loss:.4f} | "
            f"ita={val_metrics['val/l_ita']:.4f} | "
            f"sim={val_metrics['val/l_sim']:.4f} | "
            f"ortho={val_metrics['val/l_ortho']:.4f} | "
            f"lr={lr_current:.2e}"
        )

        if use_wandb:
            wandb.log(
                {**train_metrics, **val_metrics, "epoch": epoch, "stage": stage,
                 "train/lr": lr_current},
                step=epoch,
            )

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch":           epoch,
                    "stage":           stage,
                    "val_loss":        val_loss,
                    "lora_config":     lora_cfg,
                    "vit_state":       vit.state_dict(),
                    "text_enc_state":  text_enc.state_dict(),
                    "logit_scale":     logit_scale.data,
                    "log_lambda_ita":  log_lambda_ita.data,
                    "optimizer_state": optimizer.state_dict(),
                },
                run_dir / "best_checkpoint.pt",
            )
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    torch.save(
        {
            "epoch":          total_epochs,
            "val_loss":       val_loss,
            "lora_config":    lora_cfg,
            "vit_state":      vit.state_dict(),
            "text_enc_state": text_enc.state_dict(),
            "logit_scale":    logit_scale.data,
            "log_lambda_ita": log_lambda_ita.data,
        },
        run_dir / "final_checkpoint.pt",
    )
    print(f"\nDone. Best val loss: {best_val_loss:.4f}. Checkpoints: {run_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
