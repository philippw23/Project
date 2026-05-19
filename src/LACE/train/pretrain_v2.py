"""LACE v2 pretraining: mask-token architecture with three-stage curriculum.

Stage 1: L_ITA only (global image-text alignment)
Stage 2: L_ITA + L_seg (mask tokens supervised; BTXRD active)
Stage 3: L_ITA + L_seg + L_sim (local phrase alignment via heatmap pooling)
"""
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
from LACE.data.splits import build_pretrain_datasets_lace_v2
from LACE.data.transforms import build_train_transform_lace
from LACE.loss.objectives import ita_loss, seg_loss, sim_loss_v2
from LACE.models.encoders import BiomedCLIPTextEncoder, SharedViT
from LACE.models.mask_tokens import MaskTokenModule

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


def train_one_epoch(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    mask_module: MaskTokenModule,
    internal_loader: DataLoader,
    btxrd_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    logit_scale: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    trainable_params: list,
    device: torch.device,
    stage: int,
    lambda_seg: float,
    lambda_sim: float,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    vit.train()
    text_enc.train()
    mask_module.train()

    total_ita = total_seg = total_sim = total_total = 0.0
    n_batches = 0

    btxrd_cycle = (
        itertools.cycle(btxrd_loader)
        if (stage >= 2 and btxrd_loader is not None) else None
    )

    pbar = tqdm(internal_loader, desc=f"  stage{stage}", leave=False,
                disable=not sys.stdout.isatty())

    zero = torch.zeros(1, device=device)[0]

    for batch in pbar:
        img       = batch["full_image"].to(device)
        beur_ids  = batch["beurteilung_ids"].to(device)
        beur_mask = batch["beurteilung_mask"].to(device)
        bef_ids   = batch["befund_ids"].to(device)
        bef_mask  = batch["befund_mask"].to(device)
        plabels   = batch["patch_labels"].to(device)
        has_mask  = batch["has_mask"].to(device)
        has_befund= batch["has_befund"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):

            # Single ViT pass: CLS for L_ITA, patches for mask tokens
            cls_feat, patch_feat = vit.forward_all(img)          # [B,768], [B,196,768]
            z_img  = vit.img_proj(cls_feat)                       # [B, 256]

            # L_ITA (all stages)
            z_text = text_enc.encode_beurteilung(beur_ids, beur_mask)
            l_ita  = ita_loss(z_img, z_text, logit_scale)

            l_seg = zero
            l_sim = zero

            if stage >= 2:
                H_soft, H_logits, _ = mask_module(patch_feat)    # [B,N,196] each

                # L_seg from internal Dataset A (samples with GT masks)
                seg_valid = has_mask
                if seg_valid.any():
                    l_seg = seg_loss(H_logits[seg_valid], plabels[seg_valid])

                # L_seg from Dataset B (BTXRD annotated bone tumour images)
                if btxrd_cycle is not None:
                    btxrd_b       = next(btxrd_cycle)
                    btxrd_img     = btxrd_b["image"].to(device)
                    btxrd_labels  = btxrd_b["patch_labels"].to(device)
                    _, btxrd_feat = vit.forward_all(btxrd_img)
                    _, H_b_logits, _ = mask_module(btxrd_feat)
                    l_seg = l_seg + seg_loss(H_b_logits, btxrd_labels)

                # L_sim (stage 3+): project patches into phrase-alignment space
                if stage >= 3:
                    B = img.shape[0]
                    P_proj = vit.patch_proj(
                        patch_feat.reshape(-1, 768)
                    ).reshape(B, 196, vit.proj_dim)

                    _, proj_words = text_enc.encode_befund(bef_ids, bef_mask)
                    sim_valid = has_befund
                    if sim_valid.sum() >= 2:
                        l_sim = sim_loss_v2(
                            P_proj[sim_valid],
                            proj_words[sim_valid],
                            H_soft[sim_valid],
                            logit_scale,
                        )

            loss = log_lambda_ita.exp() * l_ita + lambda_seg * l_seg + lambda_sim * l_sim

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_ita   += l_ita.item()
        total_seg   += l_seg.item()
        total_sim   += l_sim.item()
        total_total += loss.item()
        n_batches   += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", ita=f"{l_ita.item():.4f}")

    d = max(1, n_batches)
    return {
        "train/loss":       total_total / d,
        "train/l_ita":      total_ita   / d,
        "train/l_seg":      total_seg   / d,
        "train/l_sim":      total_sim   / d,
        "train/lambda_ita": log_lambda_ita.exp().item(),
    }


@torch.no_grad()
def evaluate(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    mask_module: MaskTokenModule,
    val_loader: DataLoader,
    logit_scale: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    device: torch.device,
    stage: int,
    lambda_seg: float,
    lambda_sim: float,
) -> dict[str, float]:
    vit.eval()
    text_enc.eval()
    mask_module.eval()

    total_ita = total_seg = total_sim = total_total = 0.0
    n_batches = 0
    zero = torch.zeros(1, device=device)[0]

    for batch in val_loader:
        img       = batch["full_image"].to(device)
        beur_ids  = batch["beurteilung_ids"].to(device)
        beur_mask = batch["beurteilung_mask"].to(device)
        bef_ids   = batch["befund_ids"].to(device)
        bef_mask  = batch["befund_mask"].to(device)
        plabels   = batch["patch_labels"].to(device)
        has_mask  = batch["has_mask"].to(device)
        has_befund= batch["has_befund"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            cls_feat, patch_feat = vit.forward_all(img)
            z_img  = vit.img_proj(cls_feat)
            z_text = text_enc.encode_beurteilung(beur_ids, beur_mask)
            l_ita  = ita_loss(z_img, z_text, logit_scale)

            l_seg = zero
            l_sim = zero

            if stage >= 2:
                H_soft, H_logits, _ = mask_module(patch_feat)
                seg_valid = has_mask
                if seg_valid.any():
                    l_seg = seg_loss(H_logits[seg_valid], plabels[seg_valid])

                if stage >= 3:
                    B = img.shape[0]
                    P_proj = vit.patch_proj(
                        patch_feat.reshape(-1, 768)
                    ).reshape(B, 196, vit.proj_dim)
                    _, proj_words = text_enc.encode_befund(bef_ids, bef_mask)
                    sim_valid = has_befund
                    if sim_valid.sum() >= 2:
                        l_sim = sim_loss_v2(
                            P_proj[sim_valid], proj_words[sim_valid],
                            H_soft[sim_valid], logit_scale,
                        )

            loss = log_lambda_ita.exp() * l_ita + lambda_seg * l_seg + lambda_sim * l_sim

        total_ita   += l_ita.item()
        total_seg   += l_seg.item()
        total_sim   += l_sim.item()
        total_total += loss.item()
        n_batches   += 1

    d = max(1, n_batches)
    return {
        "val/loss":  total_total / d,
        "val/l_ita": total_ita   / d,
        "val/l_seg": total_seg   / d,
        "val/l_sim": total_sim   / d,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LACE v2 pretraining (mask-token architecture)")

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

    parser.add_argument("--n_mask_tokens", type=int,   default=16)
    parser.add_argument("--tau_spatial",   type=float, default=0.1,
                        help="Initial spatial temperature for MaskTokenModule")

    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--btxrd_batch_size", type=int,   default=16)
    parser.add_argument("--max_text_len",     type=int,   default=128)

    parser.add_argument("--stage1_epochs", type=int,   default=10)
    parser.add_argument("--stage2_epochs", type=int,   default=15)
    parser.add_argument("--stage3_epochs", type=int,   default=15)

    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--lambda_seg",   type=float, default=1.0)
    parser.add_argument("--lambda_sim",   type=float, default=1.0)
    parser.add_argument("--patience",     type=int,   default=20,
                        help="Early stopping patience (0 to disable)")

    parser.add_argument("--downstream_train_frac", type=float, default=0.8)
    parser.add_argument("--downstream_val_frac",   type=float, default=0.1)
    parser.add_argument("--test_frac",             type=float, default=0.1)
    parser.add_argument("--seed",                  type=int,   default=42)

    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="lace-v2-pretrain")
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
    vit         = SharedViT(args.lora_layers, args.lora_r, args.lora_alpha, args.embed_dim).to(device)
    text_enc    = BiomedCLIPTextEncoder(embed_dim=args.embed_dim).to(device)
    mask_module = MaskTokenModule(
        n_tokens=args.n_mask_tokens,
        tau_spatial_init=args.tau_spatial,
    ).to(device)
    logit_scale    = nn.Parameter(torch.ones([], device=device) * math.log(1.0 / 0.07))
    log_lambda_ita = nn.Parameter(torch.zeros([], device=device))

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = (
        Path(args.out_dir) / "lace_v2_pretrain" /
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

    train_ds, val_ds = build_pretrain_datasets_lace_v2(
        pretrain_samples, preprocess_train, preprocess_val,
        tokenizer, args.seed, max_text_len=args.max_text_len,
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

    vit_decay,  vit_nd  = _split_params(vit)
    txt_decay,  txt_nd  = _split_params(text_enc)
    msk_decay,  msk_nd  = _split_params(mask_module)

    decay_params    = vit_decay  + txt_decay  + msk_decay  + [logit_scale, log_lambda_ita]
    no_decay_params = vit_nd     + txt_nd     + msk_nd

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
        if epoch <= s1_end:
            return 1
        if epoch <= s2_end:
            return 2
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

    print(
        f"\nStarting LACE v2 training: {total_epochs} epochs "
        f"(stage1={args.stage1_epochs}, stage2={args.stage2_epochs}, "
        f"stage3={args.stage3_epochs})\n"
    )

    for epoch in range(1, total_epochs + 1):
        stage        = _get_stage(epoch)
        active_btxrd = btxrd_loader if stage >= 2 else None

        train_metrics = train_one_epoch(
            vit, text_enc, mask_module, internal_loader, active_btxrd,
            optimizer, scaler, logit_scale, log_lambda_ita, trainable_params,
            device=device, stage=stage,
            lambda_seg=args.lambda_seg, lambda_sim=args.lambda_sim,
        )
        val_metrics = evaluate(
            vit, text_enc, mask_module, val_loader,
            logit_scale, log_lambda_ita,
            device=device, stage=stage,
            lambda_seg=args.lambda_seg, lambda_sim=args.lambda_sim,
        )
        scheduler.step()

        val_loss   = val_metrics["val/loss"]
        lr_current = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d}/{total_epochs} [stage {stage}] | "
            f"train={train_metrics['train/loss']:.4f} | val={val_loss:.4f} | "
            f"ita={val_metrics['val/l_ita']:.4f} | "
            f"seg={val_metrics['val/l_seg']:.4f} | "
            f"sim={val_metrics['val/l_sim']:.4f} | "
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
                    "epoch":              epoch,
                    "stage":              stage,
                    "val_loss":           val_loss,
                    "lora_config":        lora_cfg,
                    "n_mask_tokens":      args.n_mask_tokens,
                    "vit_state":          vit.state_dict(),
                    "text_enc_state":     text_enc.state_dict(),
                    "mask_module_state":  mask_module.state_dict(),
                    "logit_scale":        logit_scale.data,
                    "log_lambda_ita":     log_lambda_ita.data,
                    "optimizer_state":    optimizer.state_dict(),
                },
                run_dir / "best_checkpoint.pt",
            )
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                break

    torch.save(
        {
            "epoch":             total_epochs,
            "val_loss":          val_loss,
            "lora_config":       lora_cfg,
            "n_mask_tokens":     args.n_mask_tokens,
            "vit_state":         vit.state_dict(),
            "text_enc_state":    text_enc.state_dict(),
            "mask_module_state": mask_module.state_dict(),
            "logit_scale":       logit_scale.data,
            "log_lambda_ita":    log_lambda_ita.data,
        },
        run_dir / "final_checkpoint.pt",
    )
    print(f"\nDone. Best val loss: {best_val_loss:.4f}. Checkpoints: {run_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
