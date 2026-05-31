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
import torch.nn.functional as F
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
from LACE.loss.objectives import multi_positive_soft_semantic_loss, seg_loss
from LACE.models.encoders import BiomedCLIPTextEncoder, SharedViT
from LACE.models.mask_tokens import MaskTokenModule

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
    Returns:
        [N_total] int64 — image index for each valid phrase
    """
    b_size, j_size = phrase_mask.shape
    idx = torch.arange(b_size, device=phrase_mask.device).unsqueeze(1).expand(-1, j_size)
    return idx[phrase_mask]


def _encode_text_v2(
    batch: dict,
    text_enc: BiomedCLIPTextEncoder,
    device: torch.device,
    text_mode: str,
    stage: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Encode beurteilung for L_ITA and befund for L_sim.

    Returns:
        z_beur_phrases [B, J, D]    — per-phrase beurteilung embeddings
        beur_pmask     [B, J] bool  — True = valid phrase
        proj_phrases   [B, J, D]    — befund phrase embeddings (None when stage < 3)
        bef_pmask      [B, J] bool  — befund phrase mask (None when stage < 3)
    """
    if text_mode == "phrase":
        beur_ids   = batch["beur_phrase_ids"].to(device)
        beur_attn  = batch["beur_phrase_attn"].to(device)
        beur_pmask = batch["beur_phrase_mask"].to(device)
        z_beur_phrases = text_enc._encode_phrase_batch(beur_ids, beur_attn, beur_pmask)
        if stage >= 3:
            bef_ids   = batch["bef_phrase_ids"].to(device)
            bef_attn  = batch["bef_phrase_attn"].to(device)
            bef_pmask = batch["bef_phrase_mask"].to(device)
            _, proj_phrases = text_enc.encode_befund_phrases(bef_ids, bef_attn, bef_pmask)
        else:
            proj_phrases, bef_pmask = None, None
    else:
        beur_ids  = batch["beurteilung_ids"].to(device)
        beur_mask = batch["beurteilung_mask"].to(device)
        z_text = text_enc.encode_beurteilung(beur_ids, beur_mask)
        z_beur_phrases = z_text.unsqueeze(1)                                       # [B, 1, D]
        beur_pmask = torch.ones(z_beur_phrases.shape[:2], dtype=torch.bool, device=device)
        if stage >= 3:
            bef_ids  = batch["befund_ids"].to(device)
            bef_mask = batch["befund_mask"].to(device)
            _, proj_phrases = text_enc.encode_befund(bef_ids, bef_mask)
            bef_pmask = bef_mask.bool()
        else:
            proj_phrases, bef_pmask = None, None
    return z_beur_phrases, beur_pmask, proj_phrases, bef_pmask


def _compute_crop_features(
    P_proj: torch.Tensor,
    H_fg: torch.Tensor,
    valid_phrases: torch.Tensor,
    phrase_to_image: torch.Tensor,
) -> torch.Tensor:
    """Phrase-specific lesion features via per-image attention loop.

    Avoids [N_total, 196, D] tensor duplication; peak memory is O(n_i * 196 * D)
    per image rather than O(N_total * 196 * D) for the batch.

    Args:
        P_proj:         [B_sim, 196, D]  L2-normalised projected patch embeddings
        H_fg:           [B_sim, 196]     detached lesion heatmap (soft weights)
        valid_phrases:  [N_total, D]     L2-normalised valid befund phrase embeddings
        phrase_to_image:[N_total]        image index for each valid phrase
    Returns:
        [N_total, D] L2-normalised crop features (one per valid phrase)
    """
    b_sim = P_proj.shape[0]
    weighted_patches = H_fg.unsqueeze(-1) * P_proj      # [B_sim, 196, D]

    crop_features_list = []
    for i in range(b_sim):
        phrases_i = valid_phrases[phrase_to_image == i]  # [n_i, D]
        if phrases_i.shape[0] == 0:
            continue
        wp_i = weighted_patches[i]                       # [196, D]
        alpha_i = F.softmax(phrases_i @ wp_i.T, dim=-1)  # [n_i, 196]
        crop_features_list.append(alpha_i @ wp_i)        # [n_i, D]

    return F.normalize(torch.cat(crop_features_list, dim=0), dim=-1)  # [N_total, D]


def train_one_epoch(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    mask_module: MaskTokenModule,
    internal_loader: DataLoader,
    btxrd_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    τ: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    trainable_params: list,
    device: torch.device,
    stage: int,
    lambda_seg: float,
    lambda_sim: float,
    text_mode: str = "phrase",
    max_grad_norm: float = 1.0,
    same_image_boost: float = 0.0,
    reweight_by_n_phrases: bool = False,
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
        img        = batch["full_image"].to(device)
        plabels    = batch["patch_labels"].to(device)
        has_mask   = batch["has_mask"].to(device)
        has_befund = batch["has_befund"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):

            # Single ViT pass: CLS for L_ITA, patches for mask tokens + L_sim
            cls_feat, patch_feat = vit.forward_all(img)          # [B,768], [B,196,768]
            z_img  = vit.img_proj(cls_feat)                       # [B, D]

            z_beur_phrases, beur_pmask, proj_phrases, bef_pmask = _encode_text_v2(
                batch, text_enc, device, text_mode, stage
            )

            # L_ITA (all stages): expand z_img to one row per valid phrase
            p2i_ita = _phrase_to_image(beur_pmask)                # [N_total_ita]
            l_ita = multi_positive_soft_semantic_loss(
                z_img[p2i_ita], z_beur_phrases, beur_pmask, τ,
                phrase_to_image=p2i_ita, same_image_boost=same_image_boost,
                reweight_by_n_phrases=reweight_by_n_phrases,
            )

            l_seg = zero
            l_sim = zero

            if stage >= 2:
                h_soft, h_logits, _ = mask_module(patch_feat)    # [B,N,196] each

                # L_seg from internal dataset (samples with GT masks)
                seg_valid = has_mask
                if seg_valid.any():
                    l_seg = seg_loss(h_logits[seg_valid], plabels[seg_valid])

                # L_seg from BTXRD annotated images
                if btxrd_cycle is not None:
                    btxrd_b       = next(btxrd_cycle)
                    btxrd_img     = btxrd_b["image"].to(device)
                    btxrd_labels  = btxrd_b["patch_labels"].to(device)
                    _, btxrd_feat = vit.forward_all(btxrd_img)
                    _, h_b_logits, _ = mask_module(btxrd_feat)
                    l_seg = l_seg + seg_loss(h_b_logits, btxrd_labels)

                # L_sim v2 (stage 3+): phrase-specific lesion features via heatmap
                if stage >= 3 and proj_phrases is not None:
                    b_size = img.shape[0]
                    p_proj = vit.patch_proj(
                        patch_feat.reshape(-1, 768)
                    ).reshape(b_size, 196, vit.proj_dim)                  # [B, 196, D]

                    sim_valid = has_befund
                    if sim_valid.sum() >= 2:
                        bef_phr_sub   = proj_phrases[sim_valid]           # [B_sim, J, D]
                        bef_pmask_sub = bef_pmask[sim_valid]              # [B_sim, J]
                        p2i_sim       = _phrase_to_image(bef_pmask_sub)   # [N_total_sim]
                        valid_phr     = bef_phr_sub[bef_pmask_sub]        # [N_total_sim, D]

                        h_fg = h_soft[sim_valid].mean(dim=1).detach()     # [B_sim, 196]
                        crop_features = _compute_crop_features(
                            p_proj[sim_valid], h_fg, valid_phr, p2i_sim
                        )                                                  # [N_total_sim, D]

                        l_sim = multi_positive_soft_semantic_loss(
                            crop_features, bef_phr_sub, bef_pmask_sub, τ,
                            phrase_to_image=p2i_sim, same_image_boost=same_image_boost,
                            reweight_by_n_phrases=reweight_by_n_phrases,
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
        "train/tau":        τ.item(),
    }


@torch.no_grad()
def evaluate(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    mask_module: MaskTokenModule,
    val_loader: DataLoader,
    τ: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    device: torch.device,
    lambda_seg: float,
    lambda_sim: float,
    text_mode: str = "phrase",
    same_image_boost: float = 0.0,
    reweight_by_n_phrases: bool = False,
) -> dict[str, float]:
    vit.eval()
    text_enc.eval()
    mask_module.eval()

    total_ita = total_seg = total_sim = total_total = 0.0
    n_batches = 0
    zero = torch.zeros(1, device=device)[0]

    for batch in val_loader:
        img        = batch["full_image"].to(device)
        plabels    = batch["patch_labels"].to(device)
        has_mask   = batch["has_mask"].to(device)
        has_befund = batch["has_befund"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            cls_feat, patch_feat = vit.forward_all(img)
            z_img  = vit.img_proj(cls_feat)

            z_beur_phrases, beur_pmask, proj_phrases, bef_pmask = _encode_text_v2(
                batch, text_enc, device, text_mode, stage=3
            )

            p2i_ita = _phrase_to_image(beur_pmask)
            l_ita = multi_positive_soft_semantic_loss(
                z_img[p2i_ita], z_beur_phrases, beur_pmask, τ,
                phrase_to_image=p2i_ita, same_image_boost=same_image_boost,
                reweight_by_n_phrases=reweight_by_n_phrases,
            )

            l_seg = zero
            l_sim = zero

            h_soft, h_logits, _ = mask_module(patch_feat)
            seg_valid = has_mask
            if seg_valid.any():
                l_seg = seg_loss(h_logits[seg_valid], plabels[seg_valid])

            if proj_phrases is not None:
                b_size = img.shape[0]
                p_proj = vit.patch_proj(
                    patch_feat.reshape(-1, 768)
                ).reshape(b_size, 196, vit.proj_dim)
                sim_valid = has_befund
                if sim_valid.sum() >= 2:
                    bef_phr_sub   = proj_phrases[sim_valid]
                    bef_pmask_sub = bef_pmask[sim_valid]
                    p2i_sim       = _phrase_to_image(bef_pmask_sub)
                    valid_phr     = bef_phr_sub[bef_pmask_sub]

                    h_fg = h_soft[sim_valid].mean(dim=1).detach()
                    crop_features = _compute_crop_features(
                        p_proj[sim_valid], h_fg, valid_phr, p2i_sim
                    )

                    l_sim = multi_positive_soft_semantic_loss(
                        crop_features, bef_phr_sub, bef_pmask_sub, τ,
                        phrase_to_image=p2i_sim, same_image_boost=same_image_boost,
                        reweight_by_n_phrases=reweight_by_n_phrases,
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
    parser.add_argument("--embed_dim",   type=int,   default=512,
                        help="Projection head output dimension (default: 512)")
    parser.add_argument("--warm_start_projections", action="store_true",
                        help="Init projection heads from pretrained BiomedCLIP weights.")

    parser.add_argument("--n_mask_tokens", type=int,   default=16)
    parser.add_argument("--tau_spatial",   type=float, default=0.1,
                        help="Initial spatial temperature for MaskTokenModule")

    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--btxrd_batch_size", type=int,   default=16)
    parser.add_argument("--max_text_len",     type=int,   default=128)

    parser.add_argument(
        "--text_mode", default="phrase",
        choices=["full", "phrase"],
        help="Text encoding strategy (default: phrase).",
    )
    parser.add_argument("--max_bef_phrases",  type=int, default=16,
                        help="Max befund phrases per sample (phrase modes only).")
    parser.add_argument("--max_beur_phrases", type=int, default=16,
                        help="Max beurteilung phrases per sample (phrase modes only).")

    parser.add_argument("--stage1_epochs", type=int,   default=10)
    parser.add_argument("--stage2_epochs", type=int,   default=15)
    parser.add_argument("--stage3_epochs", type=int,   default=15)

    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--scheduler",    default="constant", choices=["constant", "cosine"])
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--lambda_seg",       type=float, default=1.0)
    parser.add_argument("--lambda_sim",       type=float, default=1.0)
    parser.add_argument("--same_image_boost", type=float, default=0.0,
                        help="Logit boost added to same-image phrase pairs in the I2T "
                             "soft target (0.0 = original behaviour)")
    parser.add_argument("--reweight_by_n_phrases", action="store_true",
                        help="Weight each phrase's loss contribution by 1/n_phrases_i so "
                             "every image contributes equally regardless of phrase count.")
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
    vit      = SharedViT(args.lora_layers, args.lora_r, args.lora_alpha, args.embed_dim)
    text_enc = BiomedCLIPTextEncoder(embed_dim=args.embed_dim)
    if args.warm_start_projections:
        vit.load_pretrained_projections()
        text_enc.load_pretrained_projections()
    vit      = vit.to(device)
    text_enc = text_enc.to(device)
    mask_module = MaskTokenModule(
        n_tokens=args.n_mask_tokens,
        tau_spatial_init=args.tau_spatial,
    ).to(device)
    τ              = nn.Parameter(torch.tensor(0.07, device=device))
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

    vit_decay,  vit_nd  = _split_params(vit)
    txt_decay,  txt_nd  = _split_params(text_enc)
    msk_decay,  msk_nd  = _split_params(mask_module)

    decay_params    = vit_decay  + txt_decay  + msk_decay  + [τ, log_lambda_ita]
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
    scheduler     = make_scheduler(optimizer, warmup_epochs, total_epochs, args.scheduler)
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
            optimizer, scaler, τ, log_lambda_ita, trainable_params,
            device=device, stage=stage,
            lambda_seg=args.lambda_seg, lambda_sim=args.lambda_sim,
            text_mode=args.text_mode,
            same_image_boost=args.same_image_boost,
            reweight_by_n_phrases=args.reweight_by_n_phrases,
        )
        val_metrics = evaluate(
            vit, text_enc, mask_module, val_loader,
            τ, log_lambda_ita,
            device=device,
            lambda_seg=args.lambda_seg, lambda_sim=args.lambda_sim,
            text_mode=args.text_mode,
            same_image_boost=args.same_image_boost,
            reweight_by_n_phrases=args.reweight_by_n_phrases,
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
                    "tau":                τ.data,
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
            "tau":               τ.data,
            "log_lambda_ita":    log_lambda_ita.data,
        },
        run_dir / "final_checkpoint.pt",
    )
    print(f"\nDone. Best val loss: {best_val_loss:.4f}. Checkpoints: {run_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
