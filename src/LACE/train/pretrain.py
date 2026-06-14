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
from LACE.data.datasets import BTXRDOrthoDataset, InternalTripleDataset
from LACE.data.transforms import build_train_transform_lace
from LACE.eval.retrieval import evaluate_retrieval_lace
from LACE.loss.objectives import gloria_local_loss, ortho_loss, symmetric_soft_semantic_loss
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
    if text_mode == "phrase":
        beur_pids  = batch["beur_phrase_ids"].to(device)
        beur_pattn = batch["beur_phrase_attn"].to(device)
        beur_pmask = batch["beur_phrase_mask"].to(device)
        bef_pids   = batch["bef_phrase_ids"].to(device)
        bef_pattn  = batch["bef_phrase_attn"].to(device)
        bef_pmask  = batch["bef_phrase_mask"].to(device)
        z_beur_phrases = text_enc._encode_phrase_batch(beur_pids, beur_pattn, beur_pmask)
        z_bef_phrases  = text_enc._encode_phrase_batch(bef_pids,  bef_pattn,  bef_pmask)
    elif text_mode == "mixed":
        beur_ids  = batch["beurteilung_ids"].to(device)
        beur_mask = batch["beurteilung_mask"].to(device)
        z_beur    = text_enc.encode_beurteilung(beur_ids, beur_mask)
        z_beur_phrases = z_beur.unsqueeze(1)                                      # [B, 1, D]
        beur_pmask = batch["has_beurteilung"].unsqueeze(1).to(device)             # [B, 1]
        bef_pids  = batch["bef_phrase_ids"].to(device)
        bef_pattn = batch["bef_phrase_attn"].to(device)
        bef_pmask = batch["bef_phrase_mask"].to(device)
        z_bef_phrases = text_enc._encode_phrase_batch(bef_pids, bef_pattn, bef_pmask)
    else:
        beur_ids  = batch["beurteilung_ids"].to(device)
        beur_mask = batch["beurteilung_mask"].to(device)
        z_text = text_enc.encode_beurteilung(beur_ids, beur_mask)
        z_beur_phrases = z_text.unsqueeze(1)                                      # [B, 1, D]
        beur_pmask = batch["has_beurteilung"].unsqueeze(1).to(device)             # [B, 1]
        bef_ids   = batch["befund_ids"].to(device)
        bef_mask  = batch["befund_mask"].to(device)
        z_bef_cls, _ = text_enc.encode_befund(bef_ids, bef_mask)
        z_bef_phrases = z_bef_cls.unsqueeze(1)                                    # [B, 1, D]
        bef_pmask = batch["has_befund"].unsqueeze(1).to(device)                   # [B, 1]

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
    log_lambda_sim: nn.Parameter,
    log_lambda_ortho: nn.Parameter,
    trainable_params: list,
    device: torch.device,
    active_losses: set[str],
    text_mode: str = "phrase",
    max_grad_norm: float = 1.0,
    τ_s_beur: float = 0.015,
    τ_s_bef: float = 0.07,
    τ_s_img_full: float = 0.07,
    τ2: float = 0.07,
    λ_t2i: float = 1.0,
    same_image_boost: float = 0.0,
    reweight_by_n_phrases: bool = False,
    t2i_mode: str = "image_image",
    sim_lesion_only: bool = False,
) -> dict[str, float]:
    vit.train()
    text_enc.train()

    total_ita = total_ita_i2t = total_ita_t2i = 0.0
    total_sim = total_sim_i2t = total_sim_t2i = 0.0
    total_ortho = total_total = 0.0
    n_batches = 0

    use_text  = "ita" in active_losses or "sim" in active_losses
    btxrd_cycle = itertools.cycle(btxrd_loader) if btxrd_loader is not None else None

    pbar = tqdm(internal_loader, desc="  train", leave=False,
                disable=not sys.stdout.isatty())

    for batch in pbar:
        global_img = batch["global_crop"].to(device)
        crop_img   = batch["crop_image"].to(device)
        plabels    = batch["patch_labels"].to(device)
        crop_plabels = batch["crop_patch_labels"].to(device)
        has_mask   = batch["has_mask"].to(device)
        has_befund = batch["has_befund"].to(device)
        desc_vec   = batch["descriptor_vec"].to(device)               # [B, 21]

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):

            # Single pass: global CLS for L_ITA + global patches for L_ortho
            cls_raw, patch_feat = vit.forward_all(global_img)
            z_img = vit.img_proj(cls_raw)                                  # [B, D]

            if use_text:
                z_beur_phrases, beur_pmask, z_bef_phrases, bef_pmask = _encode_text(
                    batch, text_enc, device, text_mode
                )

            # L_ITA: global CLS (I2T) + beurteilung phrases (T2I), anchored by
            #        phrase-phrase (τ_s_beur) and global-image-image (τ_s_img_full)
            l_ita = l_ita_i2t = l_ita_t2i = torch.zeros(1, device=device)[0]
            if "ita" in active_losses:
                p2i_ita = _phrase_to_image(beur_pmask)
                l_ita, l_ita_i2t, l_ita_t2i = symmetric_soft_semantic_loss(
                    z_img[p2i_ita], z_beur_phrases, beur_pmask, τ,
                    τ_s_phrase=τ_s_beur, τ_s_img=τ_s_img_full,
                    phrase_to_image=p2i_ita, same_image_boost=same_image_boost,
                    reweight_by_n_phrases=reweight_by_n_phrases,
                    t2i_mode=t2i_mode,
                    descriptor_features=desc_vec[p2i_ita],
                )

            # L_sim: GLoRIA-style phrase-patch local alignment on tight lesion crop.
            #        I2T soft (phrase-phrase targets) + T2I hard InfoNCE.
            l_sim     = torch.zeros(1, device=device)[0]
            l_sim_i2t = torch.zeros(1, device=device)[0]
            l_sim_t2i = torch.zeros(1, device=device)[0]
            if "sim" in active_losses:
                sim_valid = has_mask & has_befund
                if sim_valid.sum() >= 2:
                    _, crop_patches_raw = vit.forward_all(crop_img[sim_valid])
                    B_sim = int(sim_valid.sum())
                    P_proj = vit.patch_proj(
                        crop_patches_raw.reshape(-1, 768)
                    ).reshape(B_sim, 196, vit.proj_dim)                    # [B_sim, 196, D]
                    bef_phr_sub   = z_bef_phrases[sim_valid]
                    bef_pmask_sub = bef_pmask[sim_valid]
                    p2i_sim       = _phrase_to_image(bef_pmask_sub)
                    l_sim, l_sim_i2t, l_sim_t2i = gloria_local_loss(
                        P_proj, bef_phr_sub, bef_pmask_sub, τ,
                        τ2=τ2, τ_s=τ_s_bef, τ_s_img=τ_s_img_full,
                        phrase_to_image=p2i_sim,
                        same_image_boost=same_image_boost,
                        reweight_by_n_phrases=reweight_by_n_phrases,
                        λ_t2i=λ_t2i,
                        t2i_mode=t2i_mode,
                        img_cls_features=z_img[sim_valid],
                        descriptor_features=desc_vec[sim_valid],
                        patch_mask=(crop_plabels[sim_valid] if sim_lesion_only else None),
                    )

            # L_ortho: lesion-background patch orthogonality
            l_ortho = torch.zeros(1, device=device)[0]
            if "ortho" in active_losses:
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

            loss = (log_lambda_ita.exp() * l_ita
                    + log_lambda_sim.exp() * l_sim
                    + log_lambda_ortho.exp() * l_ortho)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_ita     += l_ita.item()
        total_ita_i2t += l_ita_i2t.item()
        total_ita_t2i += l_ita_t2i.item()
        total_sim     += l_sim.item()
        total_sim_i2t += l_sim_i2t.item()
        total_sim_t2i += l_sim_t2i.item()
        total_ortho   += l_ortho.item()
        total_total   += loss.item()
        n_batches     += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", ita=f"{l_ita.item():.4f}")

    d = max(1, n_batches)
    return {
        "train/loss":       total_total   / d,
        "train/l_ita":      total_ita     / d,
        "train/l_ita_i2t":  total_ita_i2t / d,
        "train/l_ita_t2i":  total_ita_t2i / d,
        "train/l_sim":      total_sim     / d,
        "train/l_sim_i2t":  total_sim_i2t / d,
        "train/l_sim_t2i":  total_sim_t2i / d,
        "train/l_ortho":    total_ortho   / d,
        "train/lambda_ita": log_lambda_ita.exp().item(),
        "train/lambda_sim": log_lambda_sim.exp().item(),
        "train/lambda_ortho": log_lambda_ortho.exp().item(),
        "train/tau":        τ.item(),
    }


@torch.no_grad()
def evaluate_lace(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    val_loader: DataLoader,
    τ: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    log_lambda_sim: nn.Parameter,
    log_lambda_ortho: nn.Parameter,
    device: torch.device,
    active_losses: set[str],
    text_mode: str = "phrase",
    τ_s_beur: float = 0.015,
    τ_s_bef: float = 0.07,
    τ_s_img_full: float = 0.07,
    τ2: float = 0.07,
    λ_t2i: float = 1.0,
    same_image_boost: float = 0.0,
    reweight_by_n_phrases: bool = False,
    t2i_mode: str = "image_image",
    sim_lesion_only: bool = False,
) -> dict[str, float]:
    vit.eval()
    text_enc.eval()

    total_ita = total_ita_i2t = total_ita_t2i = 0.0
    total_sim = total_sim_i2t = total_sim_t2i = 0.0
    total_ortho = total_total = 0.0
    n_batches = 0

    use_text = "ita" in active_losses or "sim" in active_losses

    for batch in val_loader:
        global_img = batch["global_crop"].to(device)
        crop_img   = batch["crop_image"].to(device)
        plabels    = batch["patch_labels"].to(device)
        crop_plabels = batch["crop_patch_labels"].to(device)
        has_mask   = batch["has_mask"].to(device)
        has_befund = batch["has_befund"].to(device)
        desc_vec   = batch["descriptor_vec"].to(device)               # [B, 21]

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            cls_raw, patch_feat = vit.forward_all(global_img)
            z_img = vit.img_proj(cls_raw)                                  # [B, D]

            if use_text:
                z_beur_phrases, beur_pmask, z_bef_phrases, bef_pmask = _encode_text(
                    batch, text_enc, device, text_mode
                )

            l_ita = l_ita_i2t = l_ita_t2i = torch.zeros(1, device=device)[0]
            if "ita" in active_losses:
                p2i_ita = _phrase_to_image(beur_pmask)
                l_ita, l_ita_i2t, l_ita_t2i = symmetric_soft_semantic_loss(
                    z_img[p2i_ita], z_beur_phrases, beur_pmask, τ,
                    τ_s_phrase=τ_s_beur, τ_s_img=τ_s_img_full,
                    phrase_to_image=p2i_ita, same_image_boost=same_image_boost,
                    reweight_by_n_phrases=reweight_by_n_phrases,
                    t2i_mode=t2i_mode,
                    descriptor_features=desc_vec[p2i_ita],
                )

            l_sim     = torch.zeros(1, device=device)[0]
            l_sim_i2t = torch.zeros(1, device=device)[0]
            l_sim_t2i = torch.zeros(1, device=device)[0]
            if "sim" in active_losses:
                sim_valid = has_mask & has_befund
                if sim_valid.sum() >= 2:
                    _, crop_patches_raw = vit.forward_all(crop_img[sim_valid])
                    B_sim = int(sim_valid.sum())
                    P_proj = vit.patch_proj(
                        crop_patches_raw.reshape(-1, 768)
                    ).reshape(B_sim, 196, vit.proj_dim)                    # [B_sim, 196, D]
                    bef_phr_sub   = z_bef_phrases[sim_valid]
                    bef_pmask_sub = bef_pmask[sim_valid]
                    p2i_sim       = _phrase_to_image(bef_pmask_sub)
                    l_sim, l_sim_i2t, l_sim_t2i = gloria_local_loss(
                        P_proj, bef_phr_sub, bef_pmask_sub, τ,
                        τ2=τ2, τ_s=τ_s_bef, τ_s_img=τ_s_img_full,
                        phrase_to_image=p2i_sim,
                        same_image_boost=same_image_boost,
                        reweight_by_n_phrases=reweight_by_n_phrases,
                        λ_t2i=λ_t2i,
                        t2i_mode=t2i_mode,
                        img_cls_features=z_img[sim_valid],
                        descriptor_features=desc_vec[sim_valid],
                        patch_mask=(crop_plabels[sim_valid] if sim_lesion_only else None),
                    )

            l_ortho = torch.zeros(1, device=device)[0]
            if "ortho" in active_losses:
                l_ortho = ortho_loss(patch_feat, plabels)

            loss = (log_lambda_ita.exp() * l_ita
                    + log_lambda_sim.exp() * l_sim
                    + log_lambda_ortho.exp() * l_ortho)

        total_ita     += l_ita.item()
        total_ita_i2t += l_ita_i2t.item()
        total_ita_t2i += l_ita_t2i.item()
        total_sim     += l_sim.item()
        total_sim_i2t += l_sim_i2t.item()
        total_sim_t2i += l_sim_t2i.item()
        total_ortho   += l_ortho.item()
        total_total   += loss.item()
        n_batches     += 1

    d = max(1, n_batches)
    return {
        "val/loss":       total_total   / d,
        "val/l_ita":      total_ita     / d,
        "val/l_ita_i2t":  total_ita_i2t / d,
        "val/l_ita_t2i":  total_ita_t2i / d,
        "val/l_sim":      total_sim     / d,
        "val/l_sim_i2t":  total_sim_i2t / d,
        "val/l_sim_t2i":  total_sim_t2i / d,
        "val/l_ortho":    total_ortho   / d,
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
    parser.add_argument("--no_btxrd", action="store_true",
                        help="Disable BTXRD ortho dataset (L_ortho uses only internal patches).")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))

    parser.add_argument("--lora_layers", type=int,   default=4)
    parser.add_argument("--lora_r",      type=int,   default=8)
    parser.add_argument("--lora_alpha",  type=float, default=None,
                        help="LoRA scaling alpha. If omitted, defaults to 2*lora_r "
                             "(the common alpha=2r convention, effective scale alpha/r=2).")
    parser.add_argument("--embed_dim",   type=int,   default=512,
                        help="Projection head output dimension (default: 512)")
    parser.add_argument("--warm_start_projections", action="store_true",
                        help="Init projection heads from pretrained BiomedCLIP weights.")

    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--btxrd_batch_size", type=int,   default=16)
    parser.add_argument("--max_text_len",     type=int,   default=128)

    parser.add_argument(
        "--text_mode", default="phrase",
        choices=["full", "concat", "phrase", "mixed"],
        help="Text encoding strategy for pretraining (default: phrase).",
    )
    parser.add_argument("--max_bef_phrases",   type=int, default=16,
                        help="Max befund phrases per sample (phrase/mixed modes).")
    parser.add_argument("--max_beur_phrases",  type=int, default=16,
                        help="Max beurteilung phrases per sample (phrase mode only).")
    parser.add_argument("--max_beur_text_len", type=int, default=256,
                        help="Max token length for beurteilung full text (mixed mode, default: 256).")

    parser.add_argument("--epochs",        type=int, default=100)
    parser.add_argument("--patience",      type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--scheduler",    default="constant", choices=["constant", "cosine"])
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--lambda_ita",   type=float, default=1.0)
    parser.add_argument("--lambda_sim",   type=float, default=1.0)
    parser.add_argument("--lambda_ortho",   type=float, default=0.1)
    parser.add_argument("--tau_s_beur",     type=float, default=0.04,
                        help="Soft-target temperature for beurteilung text anchor (I2T). "
                             "Default 0.04 suits full-text mixed mode; phrase mode may prefer lower values.")
    parser.add_argument("--tau_s_bef",      type=float, default=0.07,
                        help="Soft-target temperature for befund phrase-phrase anchor (I2T, default: 0.07).")
    parser.add_argument("--tau_s_img_full", type=float, default=0.07,
                        help="Soft-target temperature for full-image-image anchor (T2I, default: 0.07).")
    parser.add_argument("--same_image_boost", type=float, default=0.0,
                        help="Logit boost added to same-image phrase pairs in the I2T "
                             "soft target (0.0 = original behaviour)")
    parser.add_argument("--reweight_by_n_phrases", action="store_true",
                        help="Weight each phrase's loss contribution by 1/n_phrases_i so "
                             "every image contributes equally regardless of phrase count.")
    parser.add_argument("--sim_lesion_only", action="store_true",
                        help="Restrict L_sim phrase-patch attention to lesion patches "
                             "(tight-crop mask); background patches are masked out.")
    parser.add_argument(
        "--t2i_mode", default="image_image",
        choices=["image_image", "text_text", "descriptor", "infonce"],
        help="T2I mode for both L_ITA and L_sim: 'image_image' = image-image cosine sim (soft KL); "
             "'text_text' = phrase-phrase cosine sim (soft KL); "
             "'descriptor' = 21-dim binary descriptor cosine sim (soft KL, no temperature); "
             "'infonce' = hard InfoNCE with diagonal positive.",
    )
    parser.add_argument(
        "--descriptor_vectors",
        default=str(ROOT_DIR / "data" / "internal_dataset" / "text" / "descriptor_vectors.json"),
        help="Path to descriptor_vectors.json (keyed by image path). "
             "Required when --t2i_mode descriptor.",
    )
    parser.add_argument("--global_context_fraction", type=float, default=0.4,
                        help="Context fraction for the global crop (default 0.4).")
    parser.add_argument("--context_fraction", type=float, default=0.15,
                        help="Context fraction for the tight tumor crop used as crop_image (default 0.15).")
    parser.add_argument("--tau2", type=float, default=0.07,
                        help="Attention softmax temperature for GLoRIA-style patch attention (default: 0.07).")
    parser.add_argument("--lambda_t2i", type=float, default=1.0,
                        help="Weight for T2I hard InfoNCE in L_sim (default: 1.0).")
    parser.add_argument("--learn_loss_weights", action="store_true",
                        help="Make λ_ita, λ_sim, λ_ortho learnable log-scale parameters.")
    parser.add_argument(
        "--losses", nargs="+", default=["ita", "sim", "ortho"],
        choices=["ita", "sim", "ortho"],
        help="Active loss terms (default: ita sim ortho).",
    )

    parser.add_argument("--downstream_train_frac", type=float, default=0.8)
    parser.add_argument("--downstream_val_frac",   type=float, default=0.1)
    parser.add_argument("--test_frac",             type=float, default=0.1)
    parser.add_argument("--seed",                  type=int,   default=42)

    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="lace-pretrain")
    parser.add_argument("--wandb_entity",  default=None)
    parser.add_argument("--wandb_run",     default=None)

    args = parser.parse_args(argv)
    if args.lora_alpha is None:
        args.lora_alpha = 2.0 * args.lora_r   # alpha=2r convention (effective scale alpha/r=2)
    return args


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    active_losses: set[str] = set(args.losses)
    print(f"Active losses: {sorted(active_losses)}")

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
    τ              = nn.Parameter(torch.tensor(0.07, device=device))
    log_lambda_ita = nn.Parameter(
        torch.tensor(math.log(args.lambda_ita), device=device),
        requires_grad=args.learn_loss_weights and "ita" in active_losses,
    )
    log_lambda_sim = nn.Parameter(
        torch.tensor(math.log(args.lambda_sim), device=device),
        requires_grad=args.learn_loss_weights and "sim" in active_losses,
    )
    log_lambda_ortho = nn.Parameter(
        torch.tensor(math.log(args.lambda_ortho), device=device),
        requires_grad=args.learn_loss_weights and "ortho" in active_losses,
    )

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
        val_samples      = split_data["val"]
        print(f"Loaded split from {args.splits} "
              f"({len(pretrain_samples)} train, {len(val_samples)} val samples)")
    else:
        pretrain_samples, val_samples, _test = build_stratified_splits(args, run_dir=run_dir)

    with open(args.descriptor_vectors, encoding="utf-8") as fh:
        descriptor_vectors = json.load(fh)
    print(f"Loaded {len(descriptor_vectors)} descriptor vectors from {args.descriptor_vectors}")

    preprocess_val   = vit.preprocess_val
    preprocess_train = build_train_transform_lace(preprocess_val)
    tokenizer        = text_enc.tokenizer

    ds_kwargs = dict(
        max_text_len=args.max_text_len,
        text_mode=args.text_mode,
        max_bef_phrases=args.max_bef_phrases,
        max_beur_phrases=args.max_beur_phrases,
        global_context_fraction=args.global_context_fraction,
        context_fraction=args.context_fraction,
        max_beur_text_len=args.max_beur_text_len,
        descriptor_vectors=descriptor_vectors,
    )
    train_ds = InternalTripleDataset(pretrain_samples, preprocess_train, tokenizer, is_train=True,  **ds_kwargs)
    val_ds   = InternalTripleDataset(val_samples,      preprocess_val,   tokenizer, is_train=False, **ds_kwargs)
    print(f"Pretrain datasets: {len(train_ds)} train / {len(val_ds)} val")

    use_pin = device.type == "cuda"
    internal_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=use_pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=use_pin,
    )
    if "ortho" in active_losses and not args.no_btxrd:
        btxrd_ds = BTXRDOrthoDataset(
            Path(args.btxrd_images), Path(args.btxrd_annots), preprocess_val,
        )
        btxrd_loader = DataLoader(
            btxrd_ds, batch_size=args.btxrd_batch_size, shuffle=True,
            num_workers=4, pin_memory=use_pin, drop_last=True,
        ) if len(btxrd_ds) > 0 else None
    else:
        btxrd_loader = None
        if "ortho" not in active_losses:
            print("BTXRD dataset disabled (ortho loss not active).")
        else:
            print("BTXRD dataset disabled (--no_btxrd).")

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
    decay_params    = vit_decay + txt_decay
    lambda_params   = [log_lambda_ita, log_lambda_sim, log_lambda_ortho] \
                      if args.learn_loss_weights else []
    no_decay_params = vit_no_decay + txt_no_decay + [τ] + lambda_params

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
    best_val_loss       = float("inf")
    best_mean_retrieval = -float("inf")
    patience_counter    = 0

    print(f"\nStarting LACE v1 training: {args.epochs} epochs, patience={args.patience}\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            vit, text_enc, internal_loader, btxrd_loader,
            optimizer, scaler, τ,
            log_lambda_ita, log_lambda_sim, log_lambda_ortho,
            trainable_params,
            device=device,
            active_losses=active_losses,
            text_mode=args.text_mode,
            τ_s_beur=args.tau_s_beur,
            τ_s_bef=args.tau_s_bef,
            τ_s_img_full=args.tau_s_img_full,
            τ2=args.tau2,
            λ_t2i=args.lambda_t2i,
            same_image_boost=args.same_image_boost,
            reweight_by_n_phrases=args.reweight_by_n_phrases,
            t2i_mode=args.t2i_mode,
            sim_lesion_only=args.sim_lesion_only,
        )
        val_metrics = evaluate_lace(
            vit, text_enc, val_loader, τ,
            log_lambda_ita, log_lambda_sim, log_lambda_ortho,
            device=device,
            active_losses=active_losses,
            text_mode=args.text_mode,
            τ_s_beur=args.tau_s_beur,
            τ_s_bef=args.tau_s_bef,
            τ_s_img_full=args.tau_s_img_full,
            τ2=args.tau2,
            λ_t2i=args.lambda_t2i,
            same_image_boost=args.same_image_boost,
            reweight_by_n_phrases=args.reweight_by_n_phrases,
            t2i_mode=args.t2i_mode,
            sim_lesion_only=args.sim_lesion_only,
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

        def _checkpoint_dict(extra: dict) -> dict:
            return {
                "epoch":           epoch,
                "val_loss":        val_loss,
                "lora_config":     lora_cfg,
                "vit_state":       vit.state_dict(),
                "text_enc_state":  text_enc.state_dict(),
                "tau":             τ.data,
                "log_lambda_ita":  log_lambda_ita.data,
                "log_lambda_sim":  log_lambda_sim.data,
                "log_lambda_ortho":  log_lambda_ortho.data,
                "optimizer_state": optimizer.state_dict(),
                **extra,
            }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = run_dir / "best_val_loss_checkpoint.pt"
            torch.save(_checkpoint_dict({"best_val_loss": best_val_loss}), ckpt_path)
            print(f"  Saved checkpoint -> {ckpt_path}")

        if retrieval:
            mean_r1 = (retrieval["retrieval/i2t_r1"] + retrieval["retrieval/t2i_r1"]) / 2
            if mean_r1 > best_mean_retrieval:
                best_mean_retrieval = mean_r1
                patience_counter    = 0
                ckpt_path = run_dir / "best_retrieval_checkpoint.pt"
                torch.save(_checkpoint_dict({"best_mean_r1": best_mean_retrieval, **retrieval}), ckpt_path)
                print(f"  Saved checkpoint -> {ckpt_path}")
            else:
                patience_counter += 1
                print(f"  No improvement in mean R@1 ({patience_counter}/{args.patience}) "
                      f"(current={mean_r1:.1%}, best={best_mean_retrieval:.1%})")
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(mean R@1 did not improve for {args.patience} epochs).\n")
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
            "log_lambda_sim": log_lambda_sim.data,
            "log_lambda_ortho": log_lambda_ortho.data,
        },
        run_dir / "final_checkpoint.pt",
    )
    print(
        f"\nDone. Best val loss: {best_val_loss:.4f} | "
        f"Best mean R@1: {best_mean_retrieval:.1%}. Checkpoints: {run_dir}"
    )
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
