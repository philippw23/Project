"""LACE v2 pretraining: MaskTokenDecoder with 2-stage curriculum.

Stage 1: L_dice + L_ortho only   (mask decoder learns to find lesion)
Stage 2: L_ITA + L_sim + L_dice + L_ortho  (all losses active)

Set --stage1_epochs 0 to disable curriculum and run all losses from epoch 1.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.data.splits import build_stratified_splits
from biomedclip.data.transforms import crop_around_mask_pair, pad_to_square
from biomedclip.utils.misc import (
    DEFAULT_DATASET_JSON,
    DEFAULT_OUT_DIR,
    DEFAULT_SPLITS,
    ROOT_DIR,
)

from LACE.data.datasets import BTXRDOrthoDataset, InternalDatasetV2
from LACE.data.transforms import build_train_transform_lace
from LACE.eval.retrieval import evaluate_retrieval_lace
from LACE.loss.objectives import (
    compute_l_dice_ce,
    gloria_local_loss,
    ortho_loss,
    select_fg_token,
    symmetric_soft_semantic_loss,
)
from LACE.models.encoders import BiomedCLIPTextEncoder, SharedViT
from LACE.models.mask_tokens import MaskPredictionHead, MaskTokenDecoder

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode beurteilung for L_ITA and befund phrases for L_sim (always encoded).

    Returns:
        z_beur_phrases [B, J, D]   — beurteilung phrase embeddings
        beur_pmask     [B, J] bool
        z_bef_phrases  [B, J, D]   — befund phrase embeddings
        bef_pmask      [B, J] bool
    """
    if text_mode == "phrase":
        z_beur_phrases = text_enc._encode_phrase_batch(
            batch["beur_phrase_ids"].to(device),
            batch["beur_phrase_attn"].to(device),
            batch["beur_phrase_mask"].to(device),
        )
        beur_pmask = batch["beur_phrase_mask"].to(device)
        _, z_bef_phrases = text_enc.encode_befund_phrases(
            batch["bef_phrase_ids"].to(device),
            batch["bef_phrase_attn"].to(device),
            batch["bef_phrase_mask"].to(device),
        )
        bef_pmask = batch["bef_phrase_mask"].to(device)
    elif text_mode == "mixed":
        # Full beurteilung text for L_ITA; befund phrases for L_sim
        z_text = text_enc.encode_beurteilung(
            batch["beurteilung_ids"].to(device), batch["beurteilung_mask"].to(device)
        )
        z_beur_phrases = z_text.unsqueeze(1)
        beur_pmask = torch.ones(
            z_beur_phrases.shape[:2], dtype=torch.bool, device=device
        )
        _, z_bef_phrases = text_enc.encode_befund_phrases(
            batch["bef_phrase_ids"].to(device),
            batch["bef_phrase_attn"].to(device),
            batch["bef_phrase_mask"].to(device),
        )
        bef_pmask = batch["bef_phrase_mask"].to(device)
    else:
        # full-text fallback
        z_text = text_enc.encode_beurteilung(
            batch["beurteilung_ids"].to(device), batch["beurteilung_mask"].to(device)
        )
        z_beur_phrases = z_text.unsqueeze(1)
        beur_pmask = torch.ones(
            z_beur_phrases.shape[:2], dtype=torch.bool, device=device
        )
        _, z_bef_phrases = text_enc.encode_befund(
            batch["befund_ids"].to(device), batch["befund_mask"].to(device)
        )
        bef_pmask = batch["befund_mask"].bool().to(device)
    return z_beur_phrases, beur_pmask, z_bef_phrases, bef_pmask


def train_one_epoch(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    mask_decoder: MaskTokenDecoder,
    mask_head: MaskPredictionHead,
    internal_loader: DataLoader,
    btxrd_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    τ: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    log_lambda_sim: nn.Parameter,
    log_lambda_ortho: nn.Parameter,
    log_lambda_dice: nn.Parameter,
    trainable_params: list,
    device: torch.device,
    stage: int,                       # 1 = dice+ortho only; 2 = all losses
    learn_loss_weights: bool,
    active_losses: set[str],          # subset of {"ita", "sim", "ortho", "dice"}
    text_mode: str = "phrase",
    max_grad_norm: float = 1.0,
    τ_s_beur: float = 0.015,
    τ_s_bef: float = 0.07,
    τ_s_img_full: float = 0.07,
    sim_attn_tau: float = 0.07,
    same_image_boost: float = 0.0,
    reweight_by_n_phrases: bool = False,
    t2i_mode: str = "image_image",
    lambda_t2i: float = 1.0,
) -> dict[str, float]:
    vit.train()
    text_enc.train()
    mask_decoder.train()
    mask_head.train()

    # Stage 1: dice + ortho only; ita + sim activate in stage 2
    stage_losses = active_losses if stage == 2 else active_losses & {"dice", "ortho"}

    total_ita = total_sim = total_sim_i2t = total_sim_t2i = total_ortho = total_dice = total_dice_only = total_bce = total_hard_neg = total_total = 0.0
    n_batches = 0

    btxrd_cycle = itertools.cycle(btxrd_loader) if btxrd_loader is not None else None
    pbar = tqdm(internal_loader, desc=f"  stage{stage}", leave=False,
                disable=not sys.stdout.isatty())
    zero = torch.zeros(1, device=device)[0]

    for batch in pbar:
        img      = batch["full_image"].to(device)    # [B, 3, 224, 224]
        plabels  = batch["patch_labels"].to(device)  # [B, 196]  float32, soft coverage
        has_mask = batch["has_mask"].to(device)      # [B,]  bool
        has_bef  = batch["has_befund"].to(device)    # [B,]  bool

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):

            # ── Single ViT pass ───────────────────────────────────────────────
            cls_raw, patch_feat = vit.forward_all(img)   # [B,768], [B,196,768]
            z_img = vit.img_proj(cls_raw)                # [B, D]

            # ── Mask token decoder ────────────────────────────────────────────
            tokens = mask_decoder(patch_feat)         # [B, N, 768]
            mask_logits = mask_head(tokens, patch_feat)  # [B, N, 196]

            # ── Text encoding ─────────────────────────────────────────────────
            z_beur_phrases, beur_pmask, z_bef_phrases, bef_pmask = _encode_text_v2(
                batch, text_enc, device, text_mode
            )

            # ── L_ITA ────────────────────────────────────────────────────────
            l_ita = zero
            if "ita" in stage_losses:
                p2i_ita = _phrase_to_image(beur_pmask)
                _, l_ita_i2t, l_ita_t2i = symmetric_soft_semantic_loss(
                    z_img[p2i_ita], z_beur_phrases, beur_pmask, τ,
                    τ_s_phrase=τ_s_beur, τ_s_img=τ_s_img_full,
                    phrase_to_image=p2i_ita, same_image_boost=same_image_boost,
                    reweight_by_n_phrases=reweight_by_n_phrases,
                    t2i_mode=t2i_mode,
                )
                l_ita = l_ita_i2t + lambda_t2i * l_ita_t2i

            # ── FG token selection: GT overlap when mask available ────────────
            B = img.shape[0]
            with torch.no_grad():
                pred      = torch.sigmoid(mask_logits.float())
                overlap   = (pred * plabels.unsqueeze(1)).sum(-1)        # [B, N]
                fg_idx_gt = overlap.argmax(dim=1)
                fg_idx    = torch.where(has_mask, fg_idx_gt,
                                        select_fg_token(mask_logits.float()))

            # ── L_dice ───────────────────────────────────────────────────────
            l_dice = zero
            l_dice_only = zero
            l_bce = zero
            l_hard_neg = zero
            if "dice" in stage_losses and has_mask.any():
                l_dice, l_dice_only, l_bce, l_hard_neg = compute_l_dice_ce(
                    mask_logits[has_mask], plabels[has_mask],
                )

            # ── L_sim ────────────────────────────────────────────────────────
            l_sim = l_sim_i2t = l_sim_t2i = zero
            if "sim" in stage_losses:
                sim_valid = has_mask & has_bef
                if sim_valid.sum() >= 2:
                    P_proj = F.normalize(
                        vit.patch_proj(patch_feat[sim_valid]), dim=-1
                    )                                                       # [B_sim, P, D]
                    bef_sub       = z_bef_phrases[sim_valid]
                    bef_pmask_sub = bef_pmask[sim_valid]
                    p2i_sim       = _phrase_to_image(bef_pmask_sub)
                    l_sim, l_sim_i2t, l_sim_t2i = gloria_local_loss(
                        P_proj, bef_sub, bef_pmask_sub, τ,
                        τ2=sim_attn_tau, τ_s=τ_s_bef, τ_s_img=τ_s_img_full,
                        phrase_to_image=p2i_sim,
                        same_image_boost=same_image_boost,
                        reweight_by_n_phrases=reweight_by_n_phrases,
                        λ_t2i=lambda_t2i,
                        t2i_mode=t2i_mode,
                        img_cls_features=z_img[sim_valid],
                        patch_mask=plabels[sim_valid].bool(),
                    )

            # ── L_ortho ──────────────────────────────────────────────────────
            l_ortho = zero
            if "ortho" in stage_losses and has_mask.any():
                l_ortho = ortho_loss(patch_feat[has_mask], plabels[has_mask])

            # ── BTXRD: extra L_dice + L_ortho ────────────────────────────────
            if btxrd_cycle is not None:
                btxrd_b      = next(btxrd_cycle)
                btxrd_img    = btxrd_b["image"].to(device)
                btxrd_labels = btxrd_b["patch_labels"].to(device)
                _, btxrd_patches = vit.forward_all(btxrd_img)
                if "dice" in stage_losses:
                    btxrd_tokens = mask_decoder(btxrd_patches)
                    btxrd_logits = mask_head(btxrd_tokens, btxrd_patches)
                    btxrd_dice, btxrd_dice_only, btxrd_bce, btxrd_hard_neg = compute_l_dice_ce(
                        btxrd_logits, btxrd_labels,
                    )
                    l_dice      = l_dice      + btxrd_dice
                    l_dice_only = l_dice_only + btxrd_dice_only
                    l_bce       = l_bce       + btxrd_bce
                    l_hard_neg  = l_hard_neg  + btxrd_hard_neg
                if "ortho" in stage_losses:
                    l_ortho = l_ortho + ortho_loss(btxrd_patches, btxrd_labels)

            # ── Loss ─────────────────────────────────────────────────────────
            loss = (log_lambda_ita.exp()   * l_ita
                  + log_lambda_sim.exp()   * l_sim
                  + log_lambda_ortho.exp() * l_ortho
                  + log_lambda_dice.exp()  * l_dice)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_ita       += l_ita.item()
        total_sim       += l_sim.item()
        total_sim_i2t   += l_sim_i2t.item()
        total_sim_t2i   += l_sim_t2i.item()
        total_ortho     += l_ortho.item()
        total_dice      += l_dice.item()
        total_dice_only += l_dice_only.item()
        total_bce       += l_bce.item()
        total_hard_neg  += l_hard_neg.item()
        total_total     += loss.item()
        n_batches       += 1
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ita=f"{l_ita.item():.4f}",
            dice=f"{l_dice.item():.4f}",
        )

    d = max(1, n_batches)
    return {
        "train/loss":         total_total     / d,
        "train/l_ita":        total_ita       / d if "ita"   in active_losses else 50.0,
        "train/l_sim":        total_sim       / d if "sim"   in active_losses else 50.0,
        "train/l_sim_i2t":    total_sim_i2t   / d if "sim"   in active_losses else 50.0,
        "train/l_sim_t2i":    total_sim_t2i   / d if "sim"   in active_losses else 50.0,
        "train/l_ortho":      total_ortho     / d if "ortho" in active_losses else 50.0,
        "train/l_dice":       total_dice      / d if "dice"  in active_losses else 50.0,
        "train/l_dice_only":  total_dice_only / d if "dice"  in active_losses else 50.0,
        "train/l_bce":        total_bce       / d if "dice"  in active_losses else 50.0,
        "train/l_hard_neg":   total_hard_neg  / d if "dice"  in active_losses else 50.0,
        "train/lambda_ita":   log_lambda_ita.exp().item(),
        "train/lambda_sim":   log_lambda_sim.exp().item(),
        "train/lambda_ortho": log_lambda_ortho.exp().item(),
        "train/lambda_dice":  log_lambda_dice.exp().item(),
        "train/tau":          τ.item(),
    }


@torch.no_grad()
def evaluate(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    mask_decoder: MaskTokenDecoder,
    mask_head: MaskPredictionHead,
    val_loader: DataLoader,
    τ: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    log_lambda_sim: nn.Parameter,
    log_lambda_ortho: nn.Parameter,
    log_lambda_dice: nn.Parameter,
    device: torch.device,
    active_losses: set[str],
    stage: int = 2,               # mirrors train curriculum; gates sim+ortho
    text_mode: str = "phrase",
    τ_s_beur: float = 0.015,
    τ_s_bef: float = 0.07,
    τ_s_img_full: float = 0.07,
    sim_attn_tau: float = 0.07,
    same_image_boost: float = 0.0,
    reweight_by_n_phrases: bool = False,
    t2i_mode: str = "image_image",
    lambda_t2i: float = 1.0,
) -> dict[str, float]:
    vit.eval()
    text_enc.eval()
    mask_decoder.eval()
    mask_head.eval()

    # Mirror training: only compute losses active for this stage
    stage_losses = active_losses if stage == 2 else active_losses & {"dice", "ortho"}

    total_ita = total_sim = total_sim_i2t = total_sim_t2i = total_ortho = total_dice = total_dice_only = total_bce = total_hard_neg = total_total = 0.0
    n_batches = 0
    zero = torch.zeros(1, device=device)[0]

    for batch in val_loader:
        img      = batch["full_image"].to(device)
        plabels  = batch["patch_labels"].to(device)
        has_mask = batch["has_mask"].to(device)
        has_bef  = batch["has_befund"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            cls_raw, patch_feat = vit.forward_all(img)
            z_img = vit.img_proj(cls_raw)

            tokens = mask_decoder(patch_feat)
            mask_logits = mask_head(tokens, patch_feat)

            z_beur_phrases, beur_pmask, z_bef_phrases, bef_pmask = _encode_text_v2(
                batch, text_enc, device, text_mode
            )

            l_ita = zero
            if "ita" in stage_losses:
                p2i_ita = _phrase_to_image(beur_pmask)
                _, l_ita_i2t, l_ita_t2i = symmetric_soft_semantic_loss(
                    z_img[p2i_ita], z_beur_phrases, beur_pmask, τ,
                    τ_s_phrase=τ_s_beur, τ_s_img=τ_s_img_full,
                    phrase_to_image=p2i_ita, same_image_boost=same_image_boost,
                    reweight_by_n_phrases=reweight_by_n_phrases,
                    t2i_mode=t2i_mode,
                )
                l_ita = l_ita_i2t + lambda_t2i * l_ita_t2i

            B_val = img.shape[0]
            pred_v    = torch.sigmoid(mask_logits.float())
            overlap_v = (pred_v * plabels.unsqueeze(1)).sum(-1)        # [B, N]
            fg_idx_val = torch.where(
                has_mask, overlap_v.argmax(dim=1),
                select_fg_token(mask_logits.float()),
            )

            l_sim = l_sim_i2t = l_sim_t2i = zero
            if "sim" in stage_losses:
                sim_valid = has_mask & has_bef
                if sim_valid.sum() >= 2:
                    P_proj_v = F.normalize(
                        vit.patch_proj(patch_feat[sim_valid]), dim=-1
                    )                                                    # [B_sim, P, D]
                    bef_sub       = z_bef_phrases[sim_valid]
                    bef_pmask_sub = bef_pmask[sim_valid]
                    p2i_sim       = _phrase_to_image(bef_pmask_sub)
                    l_sim, l_sim_i2t, l_sim_t2i = gloria_local_loss(
                        P_proj_v, bef_sub, bef_pmask_sub, τ,
                        τ2=sim_attn_tau, τ_s=τ_s_bef, τ_s_img=τ_s_img_full,
                        phrase_to_image=p2i_sim,
                        same_image_boost=same_image_boost,
                        reweight_by_n_phrases=reweight_by_n_phrases,
                        λ_t2i=lambda_t2i,
                        t2i_mode=t2i_mode,
                        img_cls_features=z_img[sim_valid],
                        patch_mask=plabels[sim_valid].bool(),
                    )

            l_ortho = zero
            if "ortho" in stage_losses and has_mask.any():
                l_ortho = ortho_loss(patch_feat[has_mask], plabels[has_mask])

            l_dice = zero
            l_dice_only = zero
            l_bce = zero
            l_hard_neg = zero
            if "dice" in stage_losses and has_mask.any():
                l_dice, l_dice_only, l_bce, l_hard_neg = compute_l_dice_ce(
                    mask_logits[has_mask], plabels[has_mask],
                )

            loss = (log_lambda_ita.exp()   * l_ita
                  + log_lambda_sim.exp()   * l_sim
                  + log_lambda_ortho.exp() * l_ortho
                  + log_lambda_dice.exp()  * l_dice)

        total_ita       += l_ita.item()
        total_sim       += l_sim.item()
        total_sim_i2t   += l_sim_i2t.item()
        total_sim_t2i   += l_sim_t2i.item()
        total_ortho     += l_ortho.item()
        total_dice      += l_dice.item()
        total_dice_only += l_dice_only.item()
        total_bce       += l_bce.item()
        total_hard_neg  += l_hard_neg.item()
        total_total     += loss.item()
        n_batches       += 1

    d = max(1, n_batches)
    return {
        "val/loss":        total_total     / d,
        "val/l_ita":       total_ita       / d if "ita"   in active_losses else 50.0,
        "val/l_sim":       total_sim       / d if "sim"   in active_losses else 50.0,
        "val/l_sim_i2t":   total_sim_i2t   / d if "sim"   in active_losses else 50.0,
        "val/l_sim_t2i":   total_sim_t2i   / d if "sim"   in active_losses else 50.0,
        "val/l_ortho":     total_ortho     / d if "ortho" in active_losses else 50.0,
        "val/l_dice":      total_dice      / d if "dice"  in active_losses else 50.0,
        "val/l_dice_only": total_dice_only / d if "dice"  in active_losses else 50.0,
        "val/l_bce":       total_bce       / d if "dice"  in active_losses else 50.0,
        "val/l_hard_neg":  total_hard_neg  / d if "dice"  in active_losses else 50.0,
    }


@torch.no_grad()
def visualize_heatmaps(
    vit: SharedViT,
    mask_decoder: MaskTokenDecoder,
    mask_head: MaskPredictionHead,
    vis_samples: list[dict],
    epoch: int,
    run_dir: Path,
    device: torch.device,
    use_wandb: bool = False,
) -> None:
    """Save heatmap overlays for fixed validation images.

    Each vis_sample dict must have keys:
        img_path   : Path  — original image on disk (for display)
        img_tensor : Tensor [3, 224, 224]  — preprocessed for the model
        mask_arr   : np.ndarray [H, W] bool  — binary GT mask for contour

    Heatmap uses the fg mask token from MaskPredictionHead — consistent with
    the dice loss supervision and L_sim representation during pretraining.
    """
    vit.eval()
    mask_decoder.eval()
    mask_head.eval()

    epoch_dir = run_dir / "heatmaps" / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    wandb_images = []

    for i, s in enumerate(vis_samples):
        img_tensor = s["img_tensor"].unsqueeze(0).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            _, patch_feat = vit.forward_all(img_tensor)
            tokens      = mask_decoder(patch_feat)               # [1, N, 768]
            mask_logits = mask_head(tokens, patch_feat)          # [1, N, 196]

        fg_idx    = select_fg_token(mask_logits.float())         # [1,]
        fg_logits = mask_logits[0, fg_idx[0].item()]             # [196]
        h = torch.sigmoid(fg_logits.float()).cpu().numpy().reshape(14, 14)

        orig_img = s.get("img_crop", np.array(Image.open(s["img_path"]).convert("RGB")))
        H, W = orig_img.shape[:2]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(orig_img)
        ax.imshow(h, cmap="jet", alpha=0.4, vmin=0, vmax=1,
                  extent=[0, W, H, 0], interpolation="nearest")
        if s["mask_arr"] is not None:
            ax.contour(s["mask_arr"].astype(float), levels=[0.5],
                       colors=["lime"], linewidths=1.5)
        ax.axis("off")
        ax.set_title(f"Epoch {epoch:03d}  |  {s['img_path'].stem}", fontsize=8)

        out_path = epoch_dir / f"sample_{i:02d}.png"
        fig.savefig(out_path, dpi=100)
        plt.close(fig)

        if use_wandb:
            wandb_images.append(wandb.Image(str(out_path), caption=s["img_path"].stem))

    if use_wandb and wandb_images:
        wandb.log({"heatmaps": wandb_images}, step=epoch)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LACE v2 pretraining (MaskTokenDecoder, 2-stage curriculum)"
    )

    parser.add_argument("--splits", default=str(DEFAULT_SPLITS),
                        help="Path to split.json. Pass '' to generate from --dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_JSON))
    parser.add_argument("--btxrd_images", default=str(DEFAULT_BTXRD_IMAGES))
    parser.add_argument("--btxrd_annots", default=str(DEFAULT_BTXRD_ANNOTS))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))

    # ── ViT / LoRA ────────────────────────────────────────────────────────────
    parser.add_argument("--lora_layers", type=int,   default=4)
    parser.add_argument("--lora_r",      type=int,   default=8)
    parser.add_argument("--lora_alpha",  type=float, default=16.0)
    parser.add_argument("--embed_dim",   type=int,   default=512)
    parser.add_argument("--warm_start_projections", action="store_true")

    # ── Mask token decoder ────────────────────────────────────────────────────
    parser.add_argument("--n_mask_tokens", type=int,   default=4)
    parser.add_argument("--n_mask_heads",  type=int,   default=8)
    parser.add_argument("--gauss_sigma",   type=float, default=1.5,
                        help="Std-dev of Gaussian smoothing applied to cross-attention weights "
                             "in MaskTokenDecoder. 0 disables smoothing (ablation).")
    parser.add_argument("--mask_head_tau", type=float, default=1.0,
                        help="Fixed temperature for MaskPredictionHead logits: "
                             "logits = (t_norm @ patches) / (sqrt(d) * tau). "
                             "Smaller values sharpen the predicted mask.")
    parser.add_argument("--sim_attn_tau",  type=float, default=0.07,
                        help="Softmax temperature for heatmap attention weights in L_sim. "
                             "Applied to fg-token mask_logits (cosine sims) before weighted "
                             "average of projected patches.")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument("--btxrd_batch_size", type=int, default=16)
    parser.add_argument("--max_text_len",      type=int, default=128)
    parser.add_argument("--max_beur_text_len", type=int, default=256,
                        help="Tokenisation length for full beurteilung text (mixed mode).")
    parser.add_argument("--text_mode", default="phrase",
                        choices=["full", "phrase", "mixed"])
    parser.add_argument("--max_bef_phrases",   type=int,   default=16)
    parser.add_argument("--max_beur_phrases",  type=int,   default=16)
    parser.add_argument("--context_fraction",  type=float, default=0.15,
                        help="Context margin around lesion mask for image crop. "
                             "0.0 = full image (no crop).")
    parser.add_argument("--no_btxrd", action="store_true",
                        help="Disable BTXRD dataset entirely (useful for ablations).")
    parser.add_argument("--overfit_n", type=int, default=None,
                        help="Overfit sanity-check: restrict train+val to the first N "
                             "samples, disable BTXRD, and set drop_last=False.")

    # ── Curriculum ────────────────────────────────────────────────────────────
    parser.add_argument("--stage1_epochs", type=int, default=10,
                        help="Epochs for stage 1 (L_dice + L_ortho only). "
                             "Set 0 to disable curriculum.")
    parser.add_argument("--epochs",        type=int, default=40,
                        help="Total training epochs (stage1 + stage2).")

    # ── Optimisation ──────────────────────────────────────────────────────────
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--scheduler",    default="constant", choices=["constant", "cosine"])
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--patience",     type=int,   default=20)

    # ── Loss weights ──────────────────────────────────────────────────────────
    parser.add_argument("--lambda_sim",   type=float, default=1.0)
    parser.add_argument("--lambda_ortho", type=float, default=0.1)
    parser.add_argument("--lambda_dice",  type=float, default=1.0)
    parser.add_argument("--learn_loss_weights", action="store_true", default=False,
                        help="Make all log-lambda weights learnable parameters.")
    parser.add_argument("--losses", nargs="+",
                        default=["ita", "sim", "ortho", "dice"],
                        choices=["ita", "sim", "ortho", "dice"],
                        help="Which loss terms to include. Curriculum still gates "
                             "sim+ortho to stage 2.")

    # ── Soft-target / t2i settings ────────────────────────────────────────────
    parser.add_argument("--t2i_mode", default="image_image",
                        choices=["image_image", "text_text", "descriptor", "infonce"])
    parser.add_argument("--tau_s_beur",     type=float, default=0.015)
    parser.add_argument("--tau_s_bef",      type=float, default=0.07)
    parser.add_argument("--tau_s_img_full", type=float, default=0.07)
    parser.add_argument("--lambda_t2i",     type=float, default=1.0)
    parser.add_argument("--same_image_boost",      type=float, default=0.0)
    parser.add_argument("--reweight_by_n_phrases", action="store_true", default=False)

    # ── Split fractions ───────────────────────────────────────────────────────
    parser.add_argument("--downstream_train_frac", type=float, default=0.8)
    parser.add_argument("--downstream_val_frac",   type=float, default=0.1)
    parser.add_argument("--test_frac",             type=float, default=0.1)
    parser.add_argument("--seed",                  type=int,   default=42)

    # ── W&B ───────────────────────────────────────────────────────────────────
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

    mask_decoder = MaskTokenDecoder(
        n_tokens=args.n_mask_tokens,
        n_heads=args.n_mask_heads,
        sigma=args.gauss_sigma,
    ).to(device)
    mask_head = MaskPredictionHead(tau=args.mask_head_tau).to(device)

    τ = nn.Parameter(torch.tensor(0.07, device=device))

    # ── Log-lambda weights ────────────────────────────────────────────────────
    log_lambda_ita   = nn.Parameter(torch.zeros([], device=device))
    log_lambda_sim   = nn.Parameter(torch.zeros([], device=device))
    log_lambda_ortho = nn.Parameter(torch.zeros([], device=device))
    log_lambda_dice  = nn.Parameter(torch.zeros([], device=device))

    if args.learn_loss_weights:
        lambda_params = [log_lambda_ita, log_lambda_sim, log_lambda_ortho, log_lambda_dice]
    else:
        log_lambda_ita.data.fill_(0.0)
        log_lambda_sim.data.fill_(math.log(args.lambda_sim))
        log_lambda_ortho.data.fill_(math.log(args.lambda_ortho))
        log_lambda_dice.data.fill_(math.log(args.lambda_dice))
        lambda_params = []
        for p in [log_lambda_ita, log_lambda_sim, log_lambda_ortho, log_lambda_dice]:
            p.requires_grad_(False)

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
        val_samples      = split_data["val"]
        print(f"Loaded split from {args.splits} "
              f"({len(pretrain_samples)} train, {len(val_samples)} val samples)")
    else:
        pretrain_samples, val_samples, _test = build_stratified_splits(args, run_dir=run_dir)

    preprocess_val   = vit.preprocess_val
    preprocess_train = build_train_transform_lace(preprocess_val)
    tokenizer        = text_enc.tokenizer

    ds_kwargs = dict(
        max_text_len=args.max_text_len,
        max_beur_text_len=args.max_beur_text_len,
        text_mode=args.text_mode,
        max_bef_phrases=args.max_bef_phrases,
        max_beur_phrases=args.max_beur_phrases,
        context_fraction=args.context_fraction,
    )
    train_ds = InternalDatasetV2(pretrain_samples, preprocess_train, tokenizer, **ds_kwargs)
    val_ds   = InternalDatasetV2(val_samples,      preprocess_val,   tokenizer, **ds_kwargs)

    if args.overfit_n is not None:
        from torch.utils.data import Subset
        n = args.overfit_n
        indices = list(range(min(n, len(train_ds))))
        train_ds = Subset(train_ds, indices)
        # val uses the same samples as train so we can observe overfitting directly
        val_ds   = Subset(InternalDatasetV2(pretrain_samples, preprocess_val, tokenizer, **ds_kwargs), indices)
        args.no_btxrd = True
        print(f"[overfit mode] using {len(train_ds)} train samples as train+val, BTXRD disabled")

    print(f"Pretrain datasets: {len(train_ds)} train / {len(val_ds)} val")

    # ── Fixed visualisation samples (picked once, reused every epoch) ─────────
    vis_samples: list[dict] = []
    raw_val_ds = val_ds.dataset if hasattr(val_ds, "dataset") else val_ds
    for s in raw_val_ds.samples:
        if len(vis_samples) >= 10:
            break
        mask_path = Path(s["mask"])
        if not mask_path.exists():
            continue
        mask_arr = np.array(Image.open(mask_path).convert("L"), dtype=float)
        if not np.any(mask_arr > 0):
            continue
        img_path = Path(s["image"])
        img_arr = np.array(Image.open(img_path).convert("RGB"))
        if args.context_fraction >= 0:
            img_crop_arr, crop_mask_arr = crop_around_mask_pair(
                img_arr, mask_arr, context_fraction=args.context_fraction,
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
    print(f"Visualisation samples: {len(vis_samples)} fixed val images with GT masks")

    use_pin = device.type == "cuda"
    internal_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=use_pin,
        drop_last=(args.overfit_n is None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=use_pin,
    )
    btxrd_loader = None
    if not args.no_btxrd:
        btxrd_ds = BTXRDOrthoDataset(
            Path(args.btxrd_images), Path(args.btxrd_annots), preprocess_val,
        )
        if len(btxrd_ds) > 0:
            btxrd_loader = DataLoader(
                btxrd_ds, batch_size=args.btxrd_batch_size, shuffle=True,
                num_workers=4, pin_memory=use_pin, drop_last=True,
            )

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
    dec_decay,  dec_nd  = _split_params(mask_decoder)
    hd_decay,   hd_nd   = _split_params(mask_head)

    optimizer = torch.optim.AdamW(
        [
            {"params": vit_decay + txt_decay + dec_decay + hd_decay,
             "weight_decay": args.weight_decay},
            {"params": vit_nd + txt_nd + dec_nd + hd_nd + [τ] + lambda_params,
             "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.98), eps=1e-6,
    )
    trainable_params = (
        vit_decay + txt_decay + dec_decay + hd_decay
        + vit_nd + txt_nd + dec_nd + hd_nd
        + [τ] + lambda_params
    )

    warmup_stage1 = max(1, args.stage1_epochs // 5) if args.stage1_epochs > 0 else 0
    warmup_stage2 = max(1, (args.epochs - args.stage1_epochs) // 5)
    scheduler     = make_scheduler(
        optimizer, warmup_stage1, args.stage1_epochs or args.epochs, args.scheduler
    )
    scaler        = (
        torch.amp.GradScaler("cuda") if device.type == "cuda"
        else torch.amp.GradScaler("cpu")
    )

    lora_cfg = {
        "lora_layers": args.lora_layers,
        "lora_r":      args.lora_r,
        "lora_alpha":  args.lora_alpha,
        "embed_dim":   args.embed_dim,
    }

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
    best_val_loss     = float("inf")
    best_mean_r1      = float("-inf")
    epochs_no_improve = 0
    val_loss          = float("inf")
    stage1_end        = args.stage1_epochs   # 0 means skip directly to stage 2

    active_losses = set(args.losses)
    print(f"Active losses: {sorted(active_losses)}")

    shared_kwargs = dict(
        active_losses=active_losses,
        text_mode=args.text_mode,
        τ_s_beur=args.tau_s_beur,
        τ_s_bef=args.tau_s_bef,
        τ_s_img_full=args.tau_s_img_full,
        sim_attn_tau=args.sim_attn_tau,
        same_image_boost=args.same_image_boost,
        reweight_by_n_phrases=args.reweight_by_n_phrases,
        t2i_mode=args.t2i_mode,
        lambda_t2i=args.lambda_t2i,
    )

    stage2_epochs = args.epochs - stage1_end
    print(
        f"\nStarting LACE v2 training: {args.epochs} epochs "
        f"(stage1={stage1_end}, stage2={stage2_epochs})\n"
    )

    checkpoint = {}
    stage2_started = False
    for epoch in range(1, args.epochs + 1):
        stage = 1 if epoch <= stage1_end else 2

        if stage == 2 and not stage2_started:
            stage2_started = True
            scheduler = make_scheduler(
                optimizer, warmup_stage2,
                args.epochs - stage1_end, args.scheduler,
            )
            # Reset early stopping: val/loss scale changes when sim+ortho activate
            best_val_loss     = float("inf")
            epochs_no_improve = 0

        train_metrics = train_one_epoch(
            vit, text_enc, mask_decoder, mask_head,
            internal_loader, btxrd_loader,
            optimizer, scaler, τ,
            log_lambda_ita, log_lambda_sim, log_lambda_ortho, log_lambda_dice,
            trainable_params, device,
            stage=stage,
            learn_loss_weights=args.learn_loss_weights,
            **shared_kwargs,
        )
        val_metrics = evaluate(
            vit, text_enc, mask_decoder, mask_head, val_loader, τ,
            log_lambda_ita, log_lambda_sim, log_lambda_ortho, log_lambda_dice,
            device, stage=stage, **shared_kwargs,
        )
        retrieval_metrics = evaluate_retrieval_lace(
            vit, text_enc, val_loader, device, text_mode=args.text_mode,
        )
        if vis_samples:
            visualize_heatmaps(
                vit, mask_decoder, mask_head, vis_samples,
                epoch, run_dir, device, use_wandb=use_wandb,
            )
        scheduler.step()

        val_loss   = val_metrics["val/loss"]
        lr_current = scheduler.get_last_lr()[0]
        i2t_r1  = retrieval_metrics.get("retrieval/i2t_r1",          float("nan"))
        i2t_r5  = retrieval_metrics.get("retrieval/i2t_r5",          float("nan"))
        i2t_med = retrieval_metrics.get("retrieval/i2t_median_rank",  float("nan"))
        t2i_r1  = retrieval_metrics.get("retrieval/t2i_r1",          float("nan"))
        t2i_r5  = retrieval_metrics.get("retrieval/t2i_r5",          float("nan"))
        t2i_med = retrieval_metrics.get("retrieval/t2i_median_rank",  float("nan"))
        mean_r1 = retrieval_metrics.get("retrieval/mean_r1",          float("nan"))
        print(
            f"Epoch {epoch:03d}/{args.epochs} [stage {stage}] | "
            f"train={train_metrics['train/loss']:.4f} | val={val_loss:.4f} | "
            f"ita={val_metrics['val/l_ita']:.4f} | "
            f"sim={val_metrics['val/l_sim']:.4f} | "
            f"ortho={val_metrics['val/l_ortho']:.4f} | "
            f"dice={val_metrics['val/l_dice']:.4f} "
            f"(d={val_metrics['val/l_dice_only']:.4f} bce={val_metrics['val/l_bce']:.4f}) | "
            f"lr={lr_current:.2e}"
        )
        print(
            f"  I2T R@1={i2t_r1:.3f} R@5={i2t_r5:.3f} med={i2t_med:.0f} | "
            f"T2I R@1={t2i_r1:.3f} R@5={t2i_r5:.3f} med={t2i_med:.0f} | "
            f"mean_r1={mean_r1:.3f}"
        )

        if use_wandb:
            wandb.log(
                {**train_metrics, **val_metrics, **retrieval_metrics,
                 "epoch": epoch, "stage": stage, "train/lr": lr_current},
                step=epoch,
            )

        checkpoint = {
            "epoch":                epoch,
            "stage":                stage,
            "val_loss":             val_loss,
            "lora_config":          lora_cfg,
            "n_mask_tokens":        args.n_mask_tokens,
            "n_mask_heads":         args.n_mask_heads,
            "gauss_sigma":          args.gauss_sigma,
            "vit_state":            vit.state_dict(),
            "text_enc_state":       text_enc.state_dict(),
            "mask_decoder_state":   mask_decoder.state_dict(),
            "mask_head_state":      mask_head.state_dict(),
            "tau":                  τ.data,
            "log_lambda_ita":       log_lambda_ita.data,
            "log_lambda_sim":       log_lambda_sim.data,
            "log_lambda_ortho":     log_lambda_ortho.data,
            "log_lambda_dice":      log_lambda_dice.data,
            "optimizer_state":      optimizer.state_dict(),
        }

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(checkpoint, run_dir / "best_checkpoint.pt")
        else:
            epochs_no_improve += 1

        if mean_r1 > best_mean_r1:
            best_mean_r1 = mean_r1
            torch.save(checkpoint, run_dir / "best_retrieval_checkpoint.pt")

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs).")
            break

    # Save final checkpoint without optimizer state
    checkpoint.pop("optimizer_state", None)
    torch.save(checkpoint, run_dir / "final_checkpoint.pt")
    print(f"\nDone. Best val loss: {best_val_loss:.4f} | Best mean R@1: {best_mean_r1:.3f}. Checkpoints: {run_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
