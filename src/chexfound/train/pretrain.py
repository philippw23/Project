"""CheXFound iBOT continued-pretraining script using SSLMetaArch (single-GPU).

Implements the joint DINO + iBOT (masked-image-modelling) objective via
SSLMetaArch, which matches the upstream CheXFound training logic exactly:
block-based rectangular masking (MaskingGenerator), correct DINO loss
normalisation, and KoLeo regularisation per the original CheXFound paper.

Usage:
    python src/chexfound/train/pretrain.py \\
        --config   src/chexfound/configs/chexfound_vitl16_bonetumor.yaml \\
        --base_cfg src/chexfound/data/config.yaml \\
        --out_dir  results/chexfound_pretrain

Checkpoints are saved as:
    results/chexfound_pretrain/
        checkpoint_ep{N}.pth   — periodic snapshots
        checkpoint_last.pth    — always overwritten with the latest epoch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add …/Project/src to sys.path so `chexfound` and `biomedclip` are importable
# regardless of how the script is invoked (torchrun, wandb agent, pytest, etc.).
_SRC = str(Path(__file__).resolve().parents[2])  # …/Project/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from chexfound.bone_tumor_patch.bone_tumor import BoneTumorDataset
from chexfound.data import DataAugmentationDINO, MaskingGenerator, collate_data_and_cast
from chexfound.models.lora import inject_lora_chexfound, _iter_real_vit_blocks
from chexfound.train.ssl_meta_arch import SSLMetaArch
from chexfound.utils.utils import CosineScheduler, fix_random_seeds
from biomedclip.data.splits import build_stratified_splits
from biomedclip.utils.misc import DEFAULT_DATASET_JSON, DEFAULT_SPLITS


# ══════════════════════════════════════════════════════════════════════════════
# Config helpers
# ══════════════════════════════════════════════════════════════════════════════

def merge_configs(base_path: str | None, override_path: str) -> dict:
    """Deep-merge override YAML on top of base YAML (override wins)."""
    cfg: dict = {}
    if base_path and Path(base_path).is_file():
        with open(base_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    with open(override_path, encoding="utf-8") as fh:
        override = yaml.safe_load(fh) or {}
    for section, values in override.items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg[section] = {**cfg[section], **values}
        else:
            cfg[section] = values
    return cfg


def _unfreeze_last_backbone_blocks(backbone: nn.Module, unfreeze_blocks: int) -> None:
    """Freeze all backbone params, then unfreeze the last N real transformer blocks + norm.

    Registers a detach hook at the frozen/unfrozen boundary so the frozen blocks
    run without building a computation graph, freeing their activations immediately.
    """
    for p in backbone.parameters():
        p.requires_grad_(False)

    blocks = _iter_real_vit_blocks(backbone)
    n_blocks = len(blocks)
    effective = min(unfreeze_blocks, n_blocks)
    first_unfrozen = blocks[n_blocks - effective]

    for block in blocks[n_blocks - effective:]:
        for p in block.parameters():
            p.requires_grad_(True)

    if hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad_(True)

    # Detach at the boundary: frozen blocks' activations are freed immediately
    # instead of being kept alive for a backward pass that never reaches them.
    def _detach(module, args):
        return tuple(x.detach() if isinstance(x, torch.Tensor) else x for x in args)
    first_unfrozen.register_forward_pre_hook(_detach)


def _apply_sweep_cfg(cfg: dict) -> None:
    """Override Tier-1 pretrain hyperparams from wandb.config (called after wandb.init)."""
    wc = wandb.config
    if "base_lr" in wc:
        cfg.setdefault("optim", {})["base_lr"] = wc["base_lr"]
    if "lora_r" in wc:
        cfg.setdefault("lora", {})["r"] = int(wc["lora_r"])
    if "lora_layers" in wc:
        cfg.setdefault("lora", {})["lora_layers"] = int(wc["lora_layers"])
    if "momentum_teacher" in wc:
        cfg.setdefault("teacher", {})["momentum_teacher"] = wc["momentum_teacher"]
    if "head_mode" in wc:
        cfg.setdefault("lora", {})["head_mode"] = wc["head_mode"]
    if "head_unfreeze_epoch" in wc:
        cfg.setdefault("lora", {})["head_unfreeze_epoch"] = int(wc["head_unfreeze_epoch"])
    if "unfreeze_blocks" in wc:
        cfg.setdefault("backbone", {})["unfreeze_blocks"] = int(wc["unfreeze_blocks"])


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def _parse_dataset_path(dataset_path: str) -> tuple[str, dict]:
    parts = dataset_path.split(":")
    name = parts[0]
    kwargs: dict = {}
    for part in parts[1:]:
        k, _, v = part.partition("=")
        kwargs[k] = v
    return name, kwargs


def build_dataset(dataset_path: str, transform, out_dir: Path) -> torch.utils.data.Dataset:
    name, kwargs = _parse_dataset_path(dataset_path)
    if name == "BoneTumor":
        splits_path = kwargs.get("splits", str(DEFAULT_SPLITS))
        if Path(splits_path).is_file():
            with open(splits_path, encoding="utf-8") as fh:
                split_data = json.load(fh)
            train = split_data["train"]
            print(f"BoneTumor: loaded {len(train)} train samples from {splits_path}")
        else:
            import types
            ns = types.SimpleNamespace(
                dataset=kwargs.get("dataset", str(DEFAULT_DATASET_JSON)),
                downstream_train_frac=float(kwargs.get("downstream_train_frac", 0.8)),
                downstream_val_frac=float(kwargs.get("downstream_val_frac", 0.1)),
                test_frac=float(kwargs.get("test_frac", 0.1)),
                seed=int(kwargs.get("seed", 42)),
                out_dir=str(out_dir),
            )
            train, _val, _test = build_stratified_splits(ns)
        image_paths = [s["image"] for s in train]
        images_dir = str(Path(image_paths[0]).parent) if image_paths else "."
        return BoneTumorDataset(image_paths=image_paths, root=images_dir, transforms=transform)
    raise ValueError(f"Unknown dataset: {name!r}. Supported: 'BoneTumor'.")


# ══════════════════════════════════════════════════════════════════════════════
# EMA update (single-GPU — bypasses FSDP module list in SSLMetaArch)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _ema_update(model: SSLMetaArch, momentum: float) -> None:
    for k in model.student.keys():
        for ps, pt in zip(model.student[k].parameters(), model.teacher[k].parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg_dict: dict, out_dir: Path, use_wandb: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {out_dir}")
    print(f"Device     : {device}")

    # ── Extract scalar hyperparams from the merged dict ──────────────────────
    train_cfg   = cfg_dict.get("train",   {})
    optim_cfg   = cfg_dict.get("optim",   {})
    teacher_cfg = cfg_dict.get("teacher", {})
    crops_cfg   = cfg_dict.get("crops",   {})
    student_cfg = cfg_dict.get("student", {})
    ibot_cfg    = cfg_dict.get("ibot",    {})

    n_epochs      = optim_cfg.get("epochs", 50)
    batch_per_gpu = train_cfg.get("batch_size_per_gpu", 4)
    _cpus         = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    n_workers     = min(train_cfg.get("num_workers", 4), _cpus)
    epoch_len     = train_cfg.get("OFFICIAL_EPOCH_LENGTH", 2500)
    save_freq     = train_cfg.get("saveckp_freq", 10)
    seed          = train_cfg.get("seed", 0)
    es_patience   = train_cfg.get("early_stopping_patience", None)   # None = disabled
    es_min_delta  = train_cfg.get("early_stopping_min_delta", 0.0)

    base_lr    = optim_cfg.get("base_lr", 5e-5)
    min_lr     = base_lr * 0.1
    wd_start   = optim_cfg.get("weight_decay", 0.04)
    wd_end     = optim_cfg.get("weight_decay_end", 0.2)
    clip_grad  = optim_cfg.get("clip_grad", 3.0)
    warmup_epochs = optim_cfg.get("warmup_epochs", 5)
    freeze_last_layer_epochs = optim_cfg.get("freeze_last_layer_epochs", 1)
    beta1 = optim_cfg.get("adamw_beta1", 0.9)
    beta2 = optim_cfg.get("adamw_beta2", 0.999)

    momentum_start  = teacher_cfg.get("momentum_teacher", 0.994)
    momentum_end    = teacher_cfg.get("final_momentum_teacher", 1.0)
    warmup_t_temp   = teacher_cfg.get("warmup_teacher_temp", 0.04)
    teacher_temp    = teacher_cfg.get("teacher_temp", 0.07)
    warmup_t_epochs = teacher_cfg.get("warmup_teacher_temp_epochs", 30)

    patch_size   = student_cfg.get("patch_size", 16)
    global_size  = crops_cfg.get("global_crops_size", 512)
    local_size   = crops_cfg.get("local_crops_size", 144)
    global_scale = tuple(crops_cfg.get("global_crops_scale", [0.32, 1.0]))
    local_scale  = tuple(crops_cfg.get("local_crops_scale", [0.05, 0.32]))
    n_local      = crops_cfg.get("local_crops_number", 8)

    mask_prob = ibot_cfg.get("mask_sample_probability", 0.5)
    mask_min  = ibot_cfg.get("mask_ratio_min_max", [0.1, 0.5])[0]
    mask_max  = ibot_cfg.get("mask_ratio_min_max", [0.1, 0.5])[1]

    n_patches = (global_size // patch_size) ** 2

    fix_random_seeds(seed)

    # ── Prepare OmegaConf config for SSLMetaArch ─────────────────────────────
    # Disable ShardedGradScaler — backprop_loss will call loss.backward() directly,
    # and we manage the optimizer step ourselves in the outer loop.
    cfg_dict.setdefault("compute_precision", {})["grad_scaler"] = False
    # Warm-start is handled below; suppress SSLMetaArch's internal loading.
    cfg_dict.setdefault("student", {})["pretrained_weights"] = ""
    # Ensure layerwise LR decay defaults are present for get_params_groups().
    cfg_dict["optim"].setdefault("layerwise_decay", 1.0)
    cfg_dict["optim"].setdefault("patch_embed_lr_mult", 0.2)

    cfg_obj = OmegaConf.create(cfg_dict)

    # ── Build model ──────────────────────────────────────────────────────────
    model = SSLMetaArch(cfg_obj)
    # Skip the FSDP stream-sync path — not applicable for single-GPU.
    model.need_to_synchronize_fsdp_streams = False
    # Replace update_teacher() with a direct parameter EMA (no FSDP module list).
    model.update_teacher = lambda m: _ema_update(model, m)

    lora_cfg            = cfg_dict.get("lora", {})
    lora_layers         = lora_cfg.get("lora_layers", 0)
    lora_r              = lora_cfg.get("r", 8)
    lora_alpha          = lora_cfg.get("alpha", 16.0)
    head_mode           = lora_cfg.get("head_mode", "frozen")
    head_unfreeze_epoch = lora_cfg.get("head_unfreeze_epoch", 10)

    backbone_cfg    = cfg_dict.get("backbone", {})
    unfreeze_blocks = backbone_cfg.get("unfreeze_blocks", 0)

    # Warm-start BEFORE LoRA injection so checkpoint key names match.
    # LoRA injection renames e.g. attn.qkv.weight → attn.qkv.linear.weight;
    # loading after injection would leave those 48 backbone weights at random init.
    # Single torch.load covers backbone + heads to avoid redundant NFS reads.
    warm_start = cfg_dict.get("MODEL", {}).get("WEIGHTS", "")
    if warm_start and Path(warm_start).is_file():
        print(f"Warm-starting from {warm_start}")
        _raw        = torch.load(warm_start, map_location="cpu", weights_only=False)
        _teacher_sd = _raw.get("teacher", _raw)

        # Backbone: strip "backbone." prefix so keys match the trunk module.
        _backbone_sd = {
            (k[len("backbone."):] if k.startswith("backbone.") else k): v
            for k, v in _teacher_sd.items()
            if not k.startswith(("dino_head.", "ibot_head."))
        }
        _msg = model.student.backbone.load_state_dict(_backbone_sd, strict=False)
        model.teacher.backbone.load_state_dict(_backbone_sd, strict=False)
        print(f"  backbone : {len(_backbone_sd)} keys "
              f"(missing={len(_msg.missing_keys)}, "
              f"unexpected={len(_msg.unexpected_keys)})")

        # Heads: load into student then copy to teacher so both start identically.
        # Previously both heads were randomly initialized from separate factory calls,
        # meaning teacher targets were based on a different random state than the student.
        _dino_sd = {k[len("dino_head."):]: v
                    for k, v in _teacher_sd.items() if k.startswith("dino_head.")}
        _ibot_sd = {k[len("ibot_head."):]: v
                    for k, v in _teacher_sd.items() if k.startswith("ibot_head.")}
        if _dino_sd:
            model.student.dino_head.load_state_dict(_dino_sd, strict=True)
            model.teacher.dino_head.load_state_dict(_dino_sd, strict=True)
            print(f"  dino_head: {len(_dino_sd)} keys (student + teacher synced)")
        if _ibot_sd and "ibot_head" in model.student:
            model.student.ibot_head.load_state_dict(_ibot_sd, strict=True)
            model.teacher.ibot_head.load_state_dict(_ibot_sd, strict=True)
            print(f"  ibot_head: {len(_ibot_sd)} keys (student + teacher synced)")

    # Inject LoRA after warm-start so pretrained weights land in the right slots.
    if lora_layers > 0 and unfreeze_blocks > 0:
        raise ValueError("lora_layers and unfreeze_blocks are mutually exclusive — set one to 0.")

    if lora_layers > 0:
        inject_lora_chexfound(
            model.student.backbone, lora_layers=lora_layers, r=lora_r, alpha=lora_alpha)
        inject_lora_chexfound(
            model.teacher.backbone, lora_layers=lora_layers, r=lora_r, alpha=lora_alpha)
        # Teacher backbone already has the same pretrained weights as student;
        # sync so both have identical (random) LoRA initialisations too.
        model.teacher.backbone.load_state_dict(model.student.backbone.state_dict())
    elif unfreeze_blocks > 0:
        _unfreeze_last_backbone_blocks(model.student.backbone, unfreeze_blocks)
        # Teacher backbone stays fully frozen (EMA updated from student).
        for p in model.teacher.backbone.parameters():
            p.requires_grad_(False)

    # Re-freeze all teacher parameters (LoRA injection may have unfrozen some).
    for p in model.teacher.parameters():
        p.requires_grad_(False)

    # Collect student head modules once — reused for freeze/unfreeze throughout.
    _head_modules = [model.student.dino_head]
    if "ibot_head" in model.student:
        _head_modules.append(model.student.ibot_head)

    # frozen     : heads excluded from updates entirely; stable pretrained targets.
    # unfreeze_after: heads frozen initially, unfrozen at head_unfreeze_epoch.
    # train      : heads fully trainable from checkpoint init (default before this change).
    if head_mode in ("frozen", "unfreeze_after"):
        for _hm in _head_modules:
            _hm.requires_grad_(False)

    model = model.to(device)
    model.train()  # keeps teacher in eval() via SSLMetaArch.train()

    trainable = sum(p.numel() for p in model.student.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Student trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    if lora_layers > 0:
        print(f"LoRA: last {lora_layers} blocks | r={lora_r} | alpha={lora_alpha}")
    elif unfreeze_blocks > 0:
        print(f"Unfreeze: last {unfreeze_blocks} backbone blocks + norm (no LoRA)")
    print(f"Head mode: {head_mode}" +
          (f" (unfreeze at epoch {head_unfreeze_epoch})" if head_mode == "unfreeze_after" else ""))

    # ── Augmentation + DataLoader ─────────────────────────────────────────────
    aug = DataAugmentationDINO(
        global_crops_scale=global_scale,
        local_crops_scale=local_scale,
        local_crops_number=n_local,
        global_crops_size=global_size,
        local_crops_size=local_size,
    )

    dataset = build_dataset(train_cfg["dataset_path"], transform=aug, out_dir=out_dir)

    mask_generator = MaskingGenerator(
        input_size=(global_size // patch_size, global_size // patch_size),
        num_masking_patches=int(n_patches * mask_max),
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=(mask_min, mask_max),
        mask_probability=mask_prob,
        dtype=torch.float32,
        n_tokens=n_patches,
        mask_generator=mask_generator,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_per_gpu,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_fn,
    )

    steps_per_epoch = min(epoch_len, len(loader))
    total_steps     = n_epochs * steps_per_epoch

    print(f"\nStarting iBOT training: {n_epochs} epochs × {steps_per_epoch} steps")
    print(f"  Dataset : {len(dataset)} images")
    print(f"  Batch   : {batch_per_gpu}")

    # ── Optimizer & schedules ─────────────────────────────────────────────────
    # unfreeze_after: heads must enter param groups now so they can be updated
    # once unfrozen, even though they start frozen. Re-freeze immediately after.
    if head_mode == "unfreeze_after":
        for _hm in _head_modules:
            _hm.requires_grad_(True)
    param_groups = list(model.get_params_groups())
    if head_mode == "unfreeze_after":
        for _hm in _head_modules:
            _hm.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        param_groups, lr=base_lr, weight_decay=wd_start, betas=(beta1, beta2))

    lr_schedule  = CosineScheduler(base_lr, min_lr,    total_steps,
                                   warmup_iters=warmup_epochs * steps_per_epoch,
                                   start_warmup_value=min_lr)
    wd_schedule  = CosineScheduler(wd_start,  wd_end,  total_steps)
    mom_schedule = CosineScheduler(momentum_start, momentum_end, total_steps)

    # Linear warmup of teacher temperature over warmup_t_epochs, then constant.
    teacher_temp_schedule = [
        warmup_t_temp + (teacher_temp - warmup_t_temp) * i / max(warmup_t_epochs - 1, 1)
        if i < warmup_t_epochs else teacher_temp
        for i in range(n_epochs)
    ]

    # Clip all params that are in the optimizer (covers newly-unfrozen heads too).
    all_trainable = [p for g in optimizer.param_groups for p in g["params"]]

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step   = 0
    log: list[dict] = []
    best_loss     = float("inf")
    no_improve    = 0
    stop_training = False

    for epoch in range(n_epochs):
        # Freeze/unfreeze prototype last layer during initial epochs.
        freeze_last = epoch < freeze_last_layer_epochs
        for g in optimizer.param_groups:
            if g.get("is_last_layer", False):
                for p in g["params"]:
                    p.requires_grad_(not freeze_last)

        # Head mode: unfreeze heads at the configured epoch.
        # Runs after is_last_layer so it correctly overrides for head last-layers.
        if head_mode == "unfreeze_after":
            head_trainable = epoch >= head_unfreeze_epoch
            for _hm in _head_modules:
                _hm.requires_grad_(head_trainable)
            if epoch == head_unfreeze_epoch:
                n_head = sum(p.numel() for _hm in _head_modules
                             for p in _hm.parameters())
                print(f"Epoch {epoch + 1:03d}: heads unfrozen ({n_head:,} params)")

        t_temp = teacher_temp_schedule[epoch]
        epoch_losses: dict[str, float] = {
            k: 0.0 for k in ["dino_local", "dino_global", "ibot", "koleo", "total"]
        }
        t0 = time.time()
        n_steps_done = 0

        data_iter = iter(loader)
        for _ in range(steps_per_epoch):
            # Per-step schedule values.
            lr_val   = lr_schedule[global_step]
            wd_val   = wd_schedule[global_step]
            momentum = mom_schedule[global_step]

            for g in optimizer.param_groups:
                g["lr"] = lr_val * g.get("lr_multiplier", 1.0)
                if not g.get("is_last_layer", False):
                    g["weight_decay"] = wd_val * g.get("wd_multiplier", 1.0)

            try:
                images = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                images = next(data_iter)

            optimizer.zero_grad(set_to_none=True)

            # forward_backward() runs the full forward pass and calls
            # loss.backward() internally (fp16_scaler is None).
            loss_dict = model.forward_backward(images, t_temp)

            if clip_grad > 0:
                nn.utils.clip_grad_norm_(all_trainable, clip_grad)
            optimizer.step()
            model.update_teacher(momentum)

            # Accumulate for epoch logging.
            epoch_losses["dino_local"]  += loss_dict.get("dino_local_crops_loss",  torch.tensor(0.0)).item()
            epoch_losses["dino_global"] += loss_dict.get("dino_global_crops_loss", torch.tensor(0.0)).item()
            epoch_losses["ibot"]        += loss_dict.get("ibot_loss",              torch.tensor(0.0)).item()
            epoch_losses["koleo"]       += loss_dict.get("koleo_loss",             torch.tensor(0.0)).item()
            epoch_losses["total"]       += sum(v.item() for v in loss_dict.values())

            global_step    += 1
            n_steps_done   += 1

        # ── End-of-epoch logging & checkpointing ─────────────────────────────
        n = max(n_steps_done, 1)
        avg = {k: v / n for k, v in epoch_losses.items()}
        lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1:03d}/{n_epochs} | "
            f"total={avg['total']:.4f} "
            f"dino_l={avg['dino_local']:.4f} dino_g={avg['dino_global']:.4f} "
            f"ibot={avg['ibot']:.4f} koleo={avg['koleo']:.4f} | "
            f"lr={lr:.2e} | {elapsed:.0f}s"
        )
        log.append({"epoch": epoch + 1, **avg, "lr": lr})
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

        if use_wandb:
            wandb.log({
                "epoch":                  epoch + 1,
                "train/loss_total":       avg["total"],
                "train/loss_dino_local":  avg["dino_local"],
                "train/loss_dino_global": avg["dino_global"],
                "train/loss_ibot":        avg["ibot"],
                "train/loss_koleo":       avg["koleo"],
                "train/lr":               lr,
            })

        ckpt = {
            "epoch":      epoch + 1,
            "student":    model.student.state_dict(),
            "teacher":    model.teacher.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "lora_config": {
                "lora_layers":         lora_layers,
                "lora_r":              lora_r,
                "lora_alpha":          lora_alpha,
                "unfreeze_blocks":     unfreeze_blocks,
                "head_mode":           head_mode,
                "head_unfreeze_epoch": head_unfreeze_epoch,
            },
        }
        torch.save(ckpt, out_dir / "checkpoint_last.pth")
        if (epoch + 1) % save_freq == 0:
            torch.save(ckpt, out_dir / f"checkpoint_ep{epoch + 1:03d}.pth")

        if es_patience is not None:
            if avg["total"] < best_loss - es_min_delta:
                best_loss = avg["total"]
                no_improve = 0
                torch.save(ckpt, out_dir / "checkpoint_best.pth")
                print(f"  New best loss: {best_loss:.4f} — saved checkpoint_best.pth")
            else:
                no_improve += 1
                print(f"  No improvement for {no_improve}/{es_patience} epochs "
                      f"(best={best_loss:.4f})")
                if no_improve >= es_patience:
                    print(f"Early stopping triggered after epoch {epoch + 1}.")
                    break

    print(f"\nTraining complete. Final checkpoint: {out_dir / 'checkpoint_last.pth'}")
    if use_wandb:
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CheXFound iBOT continued pretraining")
    p.add_argument("--config",   required=True,  help="Path to the override YAML.")
    p.add_argument("--base_cfg", default=None,   help="Optional base YAML to merge beneath --config.")
    p.add_argument("--out_dir",  required=True,  help="Directory for checkpoints and logs.")
    p.add_argument("--epochs",     type=int, default=None, help="Override epochs (useful for sweep trials).")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch_size_per_gpu.")
    p.add_argument("--sweep",  action="store_true", help="Read hyperparams from wandb.config (W&B sweep mode).")
    p.add_argument("--wandb",  action="store_true", help="Enable W&B logging without a sweep.")
    p.add_argument("--wandb_project", default="chexfound-pretrain")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--early_stopping_patience", type=int, default=None,
                   help="Stop if total loss does not improve for this many epochs. "
                        "Overrides train.early_stopping_patience in the config.")
    p.add_argument("--early_stopping_min_delta", type=float, default=None,
                   help="Minimum loss decrease to count as improvement (default 0.0).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    cfg  = merge_configs(args.base_cfg, args.config)

    if args.epochs is not None:
        cfg.setdefault("optim",  {})["epochs"]            = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size_per_gpu"] = args.batch_size
    if args.early_stopping_patience is not None:
        cfg.setdefault("train", {})["early_stopping_patience"] = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        cfg.setdefault("train", {})["early_stopping_min_delta"] = args.early_stopping_min_delta

    use_wandb = (args.wandb or args.sweep) and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        if args.sweep:
            _apply_sweep_cfg(cfg)

    # Each sweep run gets its own subdirectory named after the W&B run ID
    # so checkpoints from different runs don't overwrite each other.
    if use_wandb and wandb.run is not None:
        out_dir = Path(args.out_dir) / wandb.run.id
    else:
        out_dir = Path(args.out_dir)

    train(cfg, out_dir, use_wandb=use_wandb)
