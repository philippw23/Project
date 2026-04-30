"""Self-contained CheXFound iBOT continued-pretraining script.

Implements the joint DINO + iBOT (masked-image-modelling) objective used by
CheXFound, without any dependency on the external CheXFound / DINOv2 repository.

Architecture and hyperparameter defaults are taken directly from
configs/chexfound_vitl16_bonetumor.yaml (which overrides src/chexfound/data/config.yaml).

Usage (single GPU via torchrun):
    torchrun --nproc_per_node=1 src/chexfound/train/pretrain.py \\
        --config   configs/chexfound_vitl16_bonetumor.yaml \\
        --base_cfg src/chexfound/data/config.yaml \\
        --out_dir  results/chexfound_pretrain

The script saves checkpoints as:
    results/chexfound_pretrain/
        checkpoint_ep{N}.pth   — periodic snapshots
        checkpoint_last.pth    — always overwritten with the latest epoch
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

# Make project root importable when called via torchrun from the project root.
_SRC = str(Path(__file__).resolve().parents[3])  # …/Project/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from chexfound.bone_tumor_patch.bone_tumor import BoneTumorDataset
from chexfound.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from chexfound.models.lora import inject_lora_chexfound
from chexfound.models.model_factory import build_model_from_cfg
from chexfound.models.weight_utils import load_pretrained_weights
from biomedclip.data.splits import build_stratified_splits
from biomedclip.utils.misc import DEFAULT_EXCEL, DEFAULT_REPORTS, DEFAULT_MASKS_DIR


def _multicrop_collate(batch: list) -> list[torch.Tensor]:
    """Collate a batch of (list_of_crops, target) samples into a list of (B, C, H, W) tensors.
    Targets are discarded — pretraining uses only the crops.
    """
    crops_per_sample = [sample[0] for sample in batch]
    n_crops = len(crops_per_sample[0])
    return [torch.stack([crops_per_sample[b][i] for b in range(len(batch))]) for i in range(n_crops)]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ══════════════════════════════════════════════════════════════════════════════
# Projection heads
# ══════════════════════════════════════════════════════════════════════════════

class DINOHead(nn.Module):
    """MLP projection head followed by a weight-normalised prototype layer.

    Architecture (nlayers=3):
        Linear(in_dim → hidden_dim) → GELU
        Linear(hidden_dim → hidden_dim) → GELU
        Linear(hidden_dim → bottleneck_dim)
        L2-normalise
        WeightNorm-Linear(bottleneck_dim → n_prototypes, no bias)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 384,
        nlayers: int = 3,
    ) -> None:
        super().__init__()
        nlayers = max(1, nlayers)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        for _ in range(nlayers - 2):
            layers += [nn.GELU(), nn.Linear(hidden_dim, hidden_dim)]
        layers += [nn.GELU(), nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
        # Weight-normalised prototype layer (bias frozen at zero).
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad_(False)

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)



# ══════════════════════════════════════════════════════════════════════════════
# Multi-crop data augmentation
# ══════════════════════════════════════════════════════════════════════════════

class MultiCropAugmentation:
    """Produces n_global_crops global views + n_local_crops local views.

    Returns a list: [global_0, global_1, local_0, …, local_{n-1}]
    """

    def __init__(
        self,
        global_crops_scale: tuple = (0.32, 1.0),
        local_crops_scale: tuple = (0.05, 0.32),
        global_crops_size: int = 512,
        local_crops_size: int = 144,
        n_local_crops: int = 8,
    ) -> None:
        flip_blur = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        ]
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)

        self.global_tf = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            *flip_blur,
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            normalize,
        ])
        self.local_tf = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            *flip_blur,
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            normalize,
        ])
        self.n_local_crops = n_local_crops

    def __call__(self, img) -> list[torch.Tensor]:
        crops = [self.global_tf(img), self.global_tf(img)]
        crops += [self.local_tf(img) for _ in range(self.n_local_crops)]
        return crops


# ══════════════════════════════════════════════════════════════════════════════
# iBOT patch masking
# ══════════════════════════════════════════════════════════════════════════════

def generate_patch_masks(
    batch_size: int,
    n_patches: int,
    mask_ratio_min: float = 0.1,
    mask_ratio_max: float = 0.5,
    mask_sample_probability: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Return a (B, N) bool mask — True = replace with mask token."""
    masks = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=device)
    for b in range(batch_size):
        if torch.rand(1).item() > mask_sample_probability:
            continue
        ratio = mask_ratio_min + torch.rand(1).item() * (mask_ratio_max - mask_ratio_min)
        n_masked = max(1, int(ratio * n_patches))
        idx = torch.randperm(n_patches, device=device)[:n_masked]
        masks[b, idx] = True
    return masks


# ══════════════════════════════════════════════════════════════════════════════
# Cosine / linear schedules
# ══════════════════════════════════════════════════════════════════════════════

def cosine_schedule(
    start: float,
    end: float,
    n_steps: int,
    warmup_steps: int = 0,
    warmup_start: float = 0.0,
) -> list[float]:
    """Cosine decay from *start* to *end* over *n_steps*, with optional warmup."""
    schedule = []
    for i in range(n_steps):
        if i < warmup_steps:
            v = warmup_start + (start - warmup_start) * i / max(warmup_steps, 1)
        else:
            t = (i - warmup_steps) / max(n_steps - warmup_steps, 1)
            v = end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))
        schedule.append(v)
    return schedule


# ══════════════════════════════════════════════════════════════════════════════
# Teacher EMA update
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def update_teacher(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float,
) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset / config helpers
# ══════════════════════════════════════════════════════════════════════════════

def parse_dataset_path(dataset_path: str) -> tuple[str, dict]:
    """Parse 'DatasetName:key=val:key=val' into (name, kwargs)."""
    parts = dataset_path.split(":")
    name = parts[0]
    kwargs: dict = {}
    for part in parts[1:]:
        k, _, v = part.partition("=")
        kwargs[k] = v
    return name, kwargs


def build_dataset(dataset_path: str, transform) -> torch.utils.data.Dataset:
    name, kwargs = parse_dataset_path(dataset_path)
    if name == "BoneTumor":
        import types
        args = types.SimpleNamespace(
            excel=kwargs.get("excel", str(DEFAULT_EXCEL)),
            reports=kwargs.get("reports", str(DEFAULT_REPORTS)),
            images=kwargs.get("root", "."),
            masks=kwargs.get("masks", str(DEFAULT_MASKS_DIR)),
            english=False,
            downstream_train_frac=float(kwargs.get("downstream_train_frac", 0.8)),
            downstream_val_frac=float(kwargs.get("downstream_val_frac", 0.1)),
            test_frac=float(kwargs.get("test_frac", 0.1)),
            seed=int(kwargs.get("seed", 42)),
        )
        pretrain_samples, _, _, _ = build_stratified_splits(args)
        image_paths = [s[0] for s in pretrain_samples]
        return BoneTumorDataset(image_paths=image_paths, root=args.images, transforms=transform)
    raise ValueError(f"Unknown dataset: {name!r}. Supported: 'BoneTumor'.")


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


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: dict, out_dir: Path) -> None:
    # ── Distributed setup ────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir : {out_dir}")
        print(f"Device     : {device}  |  world_size={world_size}")

    # ── Config ───────────────────────────────────────────────────────────────
    train_cfg   = cfg.get("train",   {})
    student_cfg = cfg.get("student", {})
    teacher_cfg = cfg.get("teacher", {})
    optim_cfg   = cfg.get("optim",   {})
    dino_cfg    = cfg.get("dino",    {})
    ibot_cfg    = cfg.get("ibot",    {})
    crops_cfg   = cfg.get("crops",   {})

    n_epochs         = optim_cfg.get("epochs", 50)
    batch_per_gpu    = train_cfg.get("batch_size_per_gpu", 4)
    n_workers        = min(train_cfg.get("num_workers", 4), 8)
    epoch_len        = train_cfg.get("OFFICIAL_EPOCH_LENGTH", 2500)
    save_freq        = train_cfg.get("saveckp_freq", 10)
    seed             = train_cfg.get("seed", 0)
    warmup_epochs    = optim_cfg.get("warmup_epochs", 5)
    base_lr          = optim_cfg.get("base_lr", 5e-5)
    min_lr           = optim_cfg.get("min_lr", 1e-6)
    wd_start         = optim_cfg.get("weight_decay", 0.04)
    wd_end           = optim_cfg.get("weight_decay_end", 0.2)
    clip_grad        = optim_cfg.get("clip_grad", 3.0)

    dino_weight      = dino_cfg.get("loss_weight", 1.0)
    ibot_weight      = ibot_cfg.get("loss_weight", 3.0)
    koleo_weight     = dino_cfg.get("koleo_loss_weight", 0.1)
    dino_prototypes  = dino_cfg.get("head_n_prototypes", 131072)
    dino_bottleneck  = dino_cfg.get("head_bottleneck_dim", 384)
    dino_hidden      = dino_cfg.get("head_hidden_dim", 2048)
    dino_nlayers     = dino_cfg.get("head_nlayers", 3)
    ibot_prototypes  = ibot_cfg.get("head_n_prototypes", 131072)
    ibot_bottleneck  = ibot_cfg.get("head_bottleneck_dim", 256)
    ibot_hidden      = dino_cfg.get("head_hidden_dim", 2048)
    ibot_nlayers     = dino_cfg.get("head_nlayers", 3)
    mask_prob        = ibot_cfg.get("mask_sample_probability", 0.5)
    mask_min         = ibot_cfg.get("mask_ratio_min_max", [0.1, 0.5])[0]
    mask_max         = ibot_cfg.get("mask_ratio_min_max", [0.1, 0.5])[1]
    centering        = train_cfg.get("centering", "sinkhorn_knopp")

    momentum_start   = teacher_cfg.get("momentum_teacher", 0.994)
    momentum_end     = teacher_cfg.get("final_momentum_teacher", 1.0)
    warmup_t_temp    = teacher_cfg.get("warmup_teacher_temp", 0.04)
    teacher_temp     = teacher_cfg.get("teacher_temp", 0.07)
    warmup_t_epochs  = teacher_cfg.get("warmup_teacher_temp_epochs", 30)

    global_size      = crops_cfg.get("global_crops_size", 512)
    local_size       = crops_cfg.get("local_crops_size", 144)
    global_scale     = tuple(crops_cfg.get("global_crops_scale", [0.32, 1.0]))
    local_scale      = tuple(crops_cfg.get("local_crops_scale", [0.05, 0.32]))
    n_local          = crops_cfg.get("local_crops_number", 8)

    torch.manual_seed(seed + local_rank)

    # ── Models ───────────────────────────────────────────────────────────────
    # Build from the already merged runtime config so architecture overrides in
    # --config are actually reflected in the model.
    student_trunk, embed_dim = build_model_from_cfg(cfg)
    teacher_trunk = copy.deepcopy(student_trunk)

    # Optionally warm-start from MODEL.WEIGHTS (original CheXFound checkpoint).
    warm_start = cfg.get("MODEL", {}).get("WEIGHTS", "")
    if warm_start and Path(warm_start).is_file():
        if is_main:
            print(f"Warm-starting from {warm_start}")
        load_pretrained_weights(student_trunk, warm_start, checkpoint_key="teacher")
        load_pretrained_weights(teacher_trunk, warm_start, checkpoint_key="teacher")

    # Optionally inject LoRA into the student trunk (freezes backbone, trains adapters only).
    lora_cfg = cfg.get("lora", {})
    lora_layers = lora_cfg.get("lora_layers", 0)
    if lora_layers > 0:
        # Student and EMA teacher must have identical module structure. Inject
        # LoRA into both trunks, then keep the teacher frozen and update it only
        # through EMA below.
        inject_lora_chexfound(
            student_trunk,
            lora_layers=lora_layers,
            r=lora_cfg.get("r", 8),
            alpha=lora_cfg.get("alpha", 16.0),
        )
        inject_lora_chexfound(
            teacher_trunk,
            lora_layers=lora_layers,
            r=lora_cfg.get("r", 8),
            alpha=lora_cfg.get("alpha", 16.0),
        )
        teacher_trunk.load_state_dict(student_trunk.state_dict())
        if is_main:
            trainable = sum(p.numel() for p in student_trunk.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in student_trunk.parameters())
            print(f"LoRA: last {lora_layers} blocks | r={lora_cfg.get('r', 8)} | "
                  f"trainable trunk params: {trainable:,} / {total:,} "
                  f"({100 * trainable / total:.2f}%)")

    # Teacher is never directly trained (plain ViT, no LoRA).
    for p in teacher_trunk.parameters():
        p.requires_grad_(False)

    student_trunk = student_trunk.to(device)
    teacher_trunk = teacher_trunk.to(device)

    # Projection heads.
    dino_head_s = DINOHead(embed_dim, dino_prototypes,
                           hidden_dim=dino_hidden, bottleneck_dim=dino_bottleneck,
                           nlayers=dino_nlayers).to(device)
    dino_head_t = copy.deepcopy(dino_head_s)
    for p in dino_head_t.parameters():
        p.requires_grad_(False)

    ibot_separate = ibot_cfg.get("separate_head", True)
    if ibot_separate:
        ibot_head_s = DINOHead(embed_dim, ibot_prototypes,
                               hidden_dim=ibot_hidden, bottleneck_dim=ibot_bottleneck,
                               nlayers=ibot_nlayers).to(device)
        ibot_head_t = copy.deepcopy(ibot_head_s)
        for p in ibot_head_t.parameters():
            p.requires_grad_(False)
    else:
        ibot_head_s = dino_head_s
        ibot_head_t = dino_head_t

    # Wrap in DDP only when multi-GPU.
    if world_size > 1:
        student_trunk  = nn.parallel.DistributedDataParallel(student_trunk,  device_ids=[local_rank])
        dino_head_s    = nn.parallel.DistributedDataParallel(dino_head_s,    device_ids=[local_rank])
        if ibot_separate:
            ibot_head_s = nn.parallel.DistributedDataParallel(ibot_head_s,   device_ids=[local_rank])

    # ── Losses ───────────────────────────────────────────────────────────────
    dino_loss = DINOLoss(
        out_dim=dino_prototypes,
    ).to(device)
    ibot_loss = iBOTPatchLoss(
        patch_out_dim=ibot_prototypes,
    ).to(device)
    koleo_loss = KoLeoLoss().to(device)

    # Teacher temperature schedule — warm up from warmup_t_temp to teacher_temp.
    teacher_temp_schedule = [
        warmup_t_temp + (teacher_temp - warmup_t_temp) * i / max(warmup_t_epochs - 1, 1)
        if i < warmup_t_epochs else teacher_temp
        for i in range(n_epochs)
    ]

    # ── Dataset ──────────────────────────────────────────────────────────────
    aug = MultiCropAugmentation(
        global_crops_scale=global_scale,
        local_crops_scale=local_scale,
        global_crops_size=global_size,
        local_crops_size=local_size,
        n_local_crops=n_local,
    )
    dataset = build_dataset(train_cfg["dataset_path"], transform=aug)
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        if world_size > 1 else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_per_gpu,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=_multicrop_collate,
    )
    # Use epoch_len to limit steps per epoch (matching CheXFound convention).
    steps_per_epoch = min(epoch_len, len(loader))
    total_steps = n_epochs * steps_per_epoch

    # ── Optimiser ────────────────────────────────────────────────────────────
    all_params = (
        [p for p in student_trunk.parameters() if p.requires_grad]
        + list(dino_head_s.parameters())
        + (list(ibot_head_s.parameters()) if ibot_separate else [])
    )
    optimizer = torch.optim.AdamW(
        all_params,
        lr=base_lr,
        betas=(optim_cfg.get("adamw_beta1", 0.9), optim_cfg.get("adamw_beta2", 0.999)),
        weight_decay=wd_start,
    )

    lr_schedule = cosine_schedule(
        start=base_lr, end=min_lr,
        n_steps=total_steps,
        warmup_steps=warmup_epochs * steps_per_epoch,
        warmup_start=min_lr,
    )
    wd_schedule = cosine_schedule(start=wd_start, end=wd_end, n_steps=total_steps)
    mom_schedule = cosine_schedule(start=momentum_start, end=momentum_end,
                                   n_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # ── Number of patches (needed for mask generation) ───────────────────────
    n_patches = (global_size // student_cfg.get("patch_size", 16)) ** 2

    # ── Training loop ────────────────────────────────────────────────────────
    if is_main:
        print(f"\nStarting iBOT training: {n_epochs} epochs × {steps_per_epoch} steps")
        print(f"  Dataset  : {len(dataset)} images")
        print(f"  Batch    : {batch_per_gpu} × {world_size} GPU(s)")

    global_step = 0
    log: list[dict] = []

    for epoch in range(n_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_losses: dict[str, float] = {
            "dino": 0.0, "ibot": 0.0, "koleo": 0.0, "total": 0.0
        }
        t0 = time.time()

        data_iter: Iterator = iter(loader)
        for _ in range(steps_per_epoch):
            # Update LR, WD, teacher momentum.
            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule[global_step]
                pg["weight_decay"] = wd_schedule[global_step]
            momentum = mom_schedule[global_step]

            # ── Fetch batch ──────────────────────────────────────────────────
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            # _multicrop_collate returns a list of (B, C, H, W) tensors, one per crop.
            crops = [c.to(device) for c in batch]

            global_crops = crops[:2]
            local_crops  = crops[2:]
            B = global_crops[0].shape[0]

            # ── Generate iBOT masks for global crops ─────────────────────────
            masks = [
                generate_patch_masks(B, n_patches, mask_min, mask_max,
                                     mask_prob, device)
                for _ in global_crops
            ]

            # ── Teacher forward (no grad, no masking) ────────────────────────
            t_temp = teacher_temp_schedule[epoch]
            with torch.no_grad():
                teacher_out_g: list[dict] = []
                for gc in global_crops:
                    teacher_out_g.append(teacher_trunk.forward_features(gc))

                teacher_cls   = [o["x_norm_clstoken"]   for o in teacher_out_g]
                teacher_patch = [o["x_norm_patchtokens"] for o in teacher_out_g]
                teacher_dino_raw = [dino_head_t(c) for c in teacher_cls]
                teacher_ibot_raw = [ibot_head_t(p) for p in teacher_patch]

                # Center and sharpen teacher outputs before passing to losses.
                if centering == "sinkhorn_knopp":
                    teacher_dino = [dino_loss.sinkhorn_knopp_teacher(t, t_temp)
                                    for t in teacher_dino_raw]
                else:
                    teacher_dino = [dino_loss.softmax_center_teacher(t, t_temp)
                                    for t in teacher_dino_raw]
                teacher_ibot = [ibot_loss.softmax_center_teacher(t, t_temp)
                                for t in teacher_ibot_raw]

            # ── Student forward (with masking on global crops) ───────────────
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                student_dino_logits: list[torch.Tensor] = []
                student_ibot_logits: list[torch.Tensor] = []
                student_cls_tokens:  list[torch.Tensor] = []

                for gc, mask in zip(global_crops, masks):
                    feat = student_trunk.forward_features(gc, masks=mask)
                    student_cls_tokens.append(feat["x_norm_clstoken"])
                    student_dino_logits.append(dino_head_s(feat["x_norm_clstoken"]))
                    student_ibot_logits.append(ibot_head_s(feat["x_norm_patchtokens"]))

                for lc in local_crops:
                    feat = student_trunk.forward_features(lc)
                    student_dino_logits.append(dino_head_s(feat["x_norm_clstoken"]))

                # ── Losses ───────────────────────────────────────────────────
                n_dino_terms = len(student_dino_logits) * len(teacher_dino)
                loss_dino  = dino_loss(student_dino_logits, teacher_dino) / n_dino_terms
                loss_ibot  = sum(
                    ibot_loss(s, t, mask)
                    for s, t, mask in zip(student_ibot_logits, teacher_ibot, masks)
                ) / len(global_crops)
                loss_koleo = koleo_loss(torch.cat(student_cls_tokens))

                loss = (dino_weight  * loss_dino
                        + ibot_weight  * loss_ibot
                        + koleo_weight * loss_koleo)

            # Async center updates for next step (outside autocast, use raw logits).
            dino_loss.update_center(torch.cat(teacher_dino_raw))
            # Pass only masked patch tokens — shape (1, total_masked, K) — matching original.
            masked_teacher_ibot = torch.cat(
                [t[mask] for t, mask in zip(teacher_ibot_raw, masks)], dim=0
            ).unsqueeze(0)
            ibot_loss.update_center(masked_teacher_ibot)

            # ── Optimiser step ───────────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(all_params, clip_grad)
            scaler.step(optimizer)
            scaler.update()

            # ── Teacher EMA ──────────────────────────────────────────────────
            _s_trunk = student_trunk.module if world_size > 1 else student_trunk
            _s_dino  = dino_head_s.module  if world_size > 1 else dino_head_s
            update_teacher(_s_trunk, teacher_trunk, momentum)
            update_teacher(_s_dino, dino_head_t, momentum)
            if ibot_separate:
                _s_ibot = ibot_head_s.module if world_size > 1 else ibot_head_s
                update_teacher(_s_ibot, ibot_head_t, momentum)

            # Accumulate for logging.
            epoch_losses["dino"]  += loss_dino.item()
            epoch_losses["ibot"]  += loss_ibot.item()
            epoch_losses["koleo"] += loss_koleo.item()
            epoch_losses["total"] += loss.item()

            global_step += 1

        # ── End-of-epoch logging & checkpointing ─────────────────────────────
        if is_main:
            avg = {k: v / steps_per_epoch for k, v in epoch_losses.items()}
            lr  = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch + 1:03d}/{n_epochs} | "
                f"total={avg['total']:.4f} dino={avg['dino']:.4f} "
                f"ibot={avg['ibot']:.4f} koleo={avg['koleo']:.4f} | "
                f"lr={lr:.2e} | {elapsed:.0f}s"
            )
            log.append({"epoch": epoch + 1, **avg, "lr": lr})
            (out_dir / "log.json").write_text(json.dumps(log, indent=2))

            # Save checkpoint.
            _s_trunk = student_trunk.module if world_size > 1 else student_trunk
            ckpt = {
                "epoch": epoch + 1,
                "student": _s_trunk.state_dict(),
                "teacher": teacher_trunk.state_dict(),
                "optimizer": optimizer.state_dict(),
                "dino_loss_center": dino_loss.center,
                "ibot_loss_center": ibot_loss.center,
                "lora_config": {
                    "lora_layers": lora_layers,
                    "lora_r": lora_cfg.get("r", 8),
                    "lora_alpha": lora_cfg.get("alpha", 16.0),
                },
            }
            torch.save(ckpt, out_dir / "checkpoint_last.pth")
            if (epoch + 1) % save_freq == 0:
                torch.save(ckpt, out_dir / f"checkpoint_ep{epoch + 1:03d}.pth")

    if world_size > 1:
        dist.destroy_process_group()

    if is_main:
        print("\nTraining complete.")
        print(f"Final checkpoint: {out_dir / 'checkpoint_last.pth'}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CheXFound iBOT continued pretraining")
    p.add_argument(
        "--config", required=True,
        help="Path to the override YAML (e.g. configs/chexfound_vitl16_bonetumor.yaml).",
    )
    p.add_argument(
        "--base_cfg", default=None,
        help="Optional base YAML to merge beneath --config "
             "(e.g. src/chexfound/data/config.yaml).",
    )
    p.add_argument(
        "--out_dir", required=True,
        help="Directory for checkpoints and logs.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    cfg  = merge_configs(args.base_cfg, args.config)
    train(cfg, Path(args.out_dir))
