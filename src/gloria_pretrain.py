"""GLoRIA continued pretraining on the bone-tumor X-ray dataset.

Further adapts the CheXPert-pretrained GLoRIA (ResNet50 + BERT) to the
bone-tumor domain using GLoRIA's global+local contrastive loss.
The BERT text encoder is kept fully frozen; only the image encoder is
adapted via LoRA or partial unfreezing.

Usage:
    python src/gloria_pretrain.py \\
        --checkpoint src/gloria/pretrained/chexpert_resnet50.ckpt \\
        --splits data/internal_dataset/split.json \\
        --adapter_mode lora --n_layers 2 --lora_r 8
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import math
import random
import shutil
import sys
import types as _types
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.utils.misc import ROOT_DIR, DEFAULT_OUT_DIR
from biomedclip.data.splits import build_stratified_splits
from biomedclip.data.transforms import crop_around_mask

# ── GLoRIA module loading (avoids the old pytorch-lightning environment) ───────

GLORIA_DIR = Path(__file__).resolve().parent / "gloria"
if str(GLORIA_DIR) not in sys.path:
    sys.path.insert(0, str(GLORIA_DIR))

DEFAULT_GLORIA_CKPT = ROOT_DIR / "src" / "gloria" / "pretrained" / "chexpert_resnet50.ckpt"
DEFAULT_SPLITS       = ROOT_DIR / "data" / "internal_dataset" / "split.json"
DEFAULT_DATASET_JSON = ROOT_DIR / "data" / "internal_dataset" / "dataset_full.json"
GLORIA_NORM          = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class _AutoStub(_types.ModuleType):
    class _NoOp:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return self
        def __getattr__(self, n): return self.__class__()

    def __getattr__(self, name):
        return self._NoOp


class _StubFinder:
    _STUB_PREFIXES = ("pytorch_lightning", "skimage", "nltk", "cv2")

    def find_module(self, name, path=None):
        if any(name == p or name.startswith(p + ".") for p in self._STUB_PREFIXES):
            return self

    def load_module(self, name):
        if name not in sys.modules:
            sys.modules[name] = _AutoStub(name)
        return sys.modules[name]


sys.meta_path.insert(0, _StubFinder())


def _load_gloria_submodule(pkg_name: str, rel_path: str):
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    path = GLORIA_DIR / "gloria" / rel_path
    spec = _ilu.spec_from_file_location(pkg_name, path)
    mod  = _ilu.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_gloria_modules():
    for pkg, subpath in [
        ("gloria",        ""),
        ("gloria.models", "models"),
        ("gloria.loss",   "loss"),
    ]:
        if pkg not in sys.modules:
            m = _types.ModuleType(pkg)
            m.__path__   = [str(GLORIA_DIR / "gloria" / subpath)]
            m.__package__ = pkg
            sys.modules[pkg] = m

    _load_gloria_submodule("gloria.models.cnn_backbones", "models/cnn_backbones.py")
    vision_mod = _load_gloria_submodule("gloria.models.vision_model", "models/vision_model.py")
    text_mod   = _load_gloria_submodule("gloria.models.text_model",   "models/text_model.py")
    lora_mod   = _load_gloria_submodule("gloria.models.lora",         "models/lora.py")
    loss_mod   = _load_gloria_submodule("gloria.loss.gloria_loss",    "loss/gloria_loss.py")
    return vision_mod, text_mod, lora_mod, loss_mod


# ── Data ──────────────────────────────────────────────────────────────────────

def build_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4)),
        transforms.ToTensor(),
        transforms.Normalize(*GLORIA_NORM),
    ])


def build_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*GLORIA_NORM),
    ])


class GLoRIAPairDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, Path, str]], transform, tokenizer,
                 max_text_len: int, use_mask: bool = False):
        self.samples      = samples
        self.transform    = transform
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len
        self.use_mask     = use_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, mask_path, text = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.use_mask and mask_path is not None and mask_path.exists():
            image_arr = np.array(image.convert("L"), dtype=float)
            mask_arr  = np.array(Image.open(mask_path).convert("L"), dtype=float)
            cropped   = crop_around_mask(image_arr, mask_arr)
            image     = Image.fromarray(cropped.astype(np.uint8)).convert("RGB")

        image = self.transform(image)

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
        )
        return {
            "image":          image,
            "caption_ids":    enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
        }


def _build_samples(split_samples: list[dict]) -> list[tuple[Path, Path | None, str]]:
    """Extract (image_path, mask_path, full_en_text) triples, dropping entries with empty text."""
    result = []
    dropped = 0
    for s in split_samples:
        befund      = (s.get("befund_en") or "").strip()
        beurteilung = (s.get("beurteilung_en") or "").strip()
        text = " ".join(filter(None, [befund, beurteilung]))
        if not text:
            dropped += 1
            continue
        mask = Path(s["mask"]) if s.get("mask") else None
        result.append((Path(s["image"]), mask, text))
    if dropped:
        print(f"  Dropped {dropped} samples with empty report text.")
    return result


# ── Model setup ───────────────────────────────────────────────────────────────

def load_gloria_encoders(ckpt_path: Path, adapter_mode: str, n_layers: int,
                          lora_r: int, lora_alpha: float):
    """Load GLoRIA image + text encoders from a CheXPert lightning checkpoint.

    Returns (img_encoder, text_encoder, tokenizer, original_cfg_container).
    """
    from omegaconf import OmegaConf

    vision_mod, text_mod, lora_mod, _ = _ensure_gloria_modules()
    ImageEncoder = vision_mod.ImageEncoder
    BertEncoder  = text_mod.BertEncoder

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["hyper_parameters"]

    # Build encoders
    img_encoder  = ImageEncoder(cfg)
    text_encoder = BertEncoder(cfg)

    # Load pretrained weights
    img_state = {
        k.replace("gloria.img_encoder.", "", 1): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("gloria.img_encoder.")
    }
    img_encoder.load_state_dict(img_state, strict=True)

    txt_state = {
        k.replace("gloria.text_encoder.", "", 1): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("gloria.text_encoder.")
    }
    # Newer transformers no longer registers the `position_ids` buffer that older
    # checkpoints stored; drop it so the remaining weights still load strictly.
    txt_state = {k: v for k, v in txt_state.items() if not k.endswith("position_ids")}
    if txt_state:
        text_encoder.load_state_dict(txt_state, strict=True)

    # Apply adapter to image encoder
    if adapter_mode == "lora":
        lora_mod.inject_lora_gloria(img_encoder, lora_layers=n_layers, r=lora_r, alpha=lora_alpha)
        print(f"LoRA injected: layer4 (all blocks) + last {n_layers} layer3 blocks "
              f"(r={lora_r}, alpha={lora_alpha})")
    else:
        lora_mod.unfreeze_gloria(img_encoder, n_layers=n_layers)
        print(f"Unfrozen: layer4 (all blocks) + last {n_layers} layer3 blocks + projection heads")

    # Freeze BERT entirely (text encoder body + projection heads)
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    n_img  = lora_mod.count_trainable_params(img_encoder)
    n_tot  = sum(p.numel() for p in img_encoder.parameters())
    print(f"Image encoder — trainable: {n_img:,} / {n_tot:,} ({100 * n_img / n_tot:.2f}%)")
    print("Text encoder — fully frozen")

    tokenizer = img_encoder.__class__  # placeholder; actual tokenizer loaded below
    bert_type = cfg.model.text.bert_type
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_type)

    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    return img_encoder, text_encoder, tokenizer, cfg_container


# ── Training / evaluation ─────────────────────────────────────────────────────

def make_scheduler(optimizer, warmup_epochs: int, total_epochs: int,
                   min_lr_frac: float = 0.1):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(img_encoder, text_encoder, loader, optimizer, scaler, device,
                    local_loss_fn, global_loss_fn, args) -> float:
    img_encoder.train()
    text_encoder.eval()  # always eval: frozen
    total_loss = 0.0

    pbar = tqdm(loader, desc="  train", leave=False, disable=not sys.stdout.isatty())
    for batch in pbar:
        images          = batch["image"].to(device)
        caption_ids     = batch["caption_ids"].to(device)
        attention_mask  = batch["attention_mask"].to(device)
        token_type_ids  = batch["token_type_ids"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            global_ft, local_ft = img_encoder(images, get_local=True)
            img_emb_g, img_emb_l = img_encoder.generate_embeddings(global_ft, local_ft)

            with torch.no_grad():
                text_emb_l, text_emb_g, sents = text_encoder(
                    caption_ids, attention_mask, token_type_ids
                )

            cap_lens = [
                len([w for w in sent if not w.startswith("[")]) + 1
                for sent in sents
            ]

            l_loss0, l_loss1, _ = local_loss_fn(
                img_emb_l, text_emb_l, cap_lens,
                temp1=args.temp1, temp2=args.temp2, temp3=args.temp3,
            )
            g_loss0, g_loss1 = global_loss_fn(img_emb_g, text_emb_g, temp3=args.temp3)

            loss = (
                (l_loss0 + l_loss1) * args.local_loss_weight
                + (g_loss0 + g_loss1) * args.global_loss_weight
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable = [p for p in list(img_encoder.parameters()) if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loss(img_encoder, text_encoder, loader, device,
                  local_loss_fn, global_loss_fn, args) -> float:
    img_encoder.eval()
    text_encoder.eval()
    total_loss = 0.0

    for batch in loader:
        images         = batch["image"].to(device)
        caption_ids    = batch["caption_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            global_ft, local_ft = img_encoder(images, get_local=True)
            img_emb_g, img_emb_l = img_encoder.generate_embeddings(global_ft, local_ft)
            text_emb_l, text_emb_g, sents = text_encoder(
                caption_ids, attention_mask, token_type_ids
            )
            cap_lens = [
                len([w for w in sent if not w.startswith("[")]) + 1
                for sent in sents
            ]
            l_loss0, l_loss1, _ = local_loss_fn(
                img_emb_l, text_emb_l, cap_lens,
                temp1=args.temp1, temp2=args.temp2, temp3=args.temp3,
            )
            g_loss0, g_loss1 = global_loss_fn(img_emb_g, text_emb_g, temp3=args.temp3)
            loss = (
                (l_loss0 + l_loss1) * args.local_loss_weight
                + (g_loss0 + g_loss1) * args.global_loss_weight
            )
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_retrieval(img_encoder, text_encoder, val_samples, val_transform,
                        tokenizer, device, batch_size: int,
                        max_text_len: int, use_mask: bool = False) -> dict[str, float]:
    if len(val_samples) < 2:
        return {}

    img_encoder.eval()
    text_encoder.eval()

    ds     = GLoRIAPairDataset(val_samples, val_transform, tokenizer, max_text_len,
                               use_mask=use_mask)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=True)

    img_embs, txt_embs = [], []
    for batch in loader:
        images         = batch["image"].to(device)
        caption_ids    = batch["caption_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            global_ft = img_encoder(images, get_local=False)
            img_emb_g = img_encoder.global_embedder(global_ft)
            _, text_emb_g, _ = text_encoder(caption_ids, attention_mask, token_type_ids)

        img_embs.append(F.normalize(img_emb_g.float(), dim=-1))
        txt_embs.append(F.normalize(text_emb_g.float(), dim=-1))

    imgs = torch.cat(img_embs, dim=0)   # [N, D]
    txts = torch.cat(txt_embs, dim=0)   # [N, D]
    N    = imgs.shape[0]
    sim  = imgs @ txts.T                # [N, N]

    def _recall(mat):
        ranks = np.array([
            int((mat[i] > mat[i, i]).sum().item()) + 1 for i in range(N)
        ])
        return (ranks <= 1).mean(), (ranks <= 5).mean(), float(np.median(ranks))

    i2t_r1, i2t_r5, i2t_med = _recall(sim)
    t2i_r1, t2i_r5, t2i_med = _recall(sim.T)

    return {
        "retrieval/i2t_r1":          float(i2t_r1),
        "retrieval/i2t_r5":          float(i2t_r5),
        "retrieval/i2t_median_rank": float(i2t_med),
        "retrieval/t2i_r1":          float(t2i_r1),
        "retrieval/t2i_r5":          float(t2i_r5),
        "retrieval/t2i_median_rank": float(t2i_med),
        "retrieval/n_pairs":         float(N),
    }


def _save_checkpoint(path: Path, epoch: int, val_loss: float, mean_r1: float,
                     img_encoder, optimizer, args, cfg_container: dict) -> None:
    torch.save({
        "format":                "gloria_pretrain_v1",
        "epoch":                 epoch,
        "val_loss":              val_loss,
        "mean_r1":               mean_r1,
        "adapter_mode":          args.adapter_mode,
        "n_layers":              args.n_layers,
        "lora_r":                args.lora_r,
        "lora_alpha":            args.lora_alpha,
        "use_mask":              args.use_mask,
        "original_gloria_cfg":   cfg_container,
        "img_encoder_state_dict": img_encoder.state_dict(),
        "optimizer_state_dict":  optimizer.state_dict(),
    }, path)
    print(f"  Saved checkpoint -> {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GLoRIA continued pretraining on bone-tumor X-ray data."
    )
    p.add_argument("--checkpoint",  default=str(DEFAULT_GLORIA_CKPT),
                   help="Path to the GLoRIA lightning checkpoint (default: %(default)s)")
    p.add_argument("--splits",      default=str(DEFAULT_SPLITS),
                   help="Path to split.json (default: %(default)s)")
    p.add_argument("--dataset",     default=str(DEFAULT_DATASET_JSON),
                   help="dataset_full.json, used only when --splits is omitted")
    p.add_argument("--out_dir",     default=str(DEFAULT_OUT_DIR))

    # Adapter
    p.add_argument("--adapter_mode", choices=["lora", "unfreeze"], default="lora",
                   help="Image encoder adaptation strategy (default: %(default)s)")
    p.add_argument("--n_layers",  type=int,   default=2,
                   help="Number of last ResNet layer3 blocks to adapt "
                        "(layer4 is always included; default: %(default)s)")
    p.add_argument("--lora_r",    type=int,   default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)

    # Optimisation
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--warmup_epochs", type=int,  default=5)
    p.add_argument("--patience",     type=int,   default=15,
                   help="Early-stopping patience on mean R@1 (0 to disable)")

    # GLoRIA loss hyperparameters
    p.add_argument("--temp1", type=float, default=4.0,
                   help="Softmax temperature for local attention (default: %(default)s)")
    p.add_argument("--temp2", type=float, default=5.0,
                   help="Exponential scaling temperature (default: %(default)s)")
    p.add_argument("--temp3", type=float, default=10.0,
                   help="Global and final similarity temperature (default: %(default)s)")
    p.add_argument("--local_loss_weight",  type=float, default=1.0)
    p.add_argument("--global_loss_weight", type=float, default=1.0)

    # Text
    p.add_argument("--max_text_len", type=int, default=97,
                   help="Max BERT token length (default: %(default)s)")

    # Image
    p.add_argument("--use_mask", action="store_true",
                   help="Crop images around the lesion using segmentation masks")

    # W&B
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb_project", default="gloria-pretrain")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--wandb_run",     default=None)
    p.add_argument("--sweep",         action="store_true",
                   help="Run as wandb sweep agent (hyperparams from wandb.config)")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _apply_sweep_config(args: argparse.Namespace) -> None:
    cfg = wandb.config
    for key in ("lr", "weight_decay", "batch_size", "n_layers",
                "lora_r", "temp1", "temp2", "temp3",
                "local_loss_weight", "global_loss_weight"):
        if key in cfg:
            setattr(args, key, cfg[key])
    if args.adapter_mode == "lora":
        args.lora_alpha = 2.0 * args.lora_r


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load GLoRIA modules and loss functions ────────────────────────────────
    _, _, _, loss_mod = _ensure_gloria_modules()
    local_loss_fn  = loss_mod.local_loss
    global_loss_fn = loss_mod.global_loss

    # ── Load encoders ─────────────────────────────────────────────────────────
    img_encoder, text_encoder, tokenizer, cfg_container = load_gloria_encoders(
        Path(args.checkpoint), args.adapter_mode, args.n_layers,
        args.lora_r, args.lora_alpha,
    )
    img_encoder  = img_encoder.to(device)
    text_encoder = text_encoder.to(device)

    # ── Output directory ──────────────────────────────────────────────────────
    adapter_tag = (
        f"lora{args.n_layers}_r{args.lora_r}"
        if args.adapter_mode == "lora"
        else f"unfreeze{args.n_layers}"
    )
    run_name = f"gloria_pretrain_{adapter_tag}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = Path(args.out_dir) / "gloria_pretrain" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if Path(args.splits).exists():
        with open(args.splits, encoding="utf-8") as fh:
            split_data = json.load(fh)
        shutil.copy(args.splits, run_dir / "split.json")
        train_raw = split_data["train"]
        val_raw   = split_data["val"]
        print(f"Loaded split: {len(train_raw)} train / {len(val_raw)} val samples")
    else:
        train_raw, val_raw, _ = build_stratified_splits(args, run_dir=run_dir)

    train_samples = _build_samples(train_raw)
    val_samples   = _build_samples(val_raw)
    print(f"Paired samples: {len(train_samples)} train / {len(val_samples)} val")

    train_ds = GLoRIAPairDataset(train_samples, build_train_transform(), tokenizer,
                                 args.max_text_len, use_mask=args.use_mask)
    val_ds   = GLoRIAPairDataset(val_samples,   build_val_transform(),   tokenizer,
                                 args.max_text_len, use_mask=args.use_mask)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    trainable_params = [p for p in img_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler    = torch.cuda.amp.GradScaler()

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        if args.sweep:
            wandb.init()
            _apply_sweep_config(args)
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run or run_name,
                config=vars(args),
            )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_mean_r1  = -float("inf")
    patience_ctr  = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            img_encoder, text_encoder, train_loader, optimizer, scaler, device,
            local_loss_fn, global_loss_fn, args,
        )
        val_loss = evaluate_loss(
            img_encoder, text_encoder, val_loader, device,
            local_loss_fn, global_loss_fn, args,
        )
        ret = evaluate_retrieval(
            img_encoder, text_encoder, val_samples, build_val_transform(),
            tokenizer, device, args.batch_size, args.max_text_len,
            use_mask=args.use_mask,
        )

        mean_r1   = (ret.get("retrieval/i2t_r1", 0.0) + ret.get("retrieval/t2i_r1", 0.0)) / 2.0
        current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else args.lr

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"i2t_R@1={ret.get('retrieval/i2t_r1', 0):.4f}  "
              f"t2i_R@1={ret.get('retrieval/t2i_r1', 0):.4f}  "
              f"mean_R@1={mean_r1:.4f}  lr={current_lr:.2e}")

        scheduler.step()

        log_dict = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
            "mean_r1":    mean_r1,
            **ret,
        }
        if use_wandb:
            wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(run_dir / "best_val_loss_checkpoint.pt", epoch, val_loss,
                             mean_r1, img_encoder, optimizer, args, cfg_container)

        if mean_r1 > best_mean_r1:
            best_mean_r1 = mean_r1
            patience_ctr = 0
            _save_checkpoint(run_dir / "best_retrieval_checkpoint.pt", epoch, val_loss,
                             mean_r1, img_encoder, optimizer, args, cfg_container)
        else:
            patience_ctr += 1
            if args.patience > 0 and patience_ctr >= args.patience:
                print(f"  Early stopping: no improvement in mean R@1 for {args.patience} epochs.")
                break

    _save_checkpoint(run_dir / "final_checkpoint.pt", epoch, val_loss,
                     mean_r1, img_encoder, optimizer, args, cfg_container)
    print(f"\nTraining complete. Best mean R@1: {best_mean_r1:.4f}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
