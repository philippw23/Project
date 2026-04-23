"""BiomedCLIP contrastive pretraining with LoRA on the ViT image encoder.

Domain-adapts microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 on
(X-ray image, German radiology report) pairs from the bone-tumor dataset.
LoRA is injected into the last N transformer blocks of the ViT; the text
encoder (PubMedBERT) is kept fully frozen throughout.

Usage example:
    python src/biomedclip_pretrain.py \\
        --excel data/metadata.xlsx \\
        --reports data/text/sanitized_reports.json \\
        --use_mask --lora_layers 4 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES_DIR = ROOT_DIR / "data" / "images"
DEFAULT_MASKS_DIR = ROOT_DIR / "data" / "segmentations"
DEFAULT_EXCEL = ROOT_DIR / "data" / "metadata.xlsx"
DEFAULT_REPORTS = ROOT_DIR / "data" / "text" / "translated_reports.json" # "sanitized_reports.json"
DEFAULT_OUT_DIR = ROOT_DIR / "results"

MODEL_TAG = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


# ---------------------------------------------------------------------------
# Model inspection
# ---------------------------------------------------------------------------

def _count_parameters(module: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_biomedclip_architecture(
    model_tag: str = MODEL_TAG,
    device: torch.device | str | None = None,
    include_full_model: bool = False,
) -> None:
    """Load BioMedCLIP and print the image and text encoder architecture.

    BioMedCLIP is a dual encoder: images pass through the ViT visual encoder,
    while text token IDs pass through the PubMedBERT text encoder.  This helper
    is intentionally read-only and does not inject LoRA or start training.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Loading model: {model_tag}")
    model, _, _ = open_clip.create_model_and_transforms(model_tag)
    model = model.to(device)
    model.eval()

    total_params, trainable_params = _count_parameters(model)
    print("\n" + "=" * 80)
    print("BioMedCLIP model")
    print("=" * 80)
    print(f"Type      : {type(model).__name__}")
    print(f"Device    : {device}")
    print(f"Parameters: {total_params:,} total / {trainable_params:,} trainable")

    if include_full_model:
        print("\n" + "=" * 80)
        print("Full model")
        print("=" * 80)
        print(model)

    print("\n" + "=" * 80)
    print("Image encoder")
    print("=" * 80)
    image_encoder = getattr(model, "visual", None)
    if image_encoder is None:
        print("No 'visual' module found on the loaded model.")
    else:
        total, trainable = _count_parameters(image_encoder)
        print(f"Type      : {type(image_encoder).__name__}")
        print(f"Parameters: {total:,} total / {trainable:,} trainable")
        print(image_encoder)

    print("\n" + "=" * 80)
    print("Text encoder")
    print("=" * 80)
    text_encoder = getattr(model, "text", None)
    if text_encoder is not None:
        total, trainable = _count_parameters(text_encoder)
        print(f"Type      : {type(text_encoder).__name__}")
        print(f"Parameters: {total:,} total / {trainable:,} trainable")
        print(text_encoder)
        return

    text_module_names = [
        "token_embedding",
        "positional_embedding",
        "transformer",
        "ln_final",
        "text_projection",
    ]
    found_any = False
    for name in text_module_names:
        component = getattr(model, name, None)
        if component is None:
            continue
        found_any = True
        print(f"\n--- {name} ({type(component).__name__}) ---")
        if isinstance(component, nn.Module):
            total, trainable = _count_parameters(component)
            print(f"Parameters: {total:,} total / {trainable:,} trainable")
        elif isinstance(component, torch.Tensor):
            print(f"Shape: {tuple(component.shape)}")
        print(component)

    if not found_any:
        print("No dedicated text encoder attributes found on the loaded model.")


# ---------------------------------------------------------------------------
# Image cropping
# ---------------------------------------------------------------------------

def _compute_crop_1d(
    center: float,
    size: int,
    dim_size: int,
) -> tuple[int, int, int, int]:
    """Compute crop parameters along a single image axis.

    Centers a window of length *size* on *center*, then shifts it as far as
    possible into [0, dim_size) to maximise real pixels before any padding is
    added.  When size > dim_size the entire axis is used and padding is split
    to keep *center* as close to the output centre as possible.

    Returns (slice_start, slice_end, pad_before, pad_after).
    """
    if size <= dim_size:
        start = round(center - size / 2)
        start = max(0, min(start, dim_size - size))
        return start, start + size, 0, 0

    # size > dim_size: use the full axis, distribute padding around the lesion
    pad_total = size - dim_size
    ideal_pad_before = round(size / 2 - center)
    pad_before = max(0, min(pad_total, ideal_pad_before))
    return 0, dim_size, pad_before, pad_total - pad_before


def crop_around_mask(
    image_arr: np.ndarray,
    mask: np.ndarray,
    enable_crop: bool = True,
    context_fraction: float = 0.15,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
) -> np.ndarray:
    """Crop a square region around the lesion defined by *mask*.

    The crop side is computed as::

        context_px = ceil(min(H, W) * context_fraction)
        side = max(bbox_height, bbox_width) + 2 * context_px

    Because *context_px* is a fixed number of pixels (derived from image size,
    not lesion size), small lesions automatically receive proportionally more
    surrounding context than large lesions — which is the desired behaviour for
    radiological images where tiny findings need more anatomical landmarks.

    Algorithm
    ---------
    1. If *enable_crop* is False or the mask has no foreground pixels, the
       original image is returned unchanged.
    2. Compute the mask bounding box and lesion centroid.
    3. Determine square side: bounding-box longest dimension + 2 × context_px.
    4. Centre the square on the lesion centroid.
    5. Shift the window inside image bounds to maximise real content.
    6. If side still exceeds an image dimension, use the full extent and fill
       the remainder with *pad_mode* / *pad_value*.

    Parameters
    ----------
    image_arr : np.ndarray
        Grayscale (H, W) or colour (H, W, C) array.
    mask : np.ndarray
        Binary or soft mask, same spatial dimensions as *image_arr*.
    enable_crop : bool
        If False, skip all processing and return *image_arr* unchanged.
    context_fraction : float
        Context margin added on each side of the bounding box, expressed as a
        fraction of the image's shorter dimension (default 0.15).  A value of
        0.15 on a 512 px image gives 77 px of context per side.
    pad_mode : str
        ``numpy.pad`` mode used when padding is required (e.g. ``"constant"``,
        ``"reflect"``, ``"edge"``).
    pad_value : float
        Fill value when *pad_mode* is ``"constant"``.

    Returns
    -------
    np.ndarray
        Square crop of shape (side, side) or (side, side, C).
    """
    if not enable_crop:
        return image_arr

    binary_mask = mask > 0
    if not np.any(binary_mask):
        return image_arr

    rows, cols = np.where(binary_mask)
    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())

    box_h = r_max - r_min + 1
    box_w = c_max - c_min + 1

    H, W = image_arr.shape[:2]

    # Fixed context in pixels — independent of lesion size, so small lesions
    # get proportionally more context than large ones.
    context_px = math.ceil(min(H, W) * context_fraction)
    side = max(box_h, box_w) + 2 * context_px

    cy = (r_min + r_max) / 2
    cx = (c_min + c_max) / 2

    r_start, r_end, pad_top, pad_bottom = _compute_crop_1d(cy, side, H)
    c_start, c_end, pad_left, pad_right = _compute_crop_1d(cx, side, W)

    crop = image_arr[r_start:r_end, c_start:c_end]

    if pad_top or pad_bottom or pad_left or pad_right:
        pad_width = (
            ((pad_top, pad_bottom), (pad_left, pad_right))
            if image_arr.ndim == 2
            else ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        )
        kwargs = {"constant_values": pad_value} if pad_mode == "constant" else {}
        crop = np.pad(crop, pad_width, mode=pad_mode, **kwargs)

    return crop


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def build_train_transform(preprocess_val) -> transforms.Compose:
    """Build a custom training augmentation pipeline.

    Normalization mean/std are taken from preprocess_val so they always match
    the model, regardless of which checkpoint is loaded.

    Augmentations chosen for bone-tumour X-rays:
      - RandomResizedCrop: simulates varying patient positioning and zoom
      - RandomRotation(10°): small tilts from patient/table angle
      - RandomAdjustSharpness: varies image sharpness/contrast
      - GaussianBlur: simulates different acquisition sharpness
    Horizontal/vertical flips are intentionally omitted — left/right anatomy
    is clinically meaningful in radiographs.
    """
    # Extract normalization from the val pipeline (always correct for the model)
    norm = next(t for t in preprocess_val.transforms if isinstance(t, transforms.Normalize))

    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        norm,
    ])


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank adaptation.

    output = W(x) + (x @ A^T @ B^T) * (alpha / r)

    A is initialised with small Gaussian noise; B is initialised to zero so
    that the LoRA contribution is zero at the start of training.
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.linear = linear
        self.r = r
        self.scale = alpha / r

        in_features = linear.in_features
        out_features = linear.out_features
        device = linear.weight.device
        dtype = linear.weight.dtype

        self.lora_A = nn.Parameter(
            torch.randn(r, in_features, device=device, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, r, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


class LoRAFusedQKV(nn.Module):
    """Low-rank parametrization for fused QKV attention weights.

    PyTorch MultiheadAttention stores Q, K, and V in one parameter named
    ``in_proj_weight`` with shape (3 * d_model, d_model).  Registering this
    parametrization makes the effective weight:

        W_qkv + (B @ A) * (alpha / r)

    while keeping the original fused QKV weight frozen.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        r: int,
        alpha: float,
    ) -> None:
        super().__init__()
        out_features, in_features = weight.shape
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(
            torch.randn(r, in_features, device=weight.device, dtype=weight.dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, r, device=weight.device, dtype=weight.dtype)
        )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + (self.lora_B @ self.lora_A).view_as(weight) * self.scale


def inject_lora(
    model: nn.Module,
    lora_layers: int,
    r: int,
    alpha: float,
) -> None:
    """Freeze the whole model, then inject LoRA into the last N ViT blocks.

    BiomedCLIP uses a TimmModel wrapper, so the block path is:
        model.visual.trunk.blocks[i]

    Targets per block (timm ViT naming):
        block.attn.qkv   (fused QKV Linear: dim -> 3 * dim)
        block.attn.proj  (output projection: dim -> dim)
        block.mlp.fc1    (dim -> mlp_width)
        block.mlp.fc2    (mlp_width -> dim)

    logit_scale is unfrozen so the contrastive temperature can be learned.
    """
    # Step 1: freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Step 2: identify target blocks
    blocks = model.visual.trunk.blocks
    n_blocks = len(blocks)
    effective_layers = min(lora_layers, n_blocks)
    if effective_layers < lora_layers:
        warnings.warn(
            f"--lora_layers={lora_layers} exceeds total ViT blocks ({n_blocks}). "
            f"Applying LoRA to all {n_blocks} blocks.",
            stacklevel=2,
        )
    target_indices = range(n_blocks - effective_layers, n_blocks)

    # Step 3: replace linear layers in target blocks
    for i in target_indices:
        block = blocks[i]
        block.attn.qkv = LoRALinear(block.attn.qkv, r, alpha)
        block.attn.proj = LoRALinear(block.attn.proj, r, alpha)
        block.mlp.fc1 = LoRALinear(block.mlp.fc1, r, alpha)
        block.mlp.fc2 = LoRALinear(block.mlp.fc2, r, alpha)

    # Step 4: unfreeze learnable temperature
    model.logit_scale.requires_grad_(True)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def _load_all_samples(
    excel_path: Path,
    reports_path: Path,
    images_dir: Path,
    masks_dir: Path,
    english: bool = False,
) -> tuple[list, list]:
    """Build two independent sample pools with different requirements.

    Returns
    -------
    pretrain_cands : list of (image_path, mask_path, report)
        Every row that has an image file and a non-empty report.
        Label is not required — unlabeled images are valid pretrain samples.

    downstream_cands : list of (image_path, mask_path, report_or_None, label)
        Every row that has an image file, a malignancy label, and a parseable
        age + sex entry.  Report is included when available, else None.
    """

    # --- Load metadata: A=filename, C=malignancy, E=age, F=sex, H=patid ---
    try:
        df = pd.read_excel(
            excel_path,
            sheet_name="internal_data_matched",
            usecols=[0, 2, 4, 5, 7],  # A, C, E, F, H
            skiprows=1,
            header=0,
            dtype=str,
            engine="openpyxl",
        )
    except Exception as exc:
        raise RuntimeError(f"Cannot read Excel file '{excel_path}': {exc}") from exc

    df.columns = ["filename", "malignancy", "age", "sex", "patid"]
    df = df.dropna(subset=["filename", "patid"])
    df["filename"]   = df["filename"].str.strip()
    df["patid"]      = df["patid"].str.strip()
    df["malignancy"] = df["malignancy"].fillna("").str.strip().str.lower()
    df["age"]        = df["age"].fillna("")
    df["sex"]        = df["sex"].fillna("")

    def _normalise_id(val: str) -> str:
        try:
            return str(int(float(val)))
        except ValueError:
            return val

    df["patid"] = df["patid"].apply(_normalise_id)

    # --- Load reports ---
    with open(reports_path, encoding="utf-8") as fh:
        raw_reports = json.load(fh)

    report_lookup: dict[str, str] = {}
    for entry in raw_reports:
        pid = _normalise_id(str(entry.get("patid", "")).strip())
        if not pid:
            continue
        befund_key      = "befund_en"      if english else "befund"
        beurteilung_key = "beurteilung_en" if english else "beurteilung"
        befund      = (entry.get(befund_key) or "").strip()
        beurteilung = (entry.get(beurteilung_key) or "").strip()
        text = " ".join(filter(None, [befund, beurteilung]))
        if text:
            report_lookup[pid] = text

    # --- Build pools ---
    pretrain_cands:    list[tuple[Path, Path, str]]             = []
    downstream_cands:  list[tuple[Path, Path, str | None, str]] = []
    skipped_no_image   = 0
    skipped_no_age_sex = 0

    for _, row in df.iterrows():
        stem       = Path(row["filename"]).stem
        patid      = row["patid"]
        label      = row["malignancy"]
        image_path = images_dir / f"{stem}.png"
        mask_path  = masks_dir  / f"{stem}.png"

        if not image_path.exists():
            skipped_no_image += 1
            continue

        report = report_lookup.get(patid) or None

        # Pretrain candidate: image + report (label not required)
        if report:
            pretrain_cands.append((image_path, mask_path, report))

        # Downstream candidate: image + label + valid age/sex
        if label:
            try:
                float(row["age"])
            except (ValueError, TypeError):
                skipped_no_age_sex += 1
                continue
            if str(row["sex"]).strip().lower() not in ("m", "male", "1", "f", "female", "0"):
                skipped_no_age_sex += 1
                continue
            downstream_cands.append((image_path, mask_path, report, label))

    print(
        f"Dataset: {len(pretrain_cands)} pretrain candidates (image+report), "
        f"{len(downstream_cands)} downstream candidates (image+label+age+sex) "
        f"(skipped: {skipped_no_image} missing images, "
        f"{skipped_no_age_sex} missing/invalid age or sex)"
    )
    return pretrain_cands, downstream_cands


class BoneTumorPairDataset(Dataset):
    """(image tensor, text token tensor) pairs for contrastive pretraining."""

    def __init__(
        self,
        samples: list[tuple[Path, Path, str]],
        preprocess,
        tokenizer,
        use_mask: bool,
    ) -> None:
        self.samples = samples
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.use_mask = use_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path, mask_path, text = self.samples[idx]

        # Load image as RGB
        image = Image.open(image_path).convert("RGB")

        if self.use_mask and mask_path.exists():
            image_arr = np.array(image.convert("L"), dtype=float)
            mask_arr = np.array(Image.open(mask_path).convert("L"), dtype=float)
            cropped = crop_around_mask(image_arr, mask_arr)
            image = Image.fromarray(cropped.astype(np.uint8)).convert("RGB")

        image_tensor = self.preprocess(image)  # (3, 224, 224)

        # Tokenise (context_length=256 as per BiomedCLIP paper)
        text_tokens = self.tokenizer([text], context_length=256).squeeze(0)  # (256,)

        return {"image": image_tensor, "text": text_tokens}


def _stratified_split_two(
    samples: list,
    labels: list[str],
    split_size: float,
    seed: int,
) -> tuple[list, list]:
    """Split *samples* into two stratified groups.

    *split_size* is the fraction assigned to the second group.  Falls back to
    a random (non-stratified) split if any class has too few members.
    """
    try:
        a, b = train_test_split(
            samples, test_size=split_size, stratify=labels, random_state=seed
        )
    except ValueError:
        warnings.warn(
            "Stratified split failed (too few samples in some class). "
            "Falling back to random split.",
            stacklevel=3,
        )
        a, b = train_test_split(samples, test_size=split_size, random_state=seed)
    return a, b


def build_stratified_splits(
    args: argparse.Namespace,
    run_dir: Path | None = None,
) -> tuple[list, list, list, list]:
    """Load all samples and create independent pretrain and downstream splits.

    Returns
    -------
    pretrain_samples, downstream_train_samples, downstream_val_samples, test_samples

    Definitions
    -----------
    - pretrain_samples : list of (image_path, mask_path, report)
        All report-bearing images EXCEPT those whose image path appears in
        downstream_val or test.  No label required — unlabeled images are valid.

    - downstream_train_samples : list of (image_path, mask_path, report_or_None, label)
        Downstream candidates not assigned to val or test.  May overlap with
        pretrain_samples (same images can appear in both).

    - downstream_val_samples : list of (image_path, mask_path, report_or_None, label)
        Held-out val split.  Images here are excluded from pretrain.

    - test_samples : list of (image_path, mask_path, report_or_None, label)
        Final held-out test split.  Images here are excluded from pretrain.

    Notes
    -----
    Leakage is prevented by:
    - Downstream val/test images never appear in pretraining.
    - Downstream train images may appear in pretrain (the encoder sees them
      during contrastive learning, but without labels — no label leakage).

    The split manifest is written to:
    ``<out_dir>/biomedclip_pretrain/splits.json``
    """
    pretrain_cands, downstream_cands = _load_all_samples(
        excel_path=Path(args.excel),
        reports_path=Path(args.reports),
        images_dir=Path(args.images),
        masks_dir=Path(args.masks),
        english=args.english,
    )

    if not downstream_cands:
        raise RuntimeError(
            "No downstream candidates found (need image + label + age/sex). "
            "Check --excel, --images paths."
        )
    if not pretrain_cands:
        raise RuntimeError(
            "No pretrain candidates found (need image + report). "
            "Check --reports path."
        )

    total = args.downstream_train_frac + args.downstream_val_frac + args.test_frac
    if abs(total - 1.0) > 1e-4:
        raise ValueError(
            f"Split fractions must sum to 1.0 using "
            f"--downstream_train_frac + --downstream_val_frac + --test_frac, "
            f"got {total:.4f}."
        )

    # Separate downstream candidates by report availability.
    # No-report samples are useless for pretraining, so they should fill
    # val/test first — maximising the pretrain pool.
    no_report  = [s for s in downstream_cands if not (s[2] and str(s[2]).strip())]
    has_report = [s for s in downstream_cands if s[2] and str(s[2]).strip()]

    n_total       = len(downstream_cands)
    n_test        = max(1, round(n_total * args.test_frac))
    n_val         = max(1, round(n_total * args.downstream_val_frac))
    n_from_no_rep = min(len(no_report), n_val + n_test)
    n_supplement  = (n_val + n_test) - n_from_no_rep

    # Step 1: assign no-report samples to val/test pool (all or stratified subset)
    if n_from_no_rep == len(no_report):
        no_rep_pool, no_rep_train = no_report, []
    else:
        frac = n_from_no_rep / len(no_report)
        labels_nr = [s[3] for s in no_report]
        no_rep_train, no_rep_pool = _stratified_split_two(no_report, labels_nr, frac, args.seed)

    # Step 2: supplement from has-report samples only if no-report pool was insufficient
    if n_supplement > 0 and has_report:
        frac = n_supplement / len(has_report)
        labels_hr = [s[3] for s in has_report]
        has_rep_train, has_rep_pool = _stratified_split_two(has_report, labels_hr, frac, args.seed)
    else:
        has_rep_pool, has_rep_train = [], has_report

    # Step 3: split the combined val+test pool (stratified) into val and test
    val_test_pool = no_rep_pool + has_rep_pool
    labels_vt     = [s[3] for s in val_test_pool]
    test_frac_of_pool = n_test / len(val_test_pool)
    downstream_val, test = _stratified_split_two(val_test_pool, labels_vt, test_frac_of_pool, args.seed)

    downstream_train = no_rep_train + has_rep_train

    # Step 3: pretrain = all report-bearing images except val/test images.
    # Downstream train images may appear in pretrain — the encoder sees them
    # without labels, which is not label leakage.
    excluded_images = {s[0] for s in downstream_val} | {s[0] for s in test}
    pretrain = [s for s in pretrain_cands if s[0] not in excluded_images]

    if not pretrain:
        raise RuntimeError(
            "Pretrain set is empty after excluding val/test images."
        )

    # --- Print statistics ---
    from collections import Counter

    def _dist(split: list, label_idx: int) -> dict[str, int]:
        return dict(Counter(s[label_idx] for s in split))

    def _count_reports(split: list, report_idx: int) -> int:
        return sum(1 for s in split if s[report_idx] is not None and str(s[report_idx]).strip())

    downstream_splits = {
        "downstream_train": downstream_train,
        "downstream_val":   downstream_val,
        "test":             test,
    }
    print("\nDownstream split statistics (mutually exclusive):")
    for name, split in downstream_splits.items():
        dist = _dist(split, label_idx=3)
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
        n_reports = _count_reports(split, report_idx=2)
        print(
            f"  {name:<20} {len(split):>4} samples"
            f"  | reports: {n_reports:>4}"
            f"  | {dist_str}"
        )

    n_pretrain_with_label = sum(
        1 for s in pretrain if s[0] in {ds[0] for ds in downstream_train}
    )
    print(
        f"\nPretrain set ({len(pretrain)} samples, val/test excluded):"
        f"\n  {len(pretrain_cands) - len(pretrain)} val/test images removed"
        f"\n  {n_pretrain_with_label} samples overlap with downstream_train"
        f"\n  {len(pretrain) - n_pretrain_with_label} are unlabeled (report only)"
    )
    print()

    # --- Save split manifest to disk ---
    out_dir = run_dir if run_dir is not None else Path(args.out_dir) / "biomedclip_pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list] = {}

    # Pretrain entries are 3-tuples (image, mask, report) — no label field
    manifest["pretrain"] = [
        {"image": str(s[0]), "mask": str(s[1]), "report": s[2]}
        for s in pretrain
    ]
    # Downstream entries are 4-tuples (image, mask, report, label)
    for name, split in downstream_splits.items():
        manifest[name] = [
            {"image": str(s[0]), "mask": str(s[1]), "report": s[2], "label": s[3]}
            for s in split
        ]

    manifest_path = out_dir / "splits.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"  Split manifest saved -> {manifest_path}\n")

    return pretrain, downstream_train, downstream_val, test

def build_pretrain_datasets(
    pretrain_samples: list,
    preprocess_train,
    preprocess_val,
    tokenizer,
    use_mask: bool,
    seed: int,
    monitor_val_frac: float = 0.1,
) -> tuple[BoneTumorPairDataset, BoneTumorPairDataset]:
    """Split the pretraining samples into a training and a monitoring-val set.

    The monitoring split is a small random (non-stratified) held-out portion
    used only to track contrastive loss during pretraining — it is not a
    downstream evaluation split.
    """
    # pretrain_samples are already 3-tuples (image, mask, report)
    unlabeled = list(pretrain_samples)

    rng = random.Random(seed)
    indices = list(range(len(unlabeled)))
    rng.shuffle(indices)
    split = int(len(indices) * (1.0 - monitor_val_frac))
    train_samp = [unlabeled[i] for i in indices[:split]]
    val_samp = [unlabeled[i] for i in indices[split:]]

    print(f"Pretrain loop split: {len(train_samp)} train / {len(val_samp)} monitor-val")

    train_ds = BoneTumorPairDataset(train_samp, preprocess_train, tokenizer, use_mask)
    val_ds = BoneTumorPairDataset(val_samp, preprocess_val, tokenizer, use_mask)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def clip_loss(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP) loss.

    Both feature tensors are L2-normalised before computing logits.
    logit_scale is clamped to avoid numerical instability.
    """
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)

    scale = logit_scale.exp().clamp(max=100.0)
    logits_per_image = image_feat @ text_feat.t() * scale   # (B, B)
    logits_per_text = text_feat @ image_feat.t() * scale    # (B, B)

    labels = torch.arange(len(image_feat), device=image_feat.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2.0


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    trainable_params: list,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  train", leave=False, disable=not sys.stdout.isatty())
    for batch in pbar:
        images = batch["image"].to(device)
        texts = batch["text"].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            image_feat = model.encode_image(images)
            text_feat = model.encode_text(texts)
            loss = clip_loss(image_feat, text_feat, model.logit_scale)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        texts = batch["text"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            image_feat = model.encode_image(images)
            text_feat = model.encode_text(texts)
            loss = clip_loss(image_feat, text_feat, model.logit_scale)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_retrieval(
    model: nn.Module,
    val_samples: list,
    preprocess_val,
    tokenizer,
    device: torch.device,
    use_mask: bool,
    batch_size: int = 32,
) -> dict[str, float]:
    """Compute image-text retrieval metrics on the downstream val set.

    Only samples with non-empty reports are used (need paired image+text).
    Metrics are computed per mini-batch and averaged, matching how val loss
    is computed, so the two are directly comparable.
    Returns I2T/T2I Recall@1, Recall@5, and median rank.
    """
    paired = [(s[0], s[1], s[2]) for s in val_samples if s[2] and str(s[2]).strip()]
    if len(paired) < 2:
        return {}

    model.eval()
    ds = BoneTumorPairDataset(paired, preprocess_val, tokenizer, use_mask)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    i2t_r1_list, i2t_r5_list, i2t_med_list = [], [], []
    t2i_r1_list, t2i_r5_list, t2i_med_list = [], [], []
    n_pairs = 0

    for batch in loader:
        images = batch["image"].to(device)
        texts = batch["text"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            img_feat = F.normalize(model.encode_image(images), dim=-1)
            txt_feat = F.normalize(model.encode_text(texts), dim=-1)

        sim = img_feat.float() @ txt_feat.float().T  # (B, B)
        b = sim.shape[0]
        n_pairs += b

        def _metrics(sim_matrix: torch.Tensor) -> tuple[float, float, float]:
            ranks = np.array([
                int((sim_matrix[i] > sim_matrix[i, i]).sum().item()) + 1
                for i in range(b)
            ])
            return (ranks <= 1).mean(), (ranks <= 5).mean(), float(np.median(ranks))

        i2t_r1, i2t_r5, i2t_med = _metrics(sim)
        t2i_r1, t2i_r5, t2i_med = _metrics(sim.T)

        i2t_r1_list.append(i2t_r1)
        i2t_r5_list.append(i2t_r5)
        i2t_med_list.append(i2t_med)
        t2i_r1_list.append(t2i_r1)
        t2i_r5_list.append(t2i_r5)
        t2i_med_list.append(t2i_med)

    return {
        "retrieval/i2t_r1":          float(np.mean(i2t_r1_list)),
        "retrieval/i2t_r5":          float(np.mean(i2t_r5_list)),
        "retrieval/i2t_median_rank": float(np.mean(i2t_med_list)),
        "retrieval/t2i_r1":          float(np.mean(t2i_r1_list)),
        "retrieval/t2i_r5":          float(np.mean(t2i_r5_list)),
        "retrieval/t2i_median_rank": float(np.mean(t2i_med_list)),
        "retrieval/n_pairs":         float(n_pairs),
        "retrieval/batch_size":      float(batch_size),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: Path,
    lora_config: dict | None = None,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "lora_config": lora_config or {},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"  Saved checkpoint -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BiomedCLIP contrastive pretraining with LoRA on the ViT encoder."
    )
    parser.add_argument(
        "--excel",
        default=str(DEFAULT_EXCEL),
        help="Path to metadata.xlsx (default: %(default)s)",
    )
    parser.add_argument(
        "--reports",
        default=str(DEFAULT_REPORTS),
        help="Path to reports JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--english", action="store_true",
        help="Read befund_en/beurteilung_en instead of befund/beurteilung "
             "(use with translated_reports.json).",
    )
    parser.add_argument(
        "--images",
        default=str(DEFAULT_IMAGES_DIR),
        help="Directory containing image PNGs (default: %(default)s)",
    )
    parser.add_argument(
        "--masks",
        default=str(DEFAULT_MASKS_DIR),
        help="Directory containing segmentation mask PNGs (default: %(default)s)",
    )
    parser.add_argument(
        "--out_dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for checkpoints (default: %(default)s)",
    )
    parser.add_argument(
        "--print_architecture",
        action="store_true",
        help="Load BioMedCLIP, print image/text encoder architecture, then exit",
    )
    parser.add_argument(
        "--print_full_model",
        action="store_true",
        help="With --print_architecture, also print the full model wrapper",
    )
    parser.add_argument(
        "--lora_layers",
        type=int,
        default=4,
        help="Number of last ViT transformer blocks to apply LoRA to (default: %(default)s)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank r (default: %(default)s)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="LoRA alpha scaling factor (default: %(default)s)",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        help="Crop images around the lesion using segmentation masks",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Peak learning rate for AdamW (default: %(default)s)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.2,
        help="AdamW weight decay for non-norm parameters (default: %(default)s)",
    )
    parser.add_argument(
        "--pretrain_frac",
        type=float,
        default=0.6,
        help="Fraction of data used for contrastive pretraining (default: %(default)s)",
    )
    parser.add_argument(
        "--downstream_train_frac",
        type=float,
        default=0.2,
        help="Fraction of data for downstream classifier training (default: %(default)s)",
    )
    parser.add_argument(
        "--downstream_val_frac",
        type=float,
        default=0.1,
        help="Fraction of data for downstream classifier validation (default: %(default)s)",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.1,
        help="Fraction of data held out for final evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        default="biomedclip-pretrain",
        help="W&B project name (default: %(default)s)",
    )
    parser.add_argument(
        "--wandb_run",
        default=None,
        help="W&B run name (default: auto-generated)",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="W&B team/entity name (default: personal account)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run as wandb sweep agent (hyperparams come from wandb.config)",
    )
    return parser.parse_args(argv)


def _apply_sweep_config(args: argparse.Namespace) -> None:
    """Overwrite args with values from wandb.config when running as sweep agent."""
    cfg = wandb.config
    for key in ("lora_layers", "lora_r", "lora_alpha", "lr", "weight_decay", "batch_size"):
        if key in cfg:
            setattr(args, key, cfg[key])


def main(args: argparse.Namespace) -> None:
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.print_architecture:
        print_biomedclip_architecture(
            model_tag=MODEL_TAG,
            device=device,
            include_full_model=args.print_full_model,
        )
        return

    # --- Load BiomedCLIP ---
    print(f"Loading model: {MODEL_TAG}")
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)
    tokenizer = open_clip.get_tokenizer(MODEL_TAG)
    model = model.to(device)
    preprocess_train = build_train_transform(preprocess_val)

    # --- Inject LoRA ---
    inject_lora(model, args.lora_layers, args.lora_r, args.lora_alpha)
    n_trainable = count_trainable_params(model)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA injected into last {args.lora_layers} ViT blocks "
        f"(r={args.lora_r}, alpha={args.lora_alpha})"
    )
    print(
        f"Trainable params: {n_trainable:,} / {n_total:,} "
        f"({100 * n_trainable / n_total:.2f} %)"
    )

    # --- Run directory (timestamped) ---
    run_dir = Path(args.out_dir) / "biomedclip_pretrain" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # --- Stratified splits ---
    pretrain_samples, _, downstream_val, _ = build_stratified_splits(args, run_dir=run_dir)

    # --- Pretrain datasets (small monitor-val inside pretrain set) ---
    train_ds, val_ds = build_pretrain_datasets(
        pretrain_samples, preprocess_train, preprocess_val, tokenizer, args.use_mask, args.seed
    )

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory,
        drop_last=len(train_ds) > args.batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory,
    )

    # --- Optimizer and scheduler ---
    no_decay_suffixes = ("bias", "norm.weight", "norm.bias", "ln_1.weight", "ln_1.bias",
                         "ln_2.weight", "ln_2.bias", "ln_pre.weight", "ln_pre.bias",
                         "ln_post.weight", "ln_post.bias")
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(no_decay_suffixes):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    trainable_params = decay_params + no_decay_params
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    warmup_epochs = max(1, args.epochs // 5)
    scheduler = make_scheduler(optimizer, warmup_epochs, args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else torch.amp.GradScaler("cpu")

    # --- W&B ---
    use_wandb = (args.wandb or args.sweep) and WANDB_AVAILABLE
    if (args.wandb or args.sweep) and not WANDB_AVAILABLE:
        warnings.warn("--wandb/--sweep set but wandb is not installed. Skipping.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            config={
                "lora_layers": args.lora_layers,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "use_mask": args.use_mask,
            },
        )
        if args.sweep:
            _apply_sweep_config(args)

    # --- Training loop ---
    best_val_loss = float("inf")
    best_mean_r1 = 0.0
    lora_config = {"lora_layers": args.lora_layers, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha}
    print(f"\nStarting training for {args.epochs} epochs (warmup: {warmup_epochs})\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, trainable_params)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        lr_current = scheduler.get_last_lr()[0]
        logit_scale = model.logit_scale.item()
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={lr_current:.2e} | "
            f"logit_scale={logit_scale:.3f}"
        )
        retrieval = evaluate_retrieval(
            model, val_ds.samples, preprocess_val, tokenizer, device,
            args.use_mask, batch_size=args.batch_size,
        )
        if retrieval:
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
                f" | avg/bs={int(retrieval['retrieval/batch_size'])} n={int(retrieval['retrieval/n_pairs'])}"
            )
        if use_wandb:
            log_dict = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/lr": lr_current,
                "train/logit_scale": logit_scale,
            }
            log_dict.update(retrieval)
            wandb.log(log_dict, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                run_dir / "best_val_checkpoint.pt",
                lora_config=lora_config,
            )

        mean_r1 = retrieval.get("retrieval/mean_r1", 0.0)
        if mean_r1 > best_mean_r1:
            best_mean_r1 = mean_r1
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                run_dir / "best_r1_checkpoint.pt",
                lora_config=lora_config,
            )

    save_checkpoint(
        model, optimizer, args.epochs, val_loss,
        run_dir / "final_checkpoint.pt",
        lora_config=lora_config,
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} | Best mean R@1: {best_mean_r1:.1%}")
    print(f"Checkpoints saved to: {run_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
    #build_stratified_splits(parse_args())