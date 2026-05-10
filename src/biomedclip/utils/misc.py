from __future__ import annotations

from pathlib import Path

import open_clip
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[3]

MODEL_TAG          = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEFAULT_IMAGES_DIR = ROOT_DIR / "data" / "internal_dataset" / "images"
DEFAULT_MASKS_DIR  = ROOT_DIR / "data" / "internal_dataset" / "segmentations"
DEFAULT_EXCEL      = ROOT_DIR / "data" / "internal_dataset" / "metadata.xlsx"
DEFAULT_REPORTS    = ROOT_DIR / "data" / "internal_dataset" / "text" / "translated_reports.json"
DEFAULT_OUT_DIR    = ROOT_DIR / "results"


def _count_parameters(module: nn.Module) -> tuple[int, int]:
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_biomedclip_architecture(
    model_tag: str = MODEL_TAG,
    device: torch.device | str | None = None,
    include_full_model: bool = False,
) -> None:
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
        "token_embedding", "positional_embedding", "transformer", "ln_final", "text_projection",
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
