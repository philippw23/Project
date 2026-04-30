"""Factory: build a CheXFound ViT trunk from a YAML config."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import yaml

from . import vision_transformer as vits

# Keys in the YAML that drive the student (and by extension teacher) architecture.
_STUDENT_ARCH_KEYS = {
    "arch", "patch_size", "drop_path_rate", "layerscale",
    "drop_path_uniform", "ffn_layer", "block_chunks",
    "qkv_bias", "proj_bias", "ffn_bias",
    "num_register_tokens", "interpolate_antialias", "interpolate_offset",
}

# Default values for optional keys that may be absent in older configs.
_DEFAULTS: dict = {
    "layerscale":           1e-5,
    "drop_path_uniform":    True,
    "qkv_bias":             True,
    "proj_bias":            True,
    "ffn_bias":             True,
    "num_register_tokens":  0,
    "interpolate_antialias": False,
    "interpolate_offset":   0.1,
    "block_chunks":         0,
}

# Crops config supplies the training image size; used to set img_size on the
# model so its default positional embedding matches the stored checkpoint.
_DEFAULT_IMG_SIZE = 518


def build_model_from_cfg(
    config: str | Path | dict[str, Any],
    only_teacher: bool = True,
) -> Tuple[object, int]:
    """Parse a CheXFound YAML config and return (trunk, embed_dim).

    Args:
        config:  Path to a model config YAML, or an already-merged config dict.
        only_teacher: Kept for API compatibility — always returns a single trunk.

    Returns:
        trunk:      VisionTransformer instance (randomly initialised, call
                    load_pretrained_weights() separately to load checkpoint).
        embed_dim:  Embedding dimension (1024 for ViT-L).
    """
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = config

    student_cfg: dict = cfg.get("student", {})
    crops_cfg:   dict = cfg.get("crops",   {})

    # Merge defaults for keys that may be absent.
    merged = {**_DEFAULTS, **student_cfg}

    arch = merged["arch"]                            # e.g. "vit_large"
    img_size = crops_cfg.get("global_crops_size", _DEFAULT_IMG_SIZE)

    trunk = vits.__dict__[arch](
        img_size=img_size,
        patch_size=merged["patch_size"],
        init_values=merged["layerscale"],
        ffn_layer=merged["ffn_layer"],
        block_chunks=merged["block_chunks"],
        drop_path_rate=merged["drop_path_rate"],
        drop_path_uniform=merged["drop_path_uniform"],
        qkv_bias=merged["qkv_bias"],
        proj_bias=merged["proj_bias"],
        ffn_bias=merged["ffn_bias"],
        num_register_tokens=merged["num_register_tokens"],
        interpolate_antialias=merged["interpolate_antialias"],
        interpolate_offset=merged["interpolate_offset"],
    )

    return trunk, trunk.embed_dim
