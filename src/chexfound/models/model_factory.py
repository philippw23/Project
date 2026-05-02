"""Factory: build a CheXFound ViT trunk from a YAML config."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Tuple

import yaml

from . import vision_transformer as vits

try:
    from omegaconf import DictConfig as _DictConfig, OmegaConf as _OmegaConf
    def _to_dict(c: Any) -> Any:
        return _OmegaConf.to_container(c, resolve=True) if isinstance(c, _DictConfig) else c
except ImportError:
    def _to_dict(c: Any) -> Any:  # type: ignore[misc]
        return c

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
    only_teacher: bool = False,
) -> Tuple[object, object, int]:
    """Parse a CheXFound YAML config and return (student, teacher, embed_dim).

    Args:
        config:       Path to a model config YAML, already-merged config dict,
                      or an OmegaConf DictConfig object.
        only_teacher: Ignored — kept for call-site compatibility with encoders.py.

    Returns:
        student:    VisionTransformer instance (randomly initialised).
        teacher:    Deep copy of student (identical architecture/init weights).
        embed_dim:  Embedding dimension (1024 for ViT-L).
    """
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = _to_dict(config)

    student_cfg: dict = cfg.get("student", {})
    crops_cfg:   dict = cfg.get("crops",   {})

    # Merge defaults for keys that may be absent.
    merged = {**_DEFAULTS, **student_cfg}

    arch = merged["arch"]                            # e.g. "vit_large"
    img_size = crops_cfg.get("global_crops_size", _DEFAULT_IMG_SIZE)

    student = vits.__dict__[arch](
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
    teacher = copy.deepcopy(student)

    return student, teacher, student.embed_dim
