"""Checkpoint loading utilities for CheXFound / DINOv2 models."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def load_pretrained_weights(
    model: nn.Module,
    pretrained_weights: str | Path,
    checkpoint_key: str | None = None,
) -> None:
    """Load weights from a CheXFound / DINOv2-style checkpoint into *model*.

    The function handles the three most common checkpoint layouts:
      1. Raw state dict  (keys map directly onto the model)
      2. Nested dict     (checkpoint_key selects the relevant sub-dict)
      3. "module." / "backbone." prefixed keys  (stripped automatically)

    Loads with strict=False so that projection-head keys present in the
    checkpoint but absent from the trunk are silently ignored.
    """
    path = Path(pretrained_weights)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state_dict = torch.load(path, map_location="cpu", weights_only=False)

    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"  Extracting key '{checkpoint_key}' from checkpoint.")
        state_dict = state_dict[checkpoint_key]

    # Strip common wrapper prefixes added by DistributedDataParallel or
    # the teacher/student backbone wrapper in iBOT.
    for prefix in ("module.", "backbone."):
        state_dict = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }

    # Continued-pretrain checkpoints store blocks in a staged layout:
    #   blocks.{stage}.{global_block_idx}.{rest}
    # The model expects a flat layout:
    #   blocks.{global_block_idx}.{rest}
    # Strip the stage level by dropping the outer index entirely.
    # Also remap SwiGLU MLP weight names: w12->fc1, w3->fc2.
    import re
    _staged = re.compile(r'^blocks\.\d+\.(\d+)\.(.+)$')
    remapped = {}
    for k, v in state_dict.items():
        m = _staged.match(k)
        if m:
            new_key = f"blocks.{m.group(1)}.{m.group(2)}"
            new_key = new_key.replace(".mlp.w12.", ".mlp.fc1.").replace(".mlp.w3.", ".mlp.fc2.")
            remapped[new_key] = v
        else:
            remapped[k] = v
    state_dict = remapped

    msg = model.load_state_dict(state_dict, strict=False)

    n_loaded = len(state_dict) - len(msg.missing_keys)
    print(
        f"  Loaded {n_loaded}/{len(state_dict)} tensors from {path.name} "
        f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})"
    )
    if msg.missing_keys:
        print(f"  Missing  : {msg.missing_keys[:5]}{'...' if len(msg.missing_keys) > 5 else ''}")
    if msg.unexpected_keys:
        print(f"  Unexpected: {msg.unexpected_keys[:5]}{'...' if len(msg.unexpected_keys) > 5 else ''}")
