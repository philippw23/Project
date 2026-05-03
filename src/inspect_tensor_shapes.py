"""Prints model architecture with output tensor shapes.

Supports:
  biomedclip  — BiomedCLIP ViT-B/16 image + PubMedBERT text encoder
  chexfound   — CheXFound ViT-L/16 image encoder (requires --config)

Strategy:
  1. Register a forward hook on every submodule to record its output shape.
  2. Run one dummy forward pass to trigger all hooks.
  3. Reconstruct PyTorch's repr() format from scratch, appending the recorded
     shape as a comment to each layer line.

Usage:
  python src/inspect_tensor_shapes.py --model biomedclip
  python src/inspect_tensor_shapes.py --model biomedclip --lora_layers 4
  python src/inspect_tensor_shapes.py --model chexfound \\
      --config src/chexfound/configs/chexfound_vitl16_bonetumor.yaml
  python src/inspect_tensor_shapes.py --model chexfound \\
      --config src/chexfound/configs/chexfound_vitl16_bonetumor.yaml \\
      --lora_layers 4
"""

import argparse

import torch
import open_clip

BIOMEDCLIP_TAG = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


# ── Step 1 & 2: collect output shapes via forward hooks ───────────────────────

def collect_shapes(module: torch.nn.Module, dummy_input: torch.Tensor) -> dict:
    """Run one forward pass and return {module_path: output_shape} for every submodule.

    'module_path' mirrors PyTorch's named_modules() convention, e.g.:
        ""                    → the root module itself
        "trunk"               → module.trunk
        "trunk.patch_embed"   → module.trunk.patch_embed
    """
    shapes = {}
    hooks = []

    for name, mod in module.named_modules():
        # make_hook is a factory that closes over `name` so each hook
        # captures its own path string (avoids the late-binding closure pitfall).
        def make_hook(n):
            def hook(m, inp, out):
                # HuggingFace models (e.g. BertModel) return a dataclass instead
                # of a plain tensor, so we extract the primary tensor explicitly.
                if hasattr(out, "last_hidden_state"):
                    t = out.last_hidden_state   # (batch, seq_len, hidden)
                elif isinstance(out, tuple):
                    t = out[0]                  # many PyTorch modules return tuples
                else:
                    t = out

                if isinstance(t, torch.Tensor):
                    shapes[n] = tuple(t.shape)  # store as plain tuple, e.g. (2, 197, 768)
            return hook

        hooks.append(mod.register_forward_hook(make_hook(name)))

    # A single forward pass fires all hooks simultaneously.
    with torch.no_grad():
        module(dummy_input)

    # Clean up: hooks keep references to modules, so always remove them.
    for h in hooks:
        h.remove()

    return shapes


# ── Step 3: reconstruct PyTorch's repr with shape annotations ─────────────────

def _addindent(s: str, n: int) -> str:
    """Add n spaces to every line except the first.

    This mirrors PyTorch's internal _addindent used by nn.Module.__repr__
    to indent child module strings when nesting them inside their parent.

    Example with n=2:
        "Block(\\n  (norm): LayerNorm()"  →  "Block(\\n    (norm): LayerNorm()"
    """
    lines = s.split("\n")
    if len(lines) == 1:
        return s                        # single-line string, nothing to indent
    first = lines.pop(0)               # first line is NOT indented (it's the opening)
    return first + "\n" + "\n".join(" " * n + line for line in lines)


def module_repr(module: torch.nn.Module, shapes: dict, path: str = "") -> str:
    """Recursively build a repr string for `module`, matching PyTorch's format
    but with '  # → shape' appended to every layer line.

    The recursion walks the module tree depth-first, exactly like PyTorch's own
    __repr__, so the indentation and nesting are identical to print(model).

    Args:
        module: The module to format.
        shapes: Dict mapping module path → output shape (from collect_shapes).
        path:   Dot-separated path of this module, e.g. "trunk.patch_embed".
                Empty string "" for the root module.
    """
    # extra_repr() holds the parameters shown inline, e.g. "768, eps=1e-06"
    # for LayerNorm or "in_features=768, out_features=512" for Linear.
    extra_lines = module.extra_repr().split("\n") if module.extra_repr() else []

    # Recurse into all named children (direct submodules only, not grandchildren).
    child_lines = []
    for key, child in module._modules.items():
        if child is None:
            continue
        # Build the child's full dot-path, e.g. "trunk" + "." + "patch_embed"
        child_path = f"{path}.{key}" if path else key
        child_str = module_repr(child, shapes, child_path)
        # Indent all lines of the child string (except the first) by 2 spaces
        # so nested modules align correctly.
        child_str = _addindent(child_str, 2)
        child_lines.append(f"({key}): {child_str}")

    # Append the recorded output shape as a comment if we have one.
    shape_str = f"  # -> {shapes[path]}" if path in shapes else ""

    lines = extra_lines + child_lines
    name = type(module).__name__

    # ── Three cases that match PyTorch's repr logic ──────────────────────────

    # Case 1: no extra_repr and no children → one-liner, e.g. "Identity()  # -> (2, 768)"
    if not lines:
        return f"{name}(){shape_str}"

    # Case 2: only extra_repr, no children → one-liner with params,
    #         e.g. "LayerNorm(768, eps=1e-06)  # -> (2, 197, 768)"
    if len(extra_lines) == 1 and not child_lines:
        return f"{name}({extra_lines[0]}){shape_str}"

    # Case 3: has children → multi-line block.
    # The shape goes on the opening line so it's immediately visible, e.g.:
    #   PatchEmbed(  # -> (2, 196, 768)
    #     (proj): Conv2d(...)  # -> (2, 196, 768)
    #     (norm): Identity()   # -> (2, 196, 768)
    #   )
    return f"{name}({shape_str}\n  " + "\n  ".join(lines) + "\n)"


# ── Entry point ────────────────────────────────────────────────────────────────

def _inspect_biomedclip(device: torch.device, lora_layers: int = 0, lora_r: int = 8, lora_alpha: float = 16.0) -> None:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
    from biomedclip.models.lora import inject_lora, count_trainable_params

    model, _, _ = open_clip.create_model_and_transforms(BIOMEDCLIP_TAG)
    if lora_layers > 0:
        inject_lora(model, lora_layers, lora_r, lora_alpha)
        n_trainable = count_trainable_params(model)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"LoRA injected into last {lora_layers} ViT blocks (r={lora_r}, alpha={lora_alpha})")
        print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f} %)\n")
    model = model.to(device).eval()

    dummy_image = torch.zeros(2, 3, 224, 224, device=device)
    image_shapes = collect_shapes(model.visual, dummy_image)

    title = "BiomedCLIP -- Image encoder with output tensor shapes (batch=2)"
    if lora_layers > 0:
        title += f" [LoRA last {lora_layers} blocks]"
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(module_repr(model.visual, image_shapes))
    print("=" * 80)

    print()

    # Context length 256 comes from the model tag: PubMedBERT_256.
    context_length = getattr(model, "context_length", 256)
    dummy_text = torch.ones(2, context_length, dtype=torch.long, device=device)
    text_shapes = collect_shapes(model.text, dummy_text)

    print("=" * 80)
    print(f"BiomedCLIP -- Text encoder with output tensor shapes (batch=2, seq_len={context_length})")
    print("=" * 80)
    print(module_repr(model.text, text_shapes))
    print("=" * 80)


def _inspect_chexfound(
    device: torch.device,
    config: str,
    weights: str,
    lora_layers: int = 0,
    base_cfg: str | None = None,
) -> None:
    import yaml as _yaml
    from pathlib import Path as _Path
    from chexfound.models.encoders import CheXFoundViT
    from chexfound.models.lora import inject_lora_chexfound
    from chexfound.layers import DINOHead

    # ── Single checkpoint load ────────────────────────────────────────────────
    _teacher_sd = None
    if weights and _Path(weights).is_file():
        _raw = torch.load(weights, map_location="cpu", weights_only=False)
        _teacher_sd = _raw.get("teacher", _raw)

    # ── Merge base config + override to get head dimensions ──────────────────
    _cfg: dict = {}
    _default_base = _Path(__file__).parent / "chexfound/data/config.yaml"
    _base_path = _Path(base_cfg) if base_cfg else _default_base
    if _base_path.is_file():
        with open(_base_path) as _fh:
            _cfg = _yaml.safe_load(_fh) or {}
    with open(config) as _fh:
        _override = _yaml.safe_load(_fh) or {}
    for _sec, _val in _override.items():
        if isinstance(_val, dict) and isinstance(_cfg.get(_sec), dict):
            _cfg[_sec] = {**_cfg[_sec], **_val}
        else:
            _cfg[_sec] = _val

    # ── Build backbone (no internal load, no LoRA yet) ───────────────────────
    # load_pretrained=False avoids a second torch.load inside CheXFoundViT.
    # lora_layers=0 keeps weight key names clean so the manual load below matches.
    encoder = CheXFoundViT(config, weights_path=None,
                           lora_layers=0, r=8, alpha=16.0,
                           load_pretrained=False)
    if _teacher_sd is not None:
        _backbone_sd = {
            (k[len("backbone."):] if k.startswith("backbone.") else k): v
            for k, v in _teacher_sd.items()
            if not k.startswith(("dino_head.", "ibot_head."))
        }
        _msg = encoder.trunk.load_state_dict(_backbone_sd, strict=False)
        print(f"Backbone: loaded {len(_backbone_sd) - len(_msg.missing_keys)}"
              f"/{len(_backbone_sd)} keys "
              f"(missing={len(_msg.missing_keys)}, unexpected={len(_msg.unexpected_keys)})")
    # Inject LoRA AFTER backbone weights are loaded so key names still match.
    if lora_layers > 0:
        inject_lora_chexfound(encoder.trunk, lora_layers, r=8, alpha=16.0)
    encoder = encoder.to(device).eval()

    # ── Backbone inspection ──────────────────────────────────────────────────
    # Use the training resolution (512×512) so shapes reflect the actual run.
    global_crops_size = _cfg.get("crops", {}).get("global_crops_size", 512)
    dummy_image = torch.zeros(2, 3, global_crops_size, global_crops_size, device=device)
    image_shapes = collect_shapes(encoder.trunk, dummy_image)

    print("=" * 80)
    title = f"CheXFound ViT-L/16 backbone -- output tensor shapes (batch=2, {global_crops_size}×{global_crops_size})"
    if lora_layers > 0:
        title += f" [LoRA last {lora_layers} blocks]"
    print(title)
    print("=" * 80)
    print(module_repr(encoder.trunk, image_shapes))
    print("=" * 80)
    print()

    # ── Build heads from merged config ───────────────────────────────────────
    _dino_cfg = _cfg.get("dino", {})
    _ibot_cfg  = _cfg.get("ibot",  {})
    dino_head = DINOHead(
        in_dim=1024,
        out_dim=_dino_cfg.get("head_n_prototypes", 131072),
        hidden_dim=_dino_cfg.get("head_hidden_dim", 2048),
        bottleneck_dim=_dino_cfg.get("head_bottleneck_dim", 384),
        nlayers=_dino_cfg.get("head_nlayers", 3),
    )
    ibot_head = DINOHead(
        in_dim=1024,
        out_dim=_ibot_cfg.get("head_n_prototypes", 131072),
        hidden_dim=_ibot_cfg.get("head_hidden_dim", 2048),
        bottleneck_dim=_ibot_cfg.get("head_bottleneck_dim", 256),
        nlayers=_ibot_cfg.get("head_nlayers", 3),
    )
    if _teacher_sd is not None:
        _dino_sd = {k[len("dino_head."):]: v
                    for k, v in _teacher_sd.items() if k.startswith("dino_head.")}
        _ibot_sd = {k[len("ibot_head."):]: v
                    for k, v in _teacher_sd.items() if k.startswith("ibot_head.")}
        if _dino_sd:
            dino_head.load_state_dict(_dino_sd, strict=True)
            print(f"DINO head: loaded {len(_dino_sd)} keys from checkpoint")
        if _ibot_sd:
            ibot_head.load_state_dict(_ibot_sd, strict=True)
            print(f"iBOT head: loaded {len(_ibot_sd)} keys from checkpoint")
    dino_head = dino_head.to(device).eval()
    ibot_head = ibot_head.to(device).eval()

    # ── Head inspections ─────────────────────────────────────────────────────
    dummy_embed = torch.zeros(2, 1024, device=device)

    dino_shapes = collect_shapes(dino_head, dummy_embed)
    print("=" * 80)
    print("CheXFound DINO head -- output tensor shapes (batch=2, embed_dim=1024)")
    print("=" * 80)
    print(module_repr(dino_head, dino_shapes))
    print("=" * 80)
    print()

    ibot_shapes = collect_shapes(ibot_head, dummy_embed)
    print("=" * 80)
    print("CheXFound iBOT head -- output tensor shapes (batch=2, embed_dim=1024)")
    print("=" * 80)
    print(module_repr(ibot_head, ibot_shapes))
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model architecture with tensor shapes.")
    parser.add_argument(
        "--model", choices=["biomedclip", "chexfound"], default="biomedclip",
        help="Which model to inspect (default: biomedclip).",
    )
    parser.add_argument("--config",   default=None, help="CheXFound model config YAML (chexfound only).")
    parser.add_argument("--base_cfg", default=None, help="Base YAML to merge beneath --config for head dimensions (chexfound only).")
    parser.add_argument("--weights",  default=None, help="Optional CheXFound checkpoint to load before inspection.")
    parser.add_argument("--lora_layers", type=int, default=0, help="Inject LoRA into the last N ViT blocks (0 = no LoRA).")
    parser.add_argument("--lora_r",     type=int,   default=8,    help="LoRA rank r (default: 8).")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha scaling factor (default: 16.0).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    if args.model == "biomedclip":
        _inspect_biomedclip(device, args.lora_layers, args.lora_r, args.lora_alpha)
    else:
        if not args.config:
            parser.error("--config is required for --model chexfound")
        _inspect_chexfound(device, args.config, args.weights, args.lora_layers, base_cfg=args.base_cfg)


if __name__ == "__main__":
    main()
