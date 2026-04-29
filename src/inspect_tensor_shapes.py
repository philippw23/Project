"""Prints the BiomedCLIP image and text encoder architectures with output tensor shapes.

Strategy:
  1. Register a forward hook on every submodule to record its output shape.
  2. Run one dummy forward pass to trigger all hooks.
  3. Reconstruct PyTorch's repr() format from scratch, appending the recorded
     shape as a comment to each layer line.
"""

import torch
import open_clip

MODEL_TAG = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


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
    shape_str = f"  # → {shapes[path]}" if path in shapes else ""

    lines = extra_lines + child_lines
    name = type(module).__name__

    # ── Three cases that match PyTorch's repr logic ──────────────────────────

    # Case 1: no extra_repr and no children → one-liner, e.g. "Identity()  # → (2, 768)"
    if not lines:
        return f"{name}(){shape_str}"

    # Case 2: only extra_repr, no children → one-liner with params,
    #         e.g. "LayerNorm(768, eps=1e-06)  # → (2, 197, 768)"
    if len(extra_lines) == 1 and not child_lines:
        return f"{name}({extra_lines[0]}){shape_str}"

    # Case 3: has children → multi-line block.
    # The shape goes on the opening line so it's immediately visible, e.g.:
    #   PatchEmbed(  # → (2, 196, 768)
    #     (proj): Conv2d(...)  # → (2, 196, 768)
    #     (norm): Identity()   # → (2, 196, 768)
    #   )
    return f"{name}({shape_str}\n  " + "\n  ".join(lines) + "\n)"


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model, _, _ = open_clip.create_model_and_transforms(MODEL_TAG)
    model = model.to(device).eval()

    # ── Image encoder ─────────────────────────────────────────────────────────
    # Shape: (batch, channels, height, width) — standard image tensor.
    dummy_image = torch.zeros(2, 3, 224, 224, device=device)
    image_shapes = collect_shapes(model.visual, dummy_image)

    print("=" * 80)
    print("Image encoder with output tensor shapes (batch=2)")
    print("=" * 80)
    print(module_repr(model.visual, image_shapes))
    print("=" * 80)

    print()

    # ── Text encoder ──────────────────────────────────────────────────────────
    # Shape: (batch, seq_len) — integer token IDs, not embeddings yet.
    # Context length 256 comes from the model tag: PubMedBERT_256.
    context_length = getattr(model, "context_length", 256)
    dummy_text = torch.ones(2, context_length, dtype=torch.long, device=device)
    text_shapes = collect_shapes(model.text, dummy_text)

    print("=" * 80)
    print(f"Text encoder with output tensor shapes (batch=2, seq_len={context_length})")
    print("=" * 80)
    print(module_repr(model.text, text_shapes))
    print("=" * 80)


if __name__ == "__main__":
    main()
