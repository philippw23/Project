"""HuggingFace model loader with optional 4-bit quantization.

Centralises model loading so both extraction pipelines (joint and separated)
share the same VRAM / dtype logic without duplication.
"""

import os
import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"


def load_model(model_id: str, quantize: bool, quantize_8bit: bool = False):
    """Load a causal LM and its tokenizer from HuggingFace.

    Automatically selects the best precision for the available hardware:
    - CUDA + quantize_8bit → 8-bit via bitsandbytes (~14 GB VRAM for 14B models)
    - CUDA + quantize      → 4-bit NF4 via bitsandbytes (~8 GB VRAM for 14B models)
    - CUDA, no quantize    → float16 (~28 GB VRAM for 14B models)
    - CPU only             → float32 (slow; quantize flag is silently ignored)

    Parameters
    ----------
    model_id : HuggingFace model ID, e.g. ``"Qwen/Qwen2.5-7B-Instruct"``
    quantize : if True, load in 4-bit (requires bitsandbytes + CUDA)

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        Model is set to eval mode and placed on the device chosen by device_map="auto".
    """
    import warnings
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    print(f"\nLoading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Visible GPUs: {n_gpus}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory // 1024**3} GB")
    load_kwargs: dict = {"device_map": "auto"}

    if not torch.cuda.is_available() and (quantize or quantize_8bit):
        print("WARNING: --quantize requires CUDA. Falling back to CPU fp32.")
        load_kwargs = {}
    elif quantize_8bit:
        print("Loading in 8-bit quantization (bitsandbytes)…")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize:
        print("Loading in 4-bit quantization (bitsandbytes)…")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif torch.cuda.is_available():
        load_kwargs["dtype"] = torch.float16
    else:
        print("No CUDA found — running on CPU (slow, consider --quantize on GPU).")
        load_kwargs["dtype"] = torch.float32

    print(f"Loading model: {model_id}  (this may take a few minutes on first run)…")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on: {device}\n")
    return model, tokenizer
