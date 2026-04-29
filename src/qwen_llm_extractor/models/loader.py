"""HuggingFace model loader with optional 4-bit quantization.

Centralises model loading so both extraction pipelines (joint and separated)
share the same VRAM / dtype logic without duplication.
"""

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_model(model_id: str, quantize: bool):
    """Load a causal LM and its tokenizer from HuggingFace.

    Automatically selects the best precision for the available hardware:
    - CUDA + quantize  → 4-bit NF4 via bitsandbytes (~5 GB VRAM for 7B models)
    - CUDA, no quantize → float16 (~15 GB VRAM for 7B models)
    - CPU only         → float32 (slow; quantize flag is silently ignored)

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

    load_kwargs: dict = {"device_map": "auto"}

    if quantize:
        if not torch.cuda.is_available():
            print("WARNING: --quantize requires CUDA. Falling back to CPU fp32.")
            load_kwargs = {}
        else:
            print("Loading in 4-bit quantization (bitsandbytes)…")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # nested quantization, slightly lower VRAM
                bnb_4bit_quant_type="nf4",       # NormalFloat4: optimal for normally-distributed weights
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
