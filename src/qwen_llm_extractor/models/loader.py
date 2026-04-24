import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_model(model_id: str, quantize: bool):
    """Load a causal LM from HuggingFace.

    Parameters
    ----------
    model_id  : HuggingFace model ID
    quantize  : load in 4-bit (requires bitsandbytes + CUDA) to reduce VRAM
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
