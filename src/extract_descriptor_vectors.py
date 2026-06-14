"""Extract 21-dimensional binary descriptor vectors from bone tumor radiology reports.

Runs Qwen2.5-7B-Instruct over every dataset sample that has a non-null report,
combining befund_en (findings) + beurteilung_en (assessment) for maximum signal,
falling back to the full report field. The relaxed prompt applies tumor-type
inference rules so post-op reports still produce informative vectors.

On LLM parse failure or invalid output, a zero vector is written with
"extraction_failed": true so every processed sample has an entry.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import torch
from tqdm import tqdm

from qwen_llm_extractor.models.loader import DEFAULT_MODEL, load_model
from qwen_llm_extractor.prompts.targets import SYSTEM_PROMPT, USER_PROMPT
from qwen_llm_extractor.utils.json_repair import _parse_response

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT_DIR / "data" / "internal_dataset" / "dataset_full.json"
DEFAULT_OUTPUT  = ROOT_DIR / "data" / "internal_dataset" / "text" / "descriptor_vectorsv2.json"

DESCRIPTOR_KEYS = [
    "sclerotic_margin", "geographic_border", "permeative_pattern",
    "cortical_destruction", "endosteal_scalloping", "periosteal_reaction",
    "sunburst_periosteal", "codman_triangle", "lamellar_periosteal",
    "osteolytic", "osteoblastic", "mixed_lytic_blastic", "chondroid_matrix",
    "ossified_matrix", "expansile", "soft_tissue_extension",
    "epiphyseal_involvement", "diaphyseal_location", "metaphyseal_location",
    "pathological_fracture", "bone_remodeling",
]

ZERO_LIST = [0] * len(DESCRIPTOR_KEYS)


def _query_llm(text: str, model, tokenizer, max_new_tokens: int = 256) -> tuple[dict, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT.format(findings_text=text)},
    ]
    chat_out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    )
    input_ids = (
        chat_out["input_ids"] if hasattr(chat_out, "input_ids") else chat_out
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
    ).strip()
    return _parse_response(raw), raw


def _fix_keys(parsed: dict) -> dict:
    """Remap any misspelled keys to the closest expected descriptor key."""
    fixed = {}
    for k, v in parsed.items():
        if k in DESCRIPTOR_KEYS:
            fixed[k] = v
        else:
            matches = difflib.get_close_matches(k, DESCRIPTOR_KEYS, n=1, cutoff=0.8)
            if matches:
                fixed[matches[0]] = v
    return fixed


def _validate(parsed: dict) -> bool:
    if "error" in parsed:
        return False
    if set(parsed.keys()) != set(DESCRIPTOR_KEYS):
        return False
    return all(parsed[k] in (0, 1) for k in DESCRIPTOR_KEYS)


def main(args: argparse.Namespace) -> None:
    with open(args.dataset, encoding="utf-8") as fh:
        dataset = json.load(fh)

    samples = [s for s in dataset if s.get("report") is not None]
    print(f"Total samples: {len(dataset)}, with non-null report: {len(samples)}")

    if args.max:
        samples = samples[: args.max]
        print(f"Limited to {args.max} samples.")

    model, tokenizer = load_model(args.model, quantize=args.quantize)

    results: dict[str, dict] = {}
    n_failed = 0

    for sample in tqdm(samples, desc="Extracting descriptor vectors"):
        image_path = sample["image"]
        # Combine findings + assessment for maximum signal; fall back to report
        parts = []
        if sample.get("befund_en"):
            parts.append(sample["befund_en"])
        if sample.get("beurteilung_en"):
            parts.append(sample["beurteilung_en"])
        text = "\n\n".join(parts) if parts else sample.get("report", "")

        parsed, raw = _query_llm(text, model, tokenizer, args.max_new_tokens)
        parsed = _fix_keys(parsed)

        if _validate(parsed):
            results[image_path] = [parsed[k] for k in DESCRIPTOR_KEYS]
        else:
            results[image_path] = raw
            n_failed += 1
            tqdm.write(
                f"  [failed] {image_path} — "
                f"{str(parsed.get('error', 'invalid output'))[:80]}"
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    n_ok = len(results) - n_failed
    print(f"\nDone. {n_ok}/{len(results)} successful, {n_failed} failed.")
    print(f"Saved → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 21-dim binary descriptor vectors from bone tumor reports."
    )
    parser.add_argument("--dataset",        default=str(DEFAULT_DATASET))
    parser.add_argument("--output",         default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--quantize",       action="store_true",
                        help="Load model in 4-bit (requires bitsandbytes + CUDA).")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate per sample (default: 256).")
    parser.add_argument("--max",            type=int, default=0,
                        help="Limit to first N samples (0 = all).")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
