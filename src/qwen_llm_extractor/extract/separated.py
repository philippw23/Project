import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from qwen_llm_extractor.data.reports import load_reports_separated
from qwen_llm_extractor.eval.analysis import flatten_to_dataframe, print_summary, compare_with_medbert
from qwen_llm_extractor.models.loader import DEFAULT_MODEL, load_model
from qwen_llm_extractor.prompts.separated import (
    SYSTEM_PROMPT,
    BEFUND_PROMPT_TEMPLATE,
    BEURTEILUNG_PROMPT_TEMPLATE,
    BEURTEILUNG_SUMMARY_PROMPT_TEMPLATE,
)
from qwen_llm_extractor.utils.json_repair import _parse_response


def query_llm(
    text: str,
    model,
    tokenizer,
    prompt_template: str,
    max_new_tokens: int = 512,
) -> tuple[dict, str]:
    """Run one text through the LLM using the given prompt template and return (parsed_json, raw_text)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt_template.format(text=text)},
    ]

    chat_out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
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

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _parse_response(raw), raw


def main(args: argparse.Namespace) -> None:
    befunds, beurteilungs, patids = load_reports_separated(args.reports)
    if args.max:
        befunds      = befunds[: args.max]
        beurteilungs = beurteilungs[: args.max]
        patids       = patids[: args.max]
        print(f"Limited to {len(befunds)} reports.")

    model, tokenizer = load_model(args.model, quantize=args.quantize)

    results: list[dict] = []
    raw_responses: list[dict] = []
    for befund, beurteilung, patid in tqdm(
        zip(befunds, beurteilungs, patids), desc="LLM inference", total=len(befunds)
    ):
        befund_result,      befund_raw      = query_llm(befund,      model, tokenizer, BEFUND_PROMPT_TEMPLATE,      args.max_new_tokens)
        beurteilung_result, beurteilung_raw = query_llm(beurteilung, model, tokenizer, BEURTEILUNG_PROMPT_TEMPLATE, args.max_new_tokens)

        beurteilung_phrases = beurteilung_result.get("beurteilung_phrases", [])
        summary_raw = None
        if not beurteilung_phrases:
            beurteilung_result, summary_raw = query_llm(befund, model, tokenizer, BEURTEILUNG_SUMMARY_PROMPT_TEMPLATE, args.max_new_tokens)
            beurteilung_phrases = beurteilung_result.get("beurteilung_phrases", [])

        result = {
            "befund_phrases":      befund_result.get("befund_phrases", []),
            "beurteilung_phrases": beurteilung_phrases,
            "patid":               patid,
        }
        raw_responses.append({
            "patid":            patid,
            "befund_raw":       befund_raw,
            "beurteilung_raw":  beurteilung_raw,
            "summary_raw":      summary_raw,
        })
        for label, r in [("befund", befund_result), ("beurteilung", beurteilung_result)]:
            if "error" in r:
                tqdm.write(f"  [error:{label}] patid={patid} — {r['error'][:80]}")
        results.append(result)

    df = flatten_to_dataframe(results, patids)
    print_summary(df)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "llm_extracted_terms.csv", index=False, encoding="utf-8-sig")
    with open(out / "llm_raw_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(out / "llm_raw_responses.json", "w", encoding="utf-8") as f:
        json.dump(raw_responses, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:")
    print(f"  {out}/llm_extracted_terms.csv   ({len(df)} rows)")
    print(f"  {out}/llm_raw_results.json")
    print(f"  {out}/llm_raw_responses.json")

    if args.compare:
        compare_with_medbert(df, args.compare)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract bone tumor imaging features from German radiology reports "
                    "using a local HuggingFace LLM (separated befund/beurteilung extraction)."
    )
    parser.add_argument(
        "--reports", default=None,
        help="Path to a .json reports file.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Load model in 4-bit (requires bitsandbytes + CUDA). "
             "Reduces VRAM from ~15 GB to ~5 GB for 7B models.",
    )
    parser.add_argument("--max", type=int, default=None, help="Max number of reports to process.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens per report.")
    parser.add_argument("--out_dir", default="results", help="Output directory (default: results/).")
    parser.add_argument(
        "--compare", default=None, metavar="CSV",
        help="Path to medbert extracted_terms.csv to compare both approaches.",
    )
    return parser.parse_args()
