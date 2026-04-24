import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from qwen_llm_extractor.data.reports import load_reports_joint
from qwen_llm_extractor.eval.analysis import flatten_to_dataframe, print_summary, compare_with_medbert
from qwen_llm_extractor.models.loader import DEFAULT_MODEL, load_model
from qwen_llm_extractor.prompts.joint import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_ENGLISH,
    USER_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE_ENGLISH,
)
from qwen_llm_extractor.utils.json_repair import _parse_response


def query_llm(
    report: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt_template: str = USER_PROMPT_TEMPLATE,
) -> tuple[dict, str]:
    """Run one report through the LLM and return (parsed_json, raw_text)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt_template.format(formatted_report=report)},
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
    reports, patids = load_reports_joint(args.reports, english=args.english)
    if args.max:
        reports = reports[: args.max]
        patids  = patids[: args.max]
        print(f"Limited to {len(reports)} reports.")

    model, tokenizer = load_model(args.model, quantize=args.quantize)

    system_prompt        = SYSTEM_PROMPT_ENGLISH        if args.english else SYSTEM_PROMPT
    user_prompt_template = USER_PROMPT_TEMPLATE_ENGLISH if args.english else USER_PROMPT_TEMPLATE

    results: list[dict] = []
    raw_responses: list[dict] = []
    for report, patid in tqdm(zip(reports, patids), desc="LLM inference", total=len(reports)):
        result, raw = query_llm(
            report, model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )
        result["patid"] = patid
        results.append(result)
        raw_responses.append({"patid": patid, "raw": raw})
        if "error" in result:
            tqdm.write(f"  [error] patid={patid} — {result['error'][:80]}")

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
                    "using a local HuggingFace LLM (joint extraction)."
    )
    parser.add_argument(
        "--reports", default=None,
        help="Path to a .json reports file.",
    )
    parser.add_argument(
        "--english", action="store_true",
        help="Read befund_en/beurteilung_en instead of befund/beurteilung "
             "(use with translated_reports.json).",
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
