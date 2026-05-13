"""Joint extraction pipeline: befund and beurteilung are concatenated into one LLM call.

The model receives the full report (befund + beurteilung) in a single prompt and is
asked to return both phrase lists at once. This is simpler than the separated pipeline
but gives the model a wider context — which can help or hurt depending on report length.

Supports German (default) and English (--english) prompts. English mode is intended for
use with translated_reports.json where befund_en / beurteilung_en fields are available.

Entry point: ``main()`` / ``parse_args()`` — run as a script via the package CLI.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from qwen_llm_extractor.eval.analysis import flatten_to_dataframe, print_summary
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
    """Run one report through the LLM and return (parsed_json, raw_text).

    Builds a two-message chat (system + user), tokenises it with the chat template,
    runs greedy decoding, strips the prompt tokens from the output, and parses the
    resulting text as JSON.

    Parameters
    ----------
    report               : full radiology report text (befund + beurteilung concatenated)
    model                : loaded HuggingFace CausalLM (already on the target device)
    tokenizer            : matching AutoTokenizer
    max_new_tokens       : upper bound on generated tokens (512 is enough for short phrase lists)
    system_prompt        : system-role message; switch to English variant via caller
    user_prompt_template : format string with a single ``{formatted_report}`` placeholder

    Returns
    -------
    tuple[dict, str]
        - parsed dict (may contain an ``"error"`` key if JSON parsing failed)
        - raw decoded string before JSON extraction (useful for debugging)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt_template.format(formatted_report=report)},
    ]

    chat_out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Qwen tokenizer returns a BatchEncoding dict; older/other tokenizers return a plain tensor
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
            # Disable sampling params explicitly; some HF versions warn if left as defaults
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the prompt tokens — output_ids includes the full input + generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _parse_response(raw), raw


def main(args: argparse.Namespace) -> None:
    """Load source reports, run LLM phrase extraction, and merge results back."""
    with open(args.input, encoding="utf-8") as fh:
        source_reports: list[dict] = json.load(fh)

    # Resume: load already-processed entries keyed by patid
    out_path = Path(args.output)
    completed: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            completed = {str(e["patid"]): e for e in json.load(fh)}
        print(f"Resuming: {len(completed)} entries already processed.")

    befund_key = "befund_en" if args.english else "befund"
    beur_key   = "beurteilung_en" if args.english else "beurteilung"

    to_process = [
        e for e in source_reports
        if str(e.get("patid", "")) not in completed
        and (e.get(befund_key, "").strip() or e.get(beur_key, "").strip())
    ]
    if args.max:
        to_process = to_process[: args.max]
    print(f"Reports to process: {len(to_process)} / {len(source_reports)}\n")

    model, tokenizer = load_model(args.model, quantize=args.quantize)

    system_prompt        = SYSTEM_PROMPT_ENGLISH        if args.english else SYSTEM_PROMPT
    user_prompt_template = USER_PROMPT_TEMPLATE_ENGLISH if args.english else USER_PROMPT_TEMPLATE

    all_results: list[dict] = []
    all_patids:  list[str]  = []

    for entry in tqdm(to_process, desc="LLM inference"):
        patid = str(entry.get("patid", ""))

        parts = []
        if entry.get(befund_key, "").strip():
            parts.append(entry[befund_key].strip())
        if entry.get(beur_key, "").strip():
            parts.append(entry[beur_key].strip())
        report_text = "\n\n".join(parts)

        result, _ = query_llm(
            report_text, model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )
        result["patid"] = patid

        if "error" in result:
            tqdm.write(f"  [error] patid={patid} — {result['error'][:80]}")

        # Merge phrase lists back into the source entry; keep source fields intact
        merged = {
            **entry,
            "befund_phrases":      result.get("befund_phrases", []),
            "beurteilung_phrases": result.get("beurteilung_phrases", []),
        }
        completed[patid] = merged
        all_results.append(result)
        all_patids.append(patid)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(list(completed.values()), fh, ensure_ascii=False, indent=2)

    if all_results:
        df = flatten_to_dataframe(all_results, all_patids)
        print_summary(df)

    print(f"\nDone. {len(completed)} reports saved to {out_path}")



def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments for the joint extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract bone tumor imaging features from radiology reports "
                    "using a local HuggingFace LLM (joint extraction). "
                    "Merges befund_phrases and beurteilung_phrases back into the source file."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to source reports JSON (e.g. translated_reports.json).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to enriched output JSON (default: full_reports.json next to --input).",
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
    args = parser.parse_args()
    if args.output is None:
        args.output = str(Path(args.input).parent / "full_reports.json")
    return args
