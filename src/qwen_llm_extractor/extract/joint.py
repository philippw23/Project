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
from qwen_llm_extractor.prompts.joint_german import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from qwen_llm_extractor.prompts.joint_english import (
    SYSTEM_PROMPT_ENGLISH,
    USER_PROMPT_TEMPLATE_ENGLISH,
)
from qwen_llm_extractor.prompts.joint_two_stage import (
    SYSTEM_PROMPT_EXTRACT_ENGLISH,
    USER_PROMPT_TEMPLATE_EXTRACT_ENGLISH,
    SYSTEM_PROMPT_CLASSIFY_ENGLISH,
    USER_PROMPT_TEMPLATE_CLASSIFY_ENGLISH,
)
from qwen_llm_extractor.utils.json_repair import _parse_response


def _entry_key(e: dict) -> str:
    """Unique key per report: accnr if present, else patid."""
    accnr = str(e.get("accnr", "")).strip()
    return accnr if accnr else str(e.get("patid", ""))


# ── Two-stage helpers ──────────────────────────────────────────────────────────
_RELEVANCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _norm_phrase(p: str) -> str:
    """Normalise a phrase for matching between stage 1 and stage 2 outputs."""
    return " ".join(str(p).strip().lower().split())


def _format_phrase_list(phrases: list[str]) -> str:
    """Render stage-1 phrases as a numbered list for the stage-2 prompt."""
    return "\n".join(f"{i + 1}. {p}" for i, p in enumerate(phrases))


def _assemble_two_stage(
    stage1_phrases: list[str], stage2_parsed: dict
) -> tuple[list[str], list[str]]:
    """Combine stage-1 phrases with stage-2 {category, relevance} tags.

    Stage 1 is the source of truth for *which* phrases exist (recall-first): every
    stage-1 phrase is kept. Stage 2 only decides category and relevance ordering.
    A phrase stage 2 fails to return defaults to (befund, low) so it is never
    dropped and sorts last. Within a bucket, higher relevance comes first; ties
    preserve stage-1 order (stable sort).
    """
    lookup: dict[str, tuple[str, str]] = {}
    items = stage2_parsed.get("phrases", []) if isinstance(stage2_parsed, dict) else []
    for it in items:
        if not isinstance(it, dict):
            continue
        phrase = str(it.get("phrase", "")).strip()
        if not phrase:
            continue
        cat = str(it.get("category", "")).strip().lower()
        rel = str(it.get("relevance", "")).strip().lower()
        lookup[_norm_phrase(phrase)] = (cat, rel)

    befund: list[tuple[int, str]] = []
    beurteilung: list[tuple[int, str]] = []
    for p in stage1_phrases:
        cat, rel = lookup.get(_norm_phrase(p), ("befund", "low"))
        rank = _RELEVANCE_RANK.get(rel, 2)
        (beurteilung if cat.startswith("beur") else befund).append((rank, p))

    befund_sorted = [p for _, p in sorted(befund, key=lambda x: x[0])]
    beur_sorted = [p for _, p in sorted(beurteilung, key=lambda x: x[0])]
    return befund_sorted, beur_sorted


def query_llm_batch(
    reports: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 512,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt_template: str = USER_PROMPT_TEMPLATE,
) -> list[tuple[dict, str]]:
    """Run a batch of reports through the LLM and return [(parsed_json, raw_text), ...].

    Processes multiple reports in one model.generate call. For memory-bandwidth-bound
    inference (batch size 1 reads all weights to produce one token), batching gives
    close to N× throughput up to the compute ceiling.

    Left-pads all sequences to the same length so the generated tokens start at a
    uniform offset, making it trivial to slice them out of the combined output tensor.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Tokenize each report separately (lengths differ), collect as 1-D tensors
    per_report_ids: list[torch.Tensor] = []
    for report in reports:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt_template.format(formatted_report=report)},
        ]
        chat_out = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        ids = chat_out["input_ids"] if hasattr(chat_out, "input_ids") else chat_out
        per_report_ids.append(ids[0])  # (seq_len,)

    # Left-pad to the longest prompt in the batch
    max_len = max(t.shape[0] for t in per_report_ids)
    batch_ids  = torch.full((len(reports), max_len), pad_id, dtype=torch.long)
    attn_mask  = torch.zeros((len(reports), max_len), dtype=torch.long)
    for i, ids in enumerate(per_report_ids):
        batch_ids[i, max_len - ids.shape[0]:] = ids
        attn_mask[i, max_len - ids.shape[0]:] = 1

    device = next(model.parameters()).device
    batch_ids = batch_ids.to(device)
    attn_mask = attn_mask.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            batch_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # output_ids shape: (batch, max_len + max_new_tokens)
    # The first max_len columns are the padded prompt — slice them off uniformly.
    results = []
    for i in range(len(reports)):
        new_tokens = output_ids[i][max_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append((_parse_response(raw), raw))
    return results


def query_llm(
    report: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt_template: str = USER_PROMPT_TEMPLATE,
) -> tuple[dict, str]:
    """Single-report wrapper around query_llm_batch (kept for backward compat)."""
    return query_llm_batch(
        [report], model, tokenizer, max_new_tokens, system_prompt, user_prompt_template
    )[0]


def _run_two_stage(
    args: argparse.Namespace,
    to_process: list[dict],
    completed: dict[str, dict],
    out_path: Path,
    model,
    tokenizer,
    befund_key: str,
    beur_key: str,
) -> None:
    """Two-stage extraction: (1) atomic phrase creation, (2) classify + relevance.

    Same model for both stages. Stage 1 turns each report into a flat phrase list;
    stage 2 tags each phrase {category, relevance}; assembly builds relevance-sorted
    befund_phrases / beurteilung_phrases. The one-shot pipeline is untouched.
    """
    if not args.english:
        print("Note: --two_stage uses the English stage prompts; run with --english.")

    all_results: list[dict] = []
    all_patids: list[str] = []
    batch_size = getattr(args, "batch_size", 1)

    for batch_start in tqdm(range(0, len(to_process), batch_size), desc="LLM two-stage"):
        batch = to_process[batch_start : batch_start + batch_size]

        report_texts = []
        for entry in batch:
            parts = []
            if entry.get(befund_key, "").strip():
                parts.append(entry[befund_key].strip())
            if entry.get(beur_key, "").strip():
                parts.append(entry[beur_key].strip())
            report_texts.append("\n\n".join(parts))

        # ── Stage 1: report → flat atomic phrase list ────────────────────────────
        stage1 = query_llm_batch(
            report_texts, model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            system_prompt=SYSTEM_PROMPT_EXTRACT_ENGLISH,
            user_prompt_template=USER_PROMPT_TEMPLATE_EXTRACT_ENGLISH,
        )
        stage1_phrases = [
            (r.get("phrases", []) if isinstance(r, dict) else []) for r, _ in stage1
        ]

        # ── Stage 2: flat phrase list → {category, relevance} per phrase ──────────
        stage2_inputs = [_format_phrase_list(ph) for ph in stage1_phrases]
        stage2 = query_llm_batch(
            stage2_inputs, model, tokenizer,
            max_new_tokens=args.stage2_max_new_tokens,
            system_prompt=SYSTEM_PROMPT_CLASSIFY_ENGLISH,
            user_prompt_template=USER_PROMPT_TEMPLATE_CLASSIFY_ENGLISH,
        )

        for entry, ph_list, (r2, _) in zip(batch, stage1_phrases, stage2):
            key = _entry_key(entry)
            patid = str(entry.get("patid", ""))
            befund, beurteilung = _assemble_two_stage(ph_list, r2)
            merged = {
                **entry,
                "befund_phrases":      befund,
                "beurteilung_phrases": beurteilung,
                "stage1_phrases":      ph_list,   # kept for inspection/debugging
            }
            completed[key] = merged
            all_results.append({
                "patid": patid,
                "befund_phrases": befund,
                "beurteilung_phrases": beurteilung,
            })
            all_patids.append(patid)

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(list(completed.values()), fh, ensure_ascii=False, indent=2)

    if all_results:
        df = flatten_to_dataframe(all_results, all_patids)
        print_summary(df)
    print(f"\nDone (two-stage). {len(completed)} reports saved to {out_path}")


def main(args: argparse.Namespace) -> None:
    """Load source reports, run LLM phrase extraction, and merge results back."""
    with open(args.input, encoding="utf-8") as fh:
        source_reports: list[dict] = json.load(fh)

    # Resume: load already-processed entries keyed by accnr (or patid as fallback)
    out_path = Path(args.output)
    completed: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            completed = {_entry_key(e): e for e in json.load(fh)}
        print(f"Resuming: {len(completed)} entries already processed.")

    befund_key = "befund_en" if args.english else "befund"
    beur_key   = "beurteilung_en" if args.english else "beurteilung"

    to_process = [
        e for e in source_reports
        if _entry_key(e) not in completed
        and (e.get(befund_key, "").strip() or e.get(beur_key, "").strip())
    ]
    if args.max:
        to_process = to_process[: args.max]
    print(f"Reports to process: {len(to_process)} / {len(source_reports)}\n")

    model, tokenizer = load_model(
        args.model, quantize=args.quantize, quantize_8bit=args.quantize_8bit,
    )

    if getattr(args, "two_stage", False):
        _run_two_stage(
            args, to_process, completed, out_path, model, tokenizer, befund_key, beur_key,
        )
        return

    system_prompt        = SYSTEM_PROMPT_ENGLISH        if args.english else SYSTEM_PROMPT
    user_prompt_template = USER_PROMPT_TEMPLATE_ENGLISH if args.english else USER_PROMPT_TEMPLATE

    all_results: list[dict] = []
    all_patids:  list[str]  = []

    batch_size = getattr(args, "batch_size", 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for batch_start in tqdm(range(0, len(to_process), batch_size), desc="LLM inference"):
        batch = to_process[batch_start : batch_start + batch_size]

        report_texts = []
        for entry in batch:
            parts = []
            if entry.get(befund_key, "").strip():
                parts.append(entry[befund_key].strip())
            if entry.get(beur_key, "").strip():
                parts.append(entry[beur_key].strip())
            report_texts.append("\n\n".join(parts))

        batch_results = query_llm_batch(
            report_texts, model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )

        for entry, (result, _) in zip(batch, batch_results):
            key   = _entry_key(entry)
            patid = str(entry.get("patid", ""))
            result["patid"] = patid

            if "error" in result:
                tqdm.write(f"  [error] key={key} — {result['error'][:80]}")

            merged = {
                **entry,
                "befund_phrases":      result.get("befund_phrases", []),
                "beurteilung_phrases": result.get("beurteilung_phrases", []),
            }
            completed[key] = merged
            all_results.append(result)
            all_patids.append(patid)

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
        help="Load model in 4-bit NF4 (requires bitsandbytes + CUDA). ~8 GB VRAM for 14B.",
    )
    parser.add_argument(
        "--quantize_8bit", action="store_true",
        help="Load model in 8-bit (requires bitsandbytes + CUDA). ~14 GB VRAM for 14B; "
             "better quality than 4-bit, fits on 2 GPUs without multi-node setup.",
    )
    parser.add_argument("--max", type=int, default=None, help="Max number of reports to process.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens per report (stage 1 in --two_stage mode). "
                             "Raise (~768-1024) if many atomic phrases are expected.")
    parser.add_argument(
        "--two_stage", action="store_true",
        help="Two-stage extraction: (1) atomic phrase creation, then (2) classify each phrase "
             "into befund/beurteilung + relevance bucket. Uses the English stage prompts. "
             "The default one-shot joint extraction is unchanged.",
    )
    parser.add_argument(
        "--stage2_max_new_tokens", type=int, default=1024,
        help="Max tokens for the stage-2 classification pass (--two_stage only). "
             "Stage-2 JSON is longer than stage-1, so this defaults higher.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Reports per model.generate call. Higher = better GPU utilisation but more VRAM. "
             "Recommended: 4 for 2× A4000 in 8-bit.",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = str(Path(args.input).parent / "full_reports.json")
    return args


if __name__ == "__main__":
    main(parse_args())
