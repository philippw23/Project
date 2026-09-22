"""Separated extraction pipeline: befund and beurteilung are queried independently.

Each report section gets its own LLM call with a section-specific prompt, which
gives the model a narrower task and tends to improve precision. If the beurteilung
section is missing or yields no phrases, a third fallback call derives a summary
diagnosis from the befund text instead.

Entry point: ``main()`` / ``parse_args()`` — run as a script via the package CLI.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from qwen_llm_extractor.data.reports import load_reports_separated
from qwen_llm_extractor.eval.analysis import flatten_to_dataframe, print_summary
from qwen_llm_extractor.models.loader import DEFAULT_MODEL, load_model
from qwen_llm_extractor.prompts.separated import (
    SYSTEM_PROMPT,
    BEFUND_PROMPT_TEMPLATE,
    BEURTEILUNG_PROMPT_TEMPLATE,
    BEURTEILUNG_SUMMARY_PROMPT_TEMPLATE,
    SYSTEM_EXTRACTION_PROMPT_TEMPLATE,
    USER_EXTRACTION_PROMPT_TEMPLATE,
    BEFUND_RANK_SYSTEM_PROMPT,
    BEFUND_RANK_PROMPT_TEMPLATE,
    BEURTEILUNG_RANK_SYSTEM_PROMPT,
    BEURTEILUNG_RANK_PROMPT_TEMPLATE,
)
from qwen_llm_extractor.utils.json_repair import _parse_response


# ── Stage 2 (importance ranking) helpers ────────────────────────────────────────
_RELEVANCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _norm_phrase(p: str) -> str:
    """Normalise a phrase for matching stage-1 phrases to stage-2 relevance tags."""
    return " ".join(str(p).strip().lower().split())


def _sort_by_relevance(
    phrases: list[str], parsed: dict
) -> tuple[list[str], list[dict]]:
    """Stage 2: keep every phrase whose relevance is not "none"; order high->low.

    Buckets are high/medium/low/none. Phrases explicitly marked "none" (off-site,
    incidental, non-tumor) are dropped; high/medium/low are kept and ordered
    high->medium->low (stable within a bucket). A phrase the model omits defaults
    to "low" and is kept, so a ranking omission never silently deletes a finding.

    Returns (kept_phrases, all_tags). ``all_tags`` lists EVERY input phrase in
    stage-1 order with its assigned relevance (including "none", which are dropped
    from ``kept_phrases``) — for debugging what was extracted and how it was ranked.
    """
    if not phrases:
        return phrases, []
    lookup: dict[str, str] = {}
    items = parsed.get("phrases", []) if isinstance(parsed, dict) else []
    for it in items:
        if isinstance(it, dict) and str(it.get("phrase", "")).strip():
            lookup[_norm_phrase(it["phrase"])] = str(it.get("relevance", "")).strip().lower()

    kept: list[tuple[int, str]] = []
    all_tags: list[dict] = []
    for p in phrases:
        rel = lookup.get(_norm_phrase(p))                    # None if the model omitted it
        all_tags.append({"phrase": p, "relevance": rel if rel is not None else "low"})
        if rel == "none":
            continue                                          # drop only explicit "none"
        kept.append((_RELEVANCE_RANK.get(rel, 2), p))         # high 0 / medium 1 / low 2; omitted -> low
    kept.sort(key=lambda x: x[0])
    kept_phrases = [p for _, p in kept]
    return kept_phrases, all_tags


def query_llm(
    text: str,
    model,
    tokenizer,
    prompt_template: str,
    max_new_tokens: int = 512,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[dict, str]:
    """Run one text through the LLM using the given prompt template.

    Builds a two-message chat (system + user), tokenises it with the chat template,
    runs greedy decoding, strips the prompt tokens from the output, and parses the
    resulting text as JSON.

    Parameters
    ----------
    text             : the radiology text to extract from (befund or beurteilung)
    model            : loaded HuggingFace CausalLM (already on the target device)
    tokenizer        : matching AutoTokenizer
    prompt_template  : format string with a single ``{text}`` placeholder
    max_new_tokens   : upper bound on generated tokens (default 512 is enough for
                       a JSON list of short phrases)

    Returns
    -------
    tuple[dict, str]
        - parsed dict (may contain an ``"error"`` key if JSON parsing failed)
        - raw decoded string before JSON extraction (useful for debugging)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt_template.format(text=text)},
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


def query_llm_batch(
    texts: list[str],
    model,
    tokenizer,
    max_new_tokens: int,
    system_prompt: str,
    prompt_template: str,
) -> list[tuple[dict, str]]:
    """Batched greedy generation for a list of texts sharing one prompt template.

    Left-pads all sequences to a uniform length so generated tokens start at the
    same offset and can be sliced out per row. Returns [(parsed_json, raw), ...].
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    per_ids: list[torch.Tensor] = []
    for t in texts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt_template.format(text=t)},
        ]
        chat_out = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        ids = chat_out["input_ids"] if hasattr(chat_out, "input_ids") else chat_out
        per_ids.append(ids[0])

    max_len = max(t.shape[0] for t in per_ids)
    batch_ids = torch.full((len(texts), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(texts), max_len), dtype=torch.long)
    for i, ids in enumerate(per_ids):
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

    results = []
    for i in range(len(texts)):
        raw = tokenizer.decode(output_ids[i][max_len:], skip_special_tokens=True).strip()
        results.append((_parse_response(raw), raw))
    return results


def _batch_extract(
    texts: list[str],
    model,
    tokenizer,
    max_new_tokens: int,
    system_prompt: str,
    prompt_template: str,
) -> list[tuple[dict, str]]:
    """Stage 1 for a batch: extract phrases, skipping empty section texts.

    Empty / whitespace-only sections (e.g. a report with no beurteilung) get
    ({"phrases": []}, "") with NO model call — sending an empty section to the LLM
    made it reply with non-JSON prose ("No JSON found"). Returns [(parsed, raw), ...]
    aligned to `texts`.
    """
    results: list[tuple[dict, str]] = [({"phrases": []}, "") for _ in texts]
    idx = [i for i, t in enumerate(texts) if t and t.strip()]
    if idx:
        parsed_list = query_llm_batch(
            [texts[i] for i in idx], model, tokenizer, max_new_tokens,
            system_prompt, prompt_template,
        )
        for i, pr in zip(idx, parsed_list):
            results[i] = pr
    return results


def _batch_rank(
    phrase_lists: list[list[str]],
    model,
    tokenizer,
    max_new_tokens: int,
    rank_system_prompt: str,
    rank_prompt_template: str,
) -> tuple[list[list[str]], list[list[dict]]]:
    """Stage 2 for a batch: rank each report's phrase list in one batched call.

    Uses the given rank prompt (befund ranks by descriptive value, beurteilung by
    diagnostic value). Empty lists are skipped (no LLM call); non-empty lists are
    ranked together and scattered back. Phrases marked "none" are dropped by
    _sort_by_relevance. Returns (kept_lists, tags_lists).
    """
    sorted_out: list[list[str]] = [list(pl) for pl in phrase_lists]
    tags_out: list[list[dict]] = [[] for _ in phrase_lists]

    idx = [i for i, pl in enumerate(phrase_lists) if pl]
    if idx:
        inputs = [
            "\n".join(f"{j + 1}. {p}" for j, p in enumerate(phrase_lists[i]))
            for i in idx
        ]
        parsed_list = query_llm_batch(
            inputs, model, tokenizer, max_new_tokens,
            rank_system_prompt, rank_prompt_template,
        )
        for i, (parsed, _) in zip(idx, parsed_list):
            sorted_out[i], tags_out[i] = _sort_by_relevance(phrase_lists[i], parsed)
    return sorted_out, tags_out


def _entry_key(e: dict) -> str:
    """Unique key per report: accnr if present, else patid."""
    accnr = str(e.get("accnr", "")).strip()
    return accnr if accnr else str(e.get("patid", ""))


def _run_two_stage_separated(args: argparse.Namespace, model, tokenizer) -> None:
    """Batched two-stage per-section pipeline, writing joint-style merged output.

    Loads the full source entries (preserving patid / accnr / dates / etc.), extracts
    atomic phrases from ``befund_en`` and ``beurteilung_en`` independently (ground-truth
    split), ranks each list by importance, and merges
    ``{**entry, befund_phrases, beurteilung_phrases}`` into ``--output`` — the same
    structure as the joint pipeline. The raw stage-1 phrases and the full stage-2
    rankings (incl. dropped "none") are written to a debug file under
    ``results/llm_extraction/``. Saves incrementally after every batch and resumes by
    skipping entries already present in ``--output``.
    """
    with open(args.reports, encoding="utf-8") as fh:
        source_reports: list[dict] = json.load(fh)

    out_path = Path(args.output)
    completed: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            completed = {_entry_key(e): e for e in json.load(fh)}
        print(f"Resuming: {len(completed)} entries already processed.")

    to_process = [
        e for e in source_reports
        if _entry_key(e) not in completed
        and ((e.get("befund_en") or "").strip() or (e.get("beurteilung_en") or "").strip())
    ]
    if args.max:
        to_process = to_process[: args.max]
    print(f"Reports to process: {len(to_process)} / {len(source_reports)}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = getattr(args, "batch_size", 1)

    # Debug artefact: per-report stage-1 phrases + full stage-2 rankings (incl. dropped
    # "none"), under results/llm_extraction/<output-stem>_ranking.json.
    debug_dir = Path(args.out_dir) / "llm_extraction"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{Path(args.output).stem}_ranking.json"
    debug: dict[str, dict] = {}
    if debug_path.exists():
        with open(debug_path, encoding="utf-8") as fh:
            debug = {_entry_key(e): e for e in json.load(fh)}

    for start in tqdm(range(0, len(to_process), batch_size), desc="LLM two-stage"):
        batch = to_process[start : start + batch_size]
        bef_texts  = [(e.get("befund_en") or "").strip() for e in batch]
        beur_texts = [(e.get("beurteilung_en") or "").strip() for e in batch]

        # ── Stage 1: atomic extraction — same unified prompt; empty sections skip ─
        bef_parsed = _batch_extract(
            bef_texts, model, tokenizer, args.max_new_tokens,
            SYSTEM_EXTRACTION_PROMPT_TEMPLATE, USER_EXTRACTION_PROMPT_TEMPLATE,
        )
        beur_parsed = _batch_extract(
            beur_texts, model, tokenizer, args.max_new_tokens,
            SYSTEM_EXTRACTION_PROMPT_TEMPLATE, USER_EXTRACTION_PROMPT_TEMPLATE,
        )
        bef_lists  = [p.get("phrases", []) if isinstance(p, dict) else [] for p, _ in bef_parsed]
        beur_lists = [p.get("phrases", []) if isinstance(p, dict) else [] for p, _ in beur_parsed]

        # ── Stage 2: importance ranking; "none"-relevance phrases are dropped ─────
        # tags = full per-phrase relevance (incl. dropped "none") for the debug file.
        bef_sorted, bef_tags = _batch_rank(
            bef_lists,  model, tokenizer, args.max_new_tokens,
            BEFUND_RANK_SYSTEM_PROMPT, BEFUND_RANK_PROMPT_TEMPLATE,
        )
        beur_sorted, beur_tags = _batch_rank(
            beur_lists, model, tokenizer, args.max_new_tokens,
            BEURTEILUNG_RANK_SYSTEM_PROMPT, BEURTEILUNG_RANK_PROMPT_TEMPLATE,
        )

        for i, e in enumerate(batch):
            key = _entry_key(e)
            completed[key] = {
                **e,
                "befund_phrases":      bef_sorted[i],
                "beurteilung_phrases": beur_sorted[i],
            }
            debug[key] = {
                "patid":              e.get("patid", ""),
                "accnr":              e.get("accnr", ""),
                "befund_stage1":      bef_lists[i],       # parsed stage-1 phrases
                "befund_ranked":      bef_tags[i],        # [{phrase, relevance}], incl. "none"
                "beurteilung_stage1": beur_lists[i],
                "beurteilung_ranked": beur_tags[i],
                "befund_raw":         bef_parsed[i][1],   # raw LLM output (pre-JSON-parse)
                "beurteilung_raw":    beur_parsed[i][1],
            }
            for label, parsed in [("befund", bef_parsed[i][0]), ("beurteilung", beur_parsed[i][0])]:
                if isinstance(parsed, dict) and "error" in parsed:
                    tqdm.write(f"  [error:{label}] key={key} — {parsed['error'][:80]}")

        # Incremental save after every batch (progress visibility + crash safety).
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(list(completed.values()), fh, ensure_ascii=False, indent=2)
        with open(debug_path, "w", encoding="utf-8") as fh:
            json.dump(list(debug.values()), fh, ensure_ascii=False, indent=2)

    all_entries = list(completed.values())
    df = flatten_to_dataframe(all_entries, [e.get("patid", "") for e in all_entries])
    print_summary(df)

    print(f"\nDone (two-stage). {len(completed)} reports saved to {out_path}")
    print(f"Ranking debug written to {debug_path}")


def main(args: argparse.Namespace) -> None:
    """Load reports, run LLM inference, and save results."""
    model, tokenizer = load_model(args.model, quantize=args.quantize, quantize_8bit=args.quantize_8bit)

    # Two-stage is self-contained (loads full entries, joint-style merged output,
    # incremental save + resume). The one-shot path below is unchanged.
    if getattr(args, "two_stage", False):
        _run_two_stage_separated(args, model, tokenizer)
        return

    befunds, beurteilungs, patids = load_reports_separated(args.reports)
    if args.max:
        befunds      = befunds[: args.max]
        beurteilungs = beurteilungs[: args.max]
        patids       = patids[: args.max]
        print(f"Limited to {len(befunds)} reports.")

    results: list[dict] = []
    raw_responses: list[dict] = []
    for befund, beurteilung, patid in tqdm(
        zip(befunds, beurteilungs, patids), desc="LLM inference", total=len(befunds)
    ):
        befund_result,      befund_raw      = query_llm(befund,      model, tokenizer, BEFUND_PROMPT_TEMPLATE,      args.max_new_tokens)
        beurteilung_result, beurteilung_raw = query_llm(beurteilung, model, tokenizer, BEURTEILUNG_PROMPT_TEMPLATE, args.max_new_tokens)

        beurteilung_phrases = beurteilung_result.get("beurteilung_phrases", [])
        summary_raw = None
        # Fallback: if the beurteilung section yielded nothing (e.g. "s.o." or absent),
        # re-run with a summary prompt that derives a diagnosis from the befund instead
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

    results_path = Path(args.output)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "llm_extracted_terms.csv", index=False, encoding="utf-8-sig")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(out / "llm_raw_responses.json", "w", encoding="utf-8") as f:
        json.dump(raw_responses, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(f"  {out}/llm_extracted_terms.csv   ({len(df)} rows)")
    print(f"  {results_path}")
    print(f"  {out}/llm_raw_responses.json")



def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments for the separated extraction pipeline."""
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
        help="Load model in 4-bit NF4 (requires bitsandbytes + CUDA). ~8 GB VRAM for 14B.",
    )
    parser.add_argument(
        "--quantize_8bit", action="store_true",
        help="Load model in 8-bit (requires bitsandbytes + CUDA). ~14 GB VRAM for 14B; "
             "better quality than 4-bit, fits on 2 GPUs without multi-node setup.",
    )
    parser.add_argument("--max", type=int, default=None, help="Max number of reports to process.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens per report.")
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Reports per batched model.generate call (--two_stage only). Higher = better GPU "
             "utilisation but more VRAM. The one-shot path always runs one report at a time.",
    )
    parser.add_argument(
        "--two_stage", action="store_true",
        help="Two-stage per-section pipeline: (1) atomic extraction from befund and beurteilung "
             "independently (ground-truth split), then (2) importance ranking of each list. "
             "Persists relevance tags. The default one-shot separated extraction is unchanged.",
    )
    parser.add_argument("--out_dir", default="results", help="Output directory (default: results/).")
    parser.add_argument(
        "--output", default=None,
        help="Path to the results JSON (default: full_reports_separated.json next to --reports). "
             "The CSV and raw-responses files are still written under --out_dir.",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = str(Path(args.reports).parent / "full_reports_separated.json")
    return args


if __name__ == "__main__":
    main(parse_args())
