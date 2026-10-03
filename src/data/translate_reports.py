"""Translate German radiology reports to English using a local Qwen LLM.

Reads sanitized_reports.json, translates the `befund` and `beurteilung` fields
for each entry, and writes the result to translated_reports.json.

Supports resuming: already-translated entries (present in the output file) are
skipped so the job can be restarted after interruption.

Usage:
    python src/data/translate_reports.py                          # all reports
    python src/data/translate_reports.py --max 5                  # quick test
    python src/data/translate_reports.py --quantize               # low VRAM (4-bit)
    python src/data/translate_reports.py --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT  = ROOT_DIR / "data" / "internal_dataset" / "text" / "sanitized_reports.json"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "internal_dataset" / "text" / "translated_reports.json"
DEFAULT_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a medical translator specializing in radiology.
Translate German radiology report sections into English.
Preserve all medical terminology, anatomical details, measurements, and clinical nuances exactly.
Do not summarize, interpret, or add information. Translate only what is written.
If the input is empty or not present, return an empty string.
Reply with valid JSON only."""

USER_PROMPT_TEMPLATE = """\
Translate the following two sections of a German bone radiology report into English.

BEFUND (Findings):
{befund}

BEURTEILUNG (Assessment/Impression):
{beurteilung}

Reply ONLY with this JSON (no extra text):
{{
  "befund_en": "<English translation of BEFUND>",
  "beurteilung_en": "<English translation of BEURTEILUNG>"
}}"""


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------

# A valid JSON escape pair is kept as-is; any other lone backslash (e.g. the
# model writing "\L" for a garbled symbol) is doubled so the reply still parses.
_JSON_ESCAPE_OR_BACKSLASH = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')


def parse_json_reply(raw: str) -> dict | None:
    """Parse the JSON object the model returned, tolerating minor escape errors."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    body = raw[start:end]
    sanitized = _JSON_ESCAPE_OR_BACKSLASH.sub(
        lambda m: m.group(0) if m.group(1) else "\\\\", body
    )
    for candidate in (body, sanitized):
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, quantize: bool):
    print(f"Loading model: {model_name}  (quantize={quantize})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_kwargs: dict = {"device_map": "auto"}

    if quantize:
        if not torch.cuda.is_available():
            print("WARNING: --quantize requires CUDA. Falling back to CPU fp32.")
            load_kwargs = {}
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    elif torch.cuda.is_available():
        load_kwargs["dtype"] = torch.float16
    else:
        print("No CUDA found — running on CPU (slow).")
        load_kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on: {device}")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def translate_report(
    befund: str,
    beurteilung: str,
    tokenizer,
    model,
    max_new_tokens: int = 1024,
) -> dict[str, str]:
    """Return {'befund_en': ..., 'beurteilung_en': ...} for one report."""
    user_msg = USER_PROMPT_TEMPLATE.format(
        befund=befund.strip() if befund else "(empty)",
        beurteilung=beurteilung.strip() if beurteilung else "(empty)",
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()

    result = parse_json_reply(raw)
    if result is None:
        return {"befund_en": "", "beurteilung_en": "", "_error": raw}
    return {
        "befund_en":      result.get("befund_en", ""),
        "beurteilung_en": result.get("beurteilung_en", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def report_key(report: dict) -> tuple[str, str]:
    """Identify a report by (patid, accnr) — patid alone collides for patients
    with multiple reports/accession numbers."""
    return (str(report["patid"]), str(report.get("accnr", "")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate German radiology reports to English.")
    parser.add_argument("--input",    default=str(DEFAULT_INPUT))
    parser.add_argument("--output",   default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--max",      type=int, default=None,
                        help="Translate at most N reports (for testing)")
    parser.add_argument("--quantize", action="store_true",
                        help="Load model in 4-bit (saves VRAM)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input, encoding="utf-8") as fh:
        reports: list[dict] = json.load(fh)

    # Load already-translated entries for resuming
    out_path = Path(args.output)
    translated: dict[tuple[str, str], dict] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            existing = json.load(fh)
        translated = {report_key(e): e for e in existing}
        n_failed = sum("_error" in e for e in translated.values())
        print(f"Resuming: {len(translated) - n_failed} entries already translated, "
              f"{n_failed} failed earlier (will retry).")

    def is_done(key: tuple[str, str]) -> bool:
        return key in translated and "_error" not in translated[key]

    tokenizer, model = load_model(args.model, args.quantize)

    to_process = [r for r in reports if not is_done(report_key(r))]
    if args.max is not None:
        to_process = to_process[: args.max]

    print(f"Reports to translate: {len(to_process)} / {len(reports)}\n")

    errors = 0
    for report in tqdm(to_process, desc="Translating"):
        key = report_key(report)
        patid = str(report["patid"])
        result = translate_report(
            befund=report.get("befund", ""),
            beurteilung=report.get("beurteilung", ""),
            tokenizer=tokenizer,
            model=model,
        )

        if "_error" in result:
            errors += 1
            tqdm.write(f"  [WARN] patid={patid} — JSON parse failed: {result['_error'][:80]}")

        entry = {**report, **result}
        translated[key] = entry

        # Save after every entry so progress is never lost
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(list(translated.values()), fh, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(translated)} reports saved to {out_path}")
    if errors:
        print(f"  {errors} entries had JSON parse errors (befund_en/beurteilung_en left empty).")


if __name__ == "__main__":
    main()
