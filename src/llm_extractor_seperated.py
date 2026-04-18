"""
LLM-based Bone Tumor Evidence Phrase Extractor  (HuggingFace)
=============================================================
Uses a local generative LLM from HuggingFace to extract free-form
diagnostic evidence phrases from German radiology reports (LGDEA style),
split into two lists:
  - lesion_phrases   : direct visual characteristics of the lesion
                       (shape, margins, matrix, periosteal reaction, …)
                       → intended as mask tokens
  - context_phrases  : surrounding anatomy, organ involvement, location,
                       clinical context, patient history
                       → intended as context tokens

Purpose: Test whether free-form LGDEA-style evidence phrase extraction is
         viable compared to the KeyBERT/medbert approach in medbert_extractor.py.

Recommended models (instruction-tuned, German-capable):
    Qwen/Qwen2.5-7B-Instruct          ~15 GB fp16 / ~5 GB 4-bit  (default)
    Qwen/Qwen2.5-3B-Instruct          ~6  GB fp16 / ~2 GB 4-bit  (fast, weaker)
    mistralai/Mistral-7B-Instruct-v0.3 ~15 GB fp16 / ~5 GB 4-bit
    LeoLM/leo-mistral-hessianai-7b-chat ~15 GB fp16              (German fine-tune)

Usage:
    python llm_extractor.py                             # sample reports, default model
    python llm_extractor.py --max 5                     # quick test (5 reports)
    python llm_extractor.py --reports deine.json        # own data
    python llm_extractor.py --model Qwen/Qwen2.5-3B-Instruct --quantize  # low VRAM
    python llm_extractor.py --compare results/extracted_terms.csv        # vs medbert
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ── Default model ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ── Prompt ────────────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """\
# Du bist ein erfahrener Radiologe. Extrahiere aus deutschen Radiologiebefunden \
# diagnostische Evidenzphrasen aus zwei Sektionen: "Befund" (deskriptive Beobachtungen) \
# und "Beurteilung" (klinische Schlussfolgerungen). \
# Nutze NUR Formulierungen die im Originaltext stehen außer Beurteilung ist leer, dann schlussfolgere diese. \
# Übersetze letztlich die deutschen Phrasen in Englisch. \
# Antworte ausschließlich mit validem JSON."""

SYSTEM_PROMPT = """\
Du bist ein erfahrener Radiologe und Experte für strukturierte medizinische Informationsextraktion.

Extrahiere diagnostische Evidenzphrasen aus deutschen Radiologiebefunden.

Wichtige Prinzipien:
- Trenne strikt zwischen:
  • Befund (Beobachtungen)
  • Beurteilung (Interpretation / Diagnose)
- Nutze für Befund möglichst exakte Originalformulierungen
- Falls keine Beurteilung vorhanden ist, leite eine kurze diagnostische Einschätzung aus dem Befund ab
- Alle finalen Phrasen müssen auf Englisch ausgegeben werden

Antworte ausschließlich mit validem JSON.
"""

# USER_PROMPT_TEMPLATE = """\
# Lies den folgenden Radiologiebefund und extrahiere diagnostische Evidenzphrasen \
# getrennt nach Quelle.

# KRITISCHE REGEL:
# - "befund_phrases": Extrahiere NUR aus der BEFUND-Sektion
# - "beurteilung_phrases": Extrahiere NUR aus der BEURTEILUNG-Sektion (falls vorhanden)
#   Falls BEURTEILUNG fehlt: Destilliere die wichtigsten medizinischen Merkmale aus BEFUND

# Definitionen:
# - "befund_phrases": Einzelne deskriptive Beobachtungen aus dem Befund
#   Beispiele: "osteolytische Läsion", "chondroide Matrix", "Kortikalisdestruktion", \
#   "proximale Tibia metaphysär", "ca. 4 cm Durchmesser", "sklerotischer Randsaum"

# - "beurteilung_phrases": ALLE diagnostisch relevanten Aussagen aus der Beurteilung
#   Dies umfasst:
#   • Verdachtsdiagnosen: "V.a. Enchondrom", "DD Chondrosarkom"
#   • Zusammenfassende Beschreibungen: "randsklerosierte Osteolyse", "benigne imponierende Läsion"
#   • Einschätzungen: "vereinbar mit gutartigem Prozess", "Zeichen der Malignität"
#   • Negative Befunde: "kein Anhalt für Malignität", "keine aggressiven Merkmale"
#   • Empfehlungen: "Verlaufskontrolle empfohlen", "Biopsie indiziert"

# Regeln:
# 1. Originale deutsche Formulierung (keine Umschreibungen)
# 2. Anatomische Details beibehalten
# 3. Prägnante Phrasen (2-10 Wörter), ein Konzept pro Phrase
# 4. MINDESTENS eine Phrase pro Kategorie
# 5. Extrahiere ALLE relevanten Aussagen aus der Beurteilung, nicht nur Diagnosen

# VOR DEM ANTWORTEN PRÜFE:
# - Enthält "befund_phrases" mindestens 1 Phrase? Wenn nein, extrahiere mindestens die auffälligste Läsion, Lokalisation oder Eigenschaft aus dem Befund.
# - Enthält "beurteilung_phrases" mindestens 1 Phrase? Wenn nein, leite mindestens eine diagnostische Kurzbewertung aus dem Befund ab.
# - Antworte erst, wenn beide Listen mindestens einen Eintrag enthalten.

# Beispiel 1 (Beurteilung mit Diagnose):
# BEFUND:
# Proximale Tibia metaphysär ca. 4 cm große osteolytische Läsion mit chondroider Matrix.

# BEURTEILUNG:
# V.a. Enchondrom. DD niedriggradiges Chondrosarkom.

# → {{"befund_phrases": ["osteolytische Läsion", "chondroide Matrix", "ca. 4 cm große Läsion", 
#                        "proximale Tibia metaphysär"],
#     "beurteilung_phrases": ["V.a. Enchondrom", "DD niedriggradiges Chondrosarkom"]}}

# Beispiel 2 (Beurteilung mit Beschreibung):
# BEFUND:
# Im Grundphalanxschaft Finger III links rundliche Osteolyse mit sklerotischem Randsaum.

# BEURTEILUNG:
# Randsklerosierte Osteolyse im Bereich des Grundphalanxschaftes, vereinbar mit Enchondrom.

# → {{"befund_phrases": ["rundliche Osteolyse", "sklerotischer Randsaum", 
#                        "Grundphalanxschaft Finger III links"],
#     "beurteilung_phrases": ["randsklerosierte Osteolyse", "vereinbar mit Enchondrom"]}}

# Beispiel 3 (Beurteilung mit negativem Befund):
# BEFUND:
# Distales Femur metaphysär kleine osteolytische Läsion, scharf begrenzt.

# BEURTEILUNG:
# Kein Anhalt für Malignität. Benigne imponierende Läsion, am ehesten fibröser Kortikalisdefekt.

# → {{"befund_phrases": ["kleine osteolytische Läsion", "scharf begrenzt", 
#                        "distales Femur metaphysär"],
#     "beurteilung_phrases": ["kein Anhalt für Malignität", "benigne imponierende Läsion", 
#                             "am ehesten fibröser Kortikalisdefekt"]}}

# Beispiel 4 (ohne Beurteilung):
# BEFUND:
# Proximale Tibia unscharf begrenzte lytische Läsion mit Kortikalisdestruktion und \
# aggressiver Periostreaktion.

# [Keine Beurteilung vorhanden]

# → {{"befund_phrases": ["lytische Läsion", "unscharf begrenzt", "Kortikalisdestruktion", 
#                        "aggressive Periostreaktion", "proximale Tibia"],
#     "beurteilung_phrases": ["aggressive Destruktion", "Hinweis auf maligne Läsion"]}}

# Beispiel 5 (ohne Beurteilung, benigner Eindruck):
# BEFUND:
# Kleiner rundlicher Herd im distalen Femur metaphysär, scharf begrenzt mit sklerotischem Randsaum, ohne Kortikalisdurchbruch.

# [Keine Beurteilung vorhanden]

# → ```json
# {{"befund_phrases": ["kleiner rundlicher Herd", "distales Femur metaphysär",
#                     "scharf begrenzt", "sklerotischer Randsaum", "ohne Kortikalisdurchbruch"],
# "beurteilung_phrases": ["scharf begrenzte Läsion", "kein aggressives Wachstum", "Hinweis auf benigne Läsion"]}}

# Befund:
# {formatted_report}

# Antworte NUR mit diesem JSON:
# {{
#   "befund_phrases": [],
#   "beurteilung_phrases": []
# }}"""

BEFUND_PROMPT_TEMPLATE = """\
Extrahiere deskriptive Evidenzphrasen aus folgendem Radiologiebefund.

## WAS EXTRAHIEREN?
- Morphologie (z. B. lytic lesion, sclerotic rim)
- Lokalisation (z. B. proximal tibia metaphysis)
- Größe (z. B. approx. 4 cm)
- Struktur / Matrix / Begrenzung
- Aggressivitätszeichen (z. B. cortical destruction, periosteal reaction)

## REGELN
- Möglichst originalgetreu extrahieren (keine Halluzination)
- 2–8 Wörter pro Phrase, 1 Konzept pro Phrase
- Mindestens 2 Phrasen
- Alle Phrasen auf Englisch

---

## BEISPIELE

### Beispiel 1
Proximale Tibia metaphysär ca. 4 cm große osteolytische Läsion mit chondroider Matrix.

→ {{"befund_phrases": ["osteolytic lesion", "chondroid matrix", "approximately 4 cm lesion", "proximal tibia metaphysis"]}}

### Beispiel 2
Proximale Tibia unscharf begrenzte lytische Läsion mit Kortikalisdestruktion und aggressiver Periostreaktion.

→ {{"befund_phrases": ["lytic lesion", "ill-defined margins", "cortical destruction", "aggressive periosteal reaction", "proximal tibia"]}}

---

## INPUT
{text}

## OUTPUT (STRICT JSON ONLY)
{{"befund_phrases": []}}
"""

BEURTEILUNG_PROMPT_TEMPLATE = """\
Extrahiere diagnostische Aussagen aus der folgenden Radiologie-Beurteilung.

## ZIEL
Extrahiere kurze, explizit genannte diagnostische Aussagen.
Keine neuen Informationen hinzufügen.

## WAS EXTRAHIEREN?
- Diagnosen oder Verdachtsdiagnosen
- Differenzialdiagnosen (DD)
- Malignitäts-Einschätzung (benign / malignant / indeterminate)
- Wichtige Negationen (z. B. "kein Anhalt für Malignität")
- Klinische Einschätzungen (z. B. "am ehesten ...")

## WICHTIG
- Nur Inhalte verwenden, die im Text erwähnt sind
- Leichte Standardisierung erlaubt:
  - "V.a." → "suspected"
  - "DD" → "differential"
  - "kein Anhalt für" → "no evidence of"
- Keine neuen Diagnosen erfinden
- Keine Befunddetails (Morphologie, Lokalisation, Größe) extrahieren

## REGELN
- 2–8 Wörter pro Phrase
- 1 Konzept pro Phrase
- Englisch
- Keine Duplikate
- Wenn keine diagnostische Aussage vorhanden → leere Liste

---

## BEISPIELE

### Beispiel 1
V.a. Enchondrom. DD niedriggradiges Chondrosarkom.

→ {{"beurteilung_phrases": ["suspected enchondroma", "low-grade chondrosarcoma differential"]}}

### Beispiel 2
Kein Anhalt für Malignität. Benigne imponierende Läsion.

→ {{"beurteilung_phrases": ["no evidence of malignancy", "benign-appearing lesion"]}}

### Beispiel 3
Am ehesten fibröser Kortikalisdefekt.

→ {{"beurteilung_phrases": ["most likely fibrous cortical defect"]}}

### Beispiel 4 (keine Information)
Beurteilung: s.o.

→ {{"beurteilung_phrases": []}}

---

## INPUT
{text}

## OUTPUT (STRICT JSON ONLY)
{{"beurteilung_phrases": []}}
"""

BEURTEILUNG_SUMMARY_PROMPT_TEMPLATE = """\
Es liegt keine Beurteilung vor. Leite aus folgendem Radiologiebefund 1–3 kurze diagnostische Einschätzungen ab.

## WAS ABLEITEN?
- Wichtigste diagnostische Einschätzung (benigne / maligne Tendenz)
- Verdachtsdiagnose (falls möglich)
- Zusammenfassende Charakterisierung der Läsion

## REGELN
- Nur medizinisch begründete Schlussfolgerungen aus dem Text
- 2–8 Wörter pro Phrase, 1 Konzept pro Phrase
- Alle Phrasen auf Englisch

---

## BEISPIELE

### Beispiel 1
Proximale Tibia unscharf begrenzte lytische Läsion mit Kortikalisdestruktion und aggressiver Periostreaktion.

→ {{"beurteilung_phrases": ["aggressive lesion", "suggestive of malignancy"]}}

### Beispiel 2
Kleiner rundlicher Herd im distalen Femur metaphysär, scharf begrenzt mit sklerotischem Randsaum, ohne Kortikalisdurchbruch.

→ {{"beurteilung_phrases": ["well-defined lesion", "no aggressive features", "suggestive of benign lesion"]}}

---

## INPUT
{text}

## OUTPUT (STRICT JSON ONLY)
{{"beurteilung_phrases": []}}
"""

CATEGORIES = [
    "befund_phrases",
    "beurteilung_phrases",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, quantize: bool):
    """
    Load a causal LM from HuggingFace.

    Parameters
    ----------
    model_id  : HuggingFace model ID
    quantize  : load in 4-bit (requires bitsandbytes + CUDA) to reduce VRAM
    """
    

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


# ── Inference ─────────────────────────────────────────────────────────────────

def query_llm(
    text: str,
    model,
    tokenizer,
    prompt_template: str,
    max_new_tokens: int = 512,
) -> dict:
    """Run one text through the LLM using the given prompt template and return parsed JSON."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt_template.format(text=text)},
    ]

    # apply_chat_template returns a tensor or BatchEncoding depending on
    # the transformers version — normalise to a plain tensor either way.
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

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return _parse_response(raw), raw


def _fix_invalid_escapes(s: str) -> str:
    """Replace backslashes not part of a valid JSON escape with double backslash."""
    valid_escapes = set('"\\\/bfnrtu')
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\':
            if i + 1 >= len(s):
                result.append('\\\\')
                i += 1
            elif s[i + 1] not in valid_escapes:
                result.append('\\\\')
                i += 1
            elif s[i + 1] == 'u':
                # Validate \uXXXX — must be exactly 4 hex digits
                hex_part = s[i + 2: i + 6]
                if len(hex_part) == 4 and all(c in '0123456789abcdefABCDEF' for c in hex_part):
                    result.append(s[i: i + 6])
                    i += 6
                else:
                    result.append('\\\\')
                    i += 1
            else:
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def _repair_json(s: str) -> str:
    """Best-effort repairs for common LLM JSON mistakes."""
    # Fix invalid backslash escapes
    s = _fix_invalid_escapes(s)
    # Add missing comma between a closing quote and an opening quote on the next line
    s = re.sub(r'"\s*\n(\s*)"', '",\n\\1"', s)
    # Add missing comma between ] or } and the next key
    s = re.sub(r'([}\]])\s*\n(\s*")', r'\1,\n\2', s)
    # Fix } used instead of ] to close an array (e.g. "last item"\n  }\n})
    s = re.sub(r'(")\s*\n(\s+)\}(\s*\n\s*\})', r'\1\n\2]\3', s)
    return s


def _parse_response(raw: str) -> dict:
    """Extract JSON from LLM response, robust to markdown fences and invalid escapes."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$",          "", cleaned, flags=re.MULTILINE).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {"error": "No JSON found in response", "raw": raw}

    json_str = match.group()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return json.loads(_repair_json(json_str))
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "raw": raw}


# ── Report loading (same format as medbert_extractor.py) ─────────────────────

def load_reports(source: str | None) -> tuple[list[str], list[str], list[str]]:
    """Returns (befunds, beurteilungs, patids) as separate parallel lists."""
    p = Path(source)
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON must be a list.")
        befunds, beurteilungs, patids = [], [], []
        for entry in data:
            befund = (entry.get("befund") or "").strip()
            beurteilung = (entry.get("beurteilung") or "").strip()
            if befund or beurteilung:
                befunds.append(befund)
                beurteilungs.append(beurteilung)
                patids.append(entry.get("patid", ""))
        assert len(befunds) == len(beurteilungs) == len(patids), (
            f"List length mismatch: befunds={len(befunds)}, "
            f"beurteilungs={len(beurteilungs)}, patids={len(patids)}"
        )
        print(f"Loaded {len(befunds)} reports from {p}.")
        return befunds, beurteilungs, patids

    raise ValueError(f"Unsupported source: {source}")


# ── Result processing ─────────────────────────────────────────────────────────

def flatten_to_dataframe(results: list[dict], patids: list[str]) -> pd.DataFrame:
    """One row per (patid, report_idx, category, phrase)."""
    rows = []
    for i, result in enumerate(results):
        patid = patids[i] if i < len(patids) else ""
        if "error" in result:
            rows.append({
                "patid":      patid,
                "report_idx": i,
                "category":   "error",
                "phrase":     result.get("error", "unknown"),
            })
            continue

        for cat in CATEGORIES:
            value = result.get(cat, [])
            # Normalise: accept both list and accidental string responses
            if isinstance(value, str):
                phrases = [p.strip() for p in value.split(";") if p.strip()]
            elif isinstance(value, list):
                phrases = [str(p).strip() for p in value if str(p).strip()]
            else:
                phrases = []

            if not phrases:
                rows.append({"patid": patid, "report_idx": i, "category": cat, "phrase": "not reported"})
            else:
                for phrase in phrases:
                    rows.append({"patid": patid, "report_idx": i, "category": cat, "phrase": phrase.lower()})

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("LLM EXTRACTION SUMMARY")
    print("=" * 65)

    if df.empty or "category" not in df.columns:
        print("No data extracted.")
        return

    errors = df[df["category"] == "error"]
    if not errors.empty:
        print(f"\nFailed reports : {errors['report_idx'].nunique()}")

    ok = df[(df["category"] != "error") & (df["phrase"] != "not reported")]
    if ok.empty:
        print("No phrases extracted.")
        return

    n_reports = ok["report_idx"].nunique()
    print(f"\nTotal phrases  : {len(ok)}")
    print(f"Unique phrases : {ok['phrase'].nunique()}")
    print(f"Reports OK     : {n_reports}")
    print(f"Avg per report : {len(ok) / n_reports:.1f}")

    for cat in CATEGORIES:
        grp = ok[ok["category"] == cat]
        print(f"\n── {cat} ({len(grp)} phrases, {grp['phrase'].nunique()} unique) ──")
        top = grp["phrase"].value_counts().head(10)
        for phrase, count in top.items():
            print(f"  [{count:>3}×]  {phrase}")


# ── Comparison with medbert results ──────────────────────────────────────────

def compare_with_medbert(llm_df: pd.DataFrame, medbert_csv: str) -> None:
    mb = pd.read_csv(medbert_csv)

    llm_terms = set(llm_df[llm_df["category"] != "error"]["phrase"].str.lower())
    mb_terms  = set(mb["phrase"].str.lower())

    overlap  = llm_terms & mb_terms
    llm_only = llm_terms - mb_terms
    mb_only  = mb_terms  - llm_terms

    print("\n" + "=" * 65)
    print("COMPARISON: LLM  vs  medbert/KeyBERT")
    print("=" * 65)
    print(f"  LLM unique phrases    : {len(llm_terms)}")
    print(f"  medbert unique phrases: {len(mb_terms)}")
    print(f"  Overlap               : {len(overlap)}"
          f"  ({100*len(overlap)/max(len(llm_terms),1):.0f}% of LLM terms)")
    print(f"  Only in LLM           : {len(llm_only)}")
    print(f"  Only in medbert       : {len(mb_only)}")

    print("\nTop 15 LLM-only phrases (not found by medbert):")
    freq_llm = llm_df[llm_df["category"] != "error"].groupby("phrase")["report_idx"].nunique()
    for t in freq_llm.reindex(sorted(llm_only)).nlargest(15).index:
        print(f"  [{freq_llm[t]:>3}×]  {t}")

    print("\nTop 15 medbert-only phrases (not found by LLM):")
    freq_mb = mb.groupby("phrase")["report_idx"].nunique()
    for t in freq_mb.reindex(sorted(mb_only)).nlargest(15).index:
        print(f"  [{freq_mb[t]:>3}×]  {t}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    befunds, beurteilungs, patids = load_reports(args.reports)
    if args.max:
        befunds = befunds[: args.max]
        beurteilungs = beurteilungs[: args.max]
        patids = patids[: args.max]
        print(f"Limited to {len(befunds)} reports.")

    model, tokenizer = load_model(args.model, quantize=args.quantize)

    results: list[dict] = []
    raw_responses: list[dict] = []
    for befund, beurteilung, patid in tqdm(zip(befunds, beurteilungs, patids), desc="LLM inference", total=len(befunds)):
        befund_result, befund_raw = query_llm(befund, model, tokenizer, BEFUND_PROMPT_TEMPLATE, max_new_tokens=args.max_new_tokens)
        beurteilung_result, beurteilung_raw = query_llm(beurteilung, model, tokenizer, BEURTEILUNG_PROMPT_TEMPLATE, max_new_tokens=args.max_new_tokens)

        beurteilung_phrases = beurteilung_result.get("beurteilung_phrases", [])
        summary_raw = None
        if not beurteilung_phrases:
            beurteilung_result, summary_raw = query_llm(befund, model, tokenizer, BEURTEILUNG_SUMMARY_PROMPT_TEMPLATE, max_new_tokens=args.max_new_tokens)
            beurteilung_phrases = beurteilung_result.get("beurteilung_phrases", [])

        result = {
            "befund_phrases": befund_result.get("befund_phrases", []),
            "beurteilung_phrases": beurteilung_phrases,
            "patid": patid,
        }
        raw_responses.append({
            "patid": patid,
            "befund_raw": befund_raw,
            "beurteilung_raw": beurteilung_raw,
            "summary_raw": summary_raw,
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

    raw_path = out / "llm_raw_results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    raw_responses_path = out / "llm_raw_responses.json"
    with open(raw_responses_path, "w", encoding="utf-8") as f:
        json.dump(raw_responses, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:")
    print(f"  {out}/llm_extracted_terms.csv   ({len(df)} rows)")
    print(f"  {out}/llm_raw_results.json")
    print(f"  {out}/llm_raw_responses.json")

    if args.compare:
        compare_with_medbert(df, args.compare)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract bone tumor imaging features from German radiology reports "
                    "using a local HuggingFace LLM."
    )
    parser.add_argument(
        "--reports", default=None,
        help="Path to a .json file or directory of .txt files. "
             "Omit for bundled sample reports.",
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
    parser.add_argument(
        "--max", type=int, default=None,
        help="Max number of reports to process.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max tokens the LLM may generate per report (default: 512).",
    )
    parser.add_argument(
        "--out_dir", default="results",
        help="Output directory (default: results/).",
    )
    parser.add_argument(
        "--compare", default=None, metavar="CSV",
        help="Path to medbert extracted_terms.csv to compare both approaches.",
    )

    main(parser.parse_args())
