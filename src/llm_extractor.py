"""
LLM-based Bone Tumor Evidence Phrase Extractor  (HuggingFace)
=============================================================
Uses a local generative LLM from HuggingFace to extract free-form
diagnostic evidence phrases from German radiology reports (LGDEA style),
split into two lists but concateted into one prompt for joint extraction:
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
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

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

# SYSTEM_PROMPT = """\
# Du bist ein erfahrener Radiologe und Experte für strukturierte medizinische Informationsextraktion.

# Extrahiere diagnostische Evidenzphrasen aus deutschen Radiologiebefunden.

# Wichtige Prinzipien:
# - Trenne strikt zwischen:
#   • Befund (Beobachtungen)
#   • Beurteilung (Interpretation / Diagnose)
# - Nutze für Befund möglichst exakte Originalformulierungen
# - Falls keine Beurteilung vorhanden ist, leite eine kurze diagnostische Einschätzung aus dem Befund ab
# - Alle finalen Phrasen müssen auf Englisch ausgegeben werden

# Antworte ausschließlich mit validem JSON.
# """
SYSTEM_PROMPT = """\
Du bist ein erfahrener Radiologe und Experte für strukturierte medizinische Informationsextraktion.

Arbeite präzise, textnah und ohne Halluzination.
Unterscheide zwischen deskriptiven (Befund) und diagnostischen (Beurteilung) Informationen.

Alle Ausgaben müssen medizinisch korrekt sein.
Antworte ausschließlich mit validem JSON.
"""

SYSTEM_PROMPT_ENGLISH = """\
You are an experienced radiologist and expert in structured medical information extraction.

Work precisely, close to the source text, and without hallucination.
Distinguish between descriptive (Findings) and diagnostic (Impression) information.

All outputs must be medically correct.
Reply with valid JSON only.
"""

USER_PROMPT_TEMPLATE = """\
Lies den folgenden Radiologiebefund und extrahiere diagnostische Evidenzphrasen \
getrennt nach Quelle.

KRITISCHE REGEL:
- "befund_phrases": Extrahiere NUR aus der BEFUND-Sektion
- "beurteilung_phrases": Extrahiere NUR aus der BEURTEILUNG-Sektion (falls vorhanden)
  Falls BEURTEILUNG fehlt: Destilliere die wichtigsten medizinischen Merkmale aus BEFUND

Definitionen:
- "befund_phrases": Einzelne deskriptive Beobachtungen aus dem Befund
  Beispiele: "osteolytische Läsion", "chondroide Matrix", "Kortikalisdestruktion", \
  "proximale Tibia metaphysär", "ca. 4 cm Durchmesser", "sklerotischer Randsaum"

- "beurteilung_phrases": ALLE diagnostisch relevanten Aussagen aus der Beurteilung
  Dies umfasst:
  • Verdachtsdiagnosen: "V.a. Enchondrom", "DD Chondrosarkom"
  • Zusammenfassende Beschreibungen: "randsklerosierte Osteolyse", "benigne imponierende Läsion"
  • Einschätzungen: "vereinbar mit gutartigem Prozess", "Zeichen der Malignität"
  • Negative Befunde: "kein Anhalt für Malignität", "keine aggressiven Merkmale"
  • Empfehlungen: "Verlaufskontrolle empfohlen", "Biopsie indiziert"

Regeln:
1. Originale deutsche Formulierung (keine Umschreibungen)
2. Anatomische Details beibehalten
3. Prägnante Phrasen (2-8 Wörter), ein Konzept pro Phrase
4. MINDESTENS eine Phrase pro Kategorie
5. Extrahiere ALLE relevanten Aussagen aus der Beurteilung, nicht nur Diagnosen
6. KEINE Halluzinationen oder Erfindungen von Informationen, die nicht explizit im Text stehen

VOR DEM ANTWORTEN PRÜFE:
- Enthält "befund_phrases" mindestens 1 Phrase? Wenn nein, extrahiere mindestens die auffälligste Läsion, Lokalisation oder Eigenschaft aus dem Befund.
- Enthält "beurteilung_phrases" mindestens 1 Phrase? Wenn nein, leite mindestens eine diagnostische Kurzbewertung aus dem Befund ab.
- Antworte erst, wenn beide Listen mindestens einen Eintrag enthalten.

Beispiel 1 (Beurteilung mit Diagnose):
BEFUND:
Proximale Tibia metaphysär ca. 4 cm große osteolytische Läsion mit chondroider Matrix.

BEURTEILUNG:
V.a. Enchondrom. DD niedriggradiges Chondrosarkom.

→ {{"befund_phrases": ["osteolytische Läsion", "chondroide Matrix", "ca. 4 cm große Läsion", 
                       "proximale Tibia metaphysär"],
    "beurteilung_phrases": ["V.a. Enchondrom", "DD niedriggradiges Chondrosarkom"]}}

Beispiel 2 (Beurteilung mit Beschreibung):
BEFUND:
Im Grundphalanxschaft Finger III links rundliche Osteolyse mit sklerotischem Randsaum.

BEURTEILUNG:
Randsklerosierte Osteolyse im Bereich des Grundphalanxschaftes, vereinbar mit Enchondrom.

→ {{"befund_phrases": ["rundliche Osteolyse", "sklerotischer Randsaum", 
                       "Grundphalanxschaft Finger III links"],
    "beurteilung_phrases": ["randsklerosierte Osteolyse", "vereinbar mit Enchondrom"]}}

Beispiel 3 (Beurteilung mit negativem Befund):
BEFUND:
Distales Femur metaphysär kleine osteolytische Läsion, scharf begrenzt.

BEURTEILUNG:
Kein Anhalt für Malignität. Benigne imponierende Läsion, am ehesten fibröser Kortikalisdefekt.

→ {{"befund_phrases": ["kleine osteolytische Läsion", "scharf begrenzt", 
                       "distales Femur metaphysär"],
    "beurteilung_phrases": ["kein Anhalt für Malignität", "benigne imponierende Läsion", 
                            "am ehesten fibröser Kortikalisdefekt"]}}

Beispiel 4 (ohne Beurteilung):
BEFUND:
Proximale Tibia unscharf begrenzte lytische Läsion mit Kortikalisdestruktion und \
aggressiver Periostreaktion.

[Keine Beurteilung vorhanden]

→ {{"befund_phrases": ["lytische Läsion", "unscharf begrenzt", "Kortikalisdestruktion", 
                       "aggressive Periostreaktion", "proximale Tibia"],
    "beurteilung_phrases": ["aggressive Destruktion", "Hinweis auf maligne Läsion"]}}

Beispiel 5 (ohne Beurteilung, benigner Eindruck):
BEFUND:
Kleiner rundlicher Herd im distalen Femur metaphysär, scharf begrenzt mit sklerotischem Randsaum, ohne Kortikalisdurchbruch.

[Keine Beurteilung vorhanden]

→ ```json
{{"befund_phrases": ["kleiner rundlicher Herd", "distales Femur metaphysär",
                    "scharf begrenzt", "sklerotischer Randsaum", "ohne Kortikalisdurchbruch"],
"beurteilung_phrases": ["scharf begrenzte Läsion", "kein aggressives Wachstum", "Hinweis auf benigne Läsion"]}}

Befund:
{formatted_report}

Antworte NUR mit diesem JSON:
{{
  "befund_phrases": [],
  "beurteilung_phrases": []
}}"""

USER_PROMPT_TEMPLATE_ENGLISH = """\
Read the following radiology report and extract the medically important phrases \
separated by source section.

CRITICAL RULE:
- "befund_phrases": Extract ONLY from the FINDINGS section
- "beurteilung_phrases": Extract ONLY from the IMPRESSION section (if present)
  If IMPRESSION is missing: extract the most important descriptive features from FINDINGS

Definitions:
- "befund_phrases": Individual descriptive observations from the findings
  Examples: "osteolytic lesion", "chondroid matrix", "cortical breakthrough", \
  "proximal tibial metaphysis", "approx. 4 cm diameter", "sclerotic rim", \
  "well-defined margins", "soft tissue involvement"

- "beurteilung_phrases": ALL clinically relevant statements from the impression
  This includes:
  • Diagnoses and differential diagnoses: "enchondroma", "DDx chondrosarcoma"
  • Descriptive summaries: "rim-sclerotic osteolysis", "well-circumscribed lesion"
  • Clinical assessments: "consistent with benign process", "radiologically unremarkable"
  • Negative findings: "no periosteal reaction", "no soft tissue mass"
  • Recommendations: "follow-up recommended", "biopsy indicated"

Rules:
1. Use original phrasing from the text (no paraphrasing or interpretation)
2. Preserve anatomical details and measurements
3. Concise phrases (2–8 words), one concept per phrase
4. AT LEAST one phrase per category
5. Extract ALL relevant statements from the impression
6. NO hallucinations or invented information not explicitly stated in the text

BEFORE ANSWERING CHECK:
- Does "befund_phrases" contain at least 1 phrase? If not, extract at least the most \
prominent lesion, location, or morphological feature from the findings.
- Does "beurteilung_phrases" contain at least 1 phrase? If not, extract the most \
important clinical statement from the findings section.
- Only reply once both lists contain at least one entry.

Example 1 (with impression):
FINDINGS:
Proximal tibial metaphysis approx. 4 cm osteolytic lesion with chondroid matrix.

IMPRESSION:
Suspected enchondroma. DDx low-grade chondrosarcoma.

→ {{"befund_phrases": ["osteolytic lesion", "chondroid matrix", "approx. 4 cm lesion",
                       "proximal tibial metaphysis"],
    "beurteilung_phrases": ["suspected enchondroma", "DDx low-grade chondrosarcoma"]}}

Example 2 (with impression):
FINDINGS:
Rounded osteolysis with sclerotic rim in the proximal phalanx shaft of finger III left.

IMPRESSION:
Rim-sclerotic osteolysis in the proximal phalanx shaft, consistent with enchondroma.

→ {{"befund_phrases": ["rounded osteolysis", "sclerotic rim",
                       "proximal phalanx shaft finger III left"],
    "beurteilung_phrases": ["rim-sclerotic osteolysis", "consistent with enchondroma"]}}

Example 3 (no impression):
FINDINGS:
Distal femoral metaphysis small osteolytic lesion, sharply marginated, no cortical breakthrough, \
no periosteal reaction.

[No impression available]

→ {{"befund_phrases": ["small osteolytic lesion", "sharply marginated",
                       "distal femoral metaphysis", "no cortical breakthrough",
                       "no periosteal reaction"],
    "beurteilung_phrases": ["sharply marginated osteolytic lesion", "no cortical involvement"]}}

Report:
{formatted_report}

Reply ONLY with this JSON:
{{
  "befund_phrases": [],
  "beurteilung_phrases": []
}}"""

# USER_PROMPT_TEMPLATE = """\
# Extrahiere medizinische Evidenzphrasen aus einem radiologischen Bericht.

# ## WICHTIG: SEMANTISCHE TRENNUNG (NICHT STRUKTURELL)
# - "befund_phrases": deskriptive, bildnahe Informationen
# - "beurteilung_phrases": diagnostische oder interpretative Aussagen
# - Die Zuordnung erfolgt nach INHALT, nicht nach Textabschnitt

# ## FALL: "s.o." ODER FEHLENDE BEURTEILUNG
# - Wenn die Beurteilung leer ist oder nur referenziell ("s.o."):
#   → extrahiere diagnostische Aussagen aus dem Befund
# - Verwende nur explizit genannte Informationen
# - KEINE neuen Diagnosen erfinden

# ## WAS EXTRAHIEREN?

# ### befund_phrases (deskriptiv / lokal)
# - Morphologie
# - Lokalisation
# - Größe
# - Struktur / Matrix / Begrenzung
# - Aggressivitätszeichen

# ### beurteilung_phrases (interpretativ / global)
# - Diagnosen / Verdachtsdiagnosen
# - Differenzialdiagnosen
# - Malignitäts-Einschätzung
# - Negative Aussagen
# - Klinische Gesamteinschätzung
# - Bestätigende Wiederholung eines Befunds (auch rein deskriptiv — Wiederholung in der Beurteilung = Diagnosebestätigung)


# ## REGELN
# - Befund: möglichst originalgetreu
# - Beurteilung: leichte Normalisierung erlaubt (z. B. "V.a." → "suspected")
# - 2–8 Wörter pro Phrase
# - 1 Konzept pro Phrase
# - Keine Halluzination
# - Keine Duplikate innerhalb einer Liste (Überschneidungen zwischen befund_ und beurteilung_phrases sind erlaubt und erwünscht)


# ## QUALITÄTSCHECK
# - Wenn möglich: ≥2 befund_phrases
# - Wenn vorhanden: ≥1 beurteilung_phrase
# - Keine künstlichen Ergänzungen erzwingen
# ---

# ## BEISPIELE

# ### Beispiel 1 (Diagnose vorhanden)
# BEFUND:
# Proximale Tibia metaphysär ca. 4 cm große osteolytische Läsion mit chondroider Matrix.

# BEURTEILUNG:
# V.a. Enchondrom. DD niedriggradiges Chondrosarkom.

# →
# {{
#   "befund_phrases": [
#     "osteolytische Läsion",
#     "chondroide Matrix",
#     "ca. 4 cm große Läsion",
#     "proximale Tibia metaphysär"
#   ],
#   "beurteilung_phrases": [
#     "V.a. Enchondrom",
#     "DD niedriggradiges Chondrosarkom"
#   ]
# }}

# ---

# ### Beispiel 2 (Beschreibung + Diagnose)
# BEFUND:
# Im Grundphalanxschaft Finger III links rundliche Osteolyse mit sklerotischem Randsaum.

# BEURTEILUNG:
# Randsklerosierte Osteolyse im Bereich des Grundphalanxschaftes, vereinbar mit Enchondrom.

# →
# {{
#   "befund_phrases": [
#     "rundliche Osteolyse",
#     "sklerotischer Randsaum",
#     "Grundphalanxschaft Finger III links"
#   ],
#   "beurteilung_phrases": [
#     "randsklerosierte Osteolyse",
#     "vereinbar mit Enchondrom"
#   ]
# }}

# ---

# ### Beispiel 3 (negativer Befund)
# BEFUND:
# Distales Femur metaphysär kleine osteolytische Läsion, scharf begrenzt.

# BEURTEILUNG:
# Kein Anhalt für Malignität. Benigne imponierende Läsion, am ehesten fibröser Kortikalisdefekt.

# →
# {{
#   "befund_phrases": [
#     "kleine osteolytische Läsion",
#     "scharf begrenzt",
#     "distales Femur metaphysär"
#   ],
#   "beurteilung_phrases": [
#     "kein Anhalt für Malignität",
#     "benigne imponierende Läsion",
#     "am ehesten fibröser Kortikalisdefekt"
#   ]
# }}

# ---

# ### Beispiel 4 (keine Beurteilung, aggressive Merkmale)
# BEFUND:
# Proximale Tibia unscharf begrenzte lytische Läsion mit Kortikalisdestruktion und aggressiver Periostreaktion.

# [Keine Beurteilung vorhanden]

# →
# {{
#   "befund_phrases": [
#     "lytische Läsion",
#     "unscharf begrenzt",
#     "Kortikalisdestruktion",
#     "aggressive Periostreaktion",
#     "proximale Tibia"
#   ],
#   "beurteilung_phrases": [
#     "aggressive Merkmale",
#     "Hinweis auf Malignität"
#   ]
# }}

# ---

# ### Beispiel 5 (keine Beurteilung, benigne Merkmale)
# BEFUND:
# Kleiner rundlicher Herd im distalen Femur metaphysär, scharf begrenzt mit sklerotischem Randsaum, ohne Kortikalisdurchbruch.

# [Keine Beurteilung vorhanden]

# →
# {{
#   "befund_phrases": [
#     "kleiner rundlicher Herd",
#     "distales Femur metaphysär",
#     "scharf begrenzt",
#     "sklerotischer Randsaum",
#     "kein Kortikalisdurchbruch"
#   ],
#   "beurteilung_phrases": [
#     "benigne Bildmorphologie",
#     "keine aggressiven Merkmale"
#   ]
# }}

# ---

# ### Beispiel 6 (Beurteilung bestätigt Befund deskriptiv)
# BEFUND:
# Regelrechte Stellung im Kniegelenk. Im distalen medialen Femur die bekannte
# kartilaginäre Exostose. Kein Hinweis auf Fraktur. Unauffällige Weichteile.

# BEURTEILUNG:
# Im distalen medialen ventralen Femur die bekannte kartilaginäre Exostose.

# →
# {{
#   "befund_phrases": [
#     "kartilaginäre Exostose distales mediales Femur",
#     "regelrechte Gelenkstellung",
#     "kein Frakturhinweis",
#     "unauffällige Weichteile"
#   ],
#   "beurteilung_phrases": [
#     "bekannte kartilaginäre Exostose distales mediales ventrales Femur"
#   ]
# }}

# ---

# ## INPUT
# {formatted_report}

# ## OUTPUT (STRICT JSON ONLY)
# {{
#   "befund_phrases": [],
#   "beurteilung_phrases": []
# }}
# """

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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    report: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt_template: str = USER_PROMPT_TEMPLATE,
) -> dict:
    """Run one report through the LLM and return parsed JSON."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt_template.format(formatted_report=report)},
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

def load_reports(source: str | None, english: bool = False) -> tuple[list[str], list[str]]:
    """Returns (reports, patids). patids is empty strings for non-JSON sources.

    english=True  → use befund_en / beurteilung_en (translated_reports.json)
    english=False → use befund / beurteilung        (sanitized_reports.json)
    """
    p = Path(source)
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON must be a list.")
        if data and isinstance(data[0], str):
            return data, [""] * len(data)

        befund_key      = "befund_en"      if english else "befund"
        beurteilung_key = "beurteilung_en" if english else "beurteilung"

        reports, patids = [], []
        for entry in data:
            parts = []
            if entry.get(befund_key, "").strip():
                parts.append(entry[befund_key].strip())
            if entry.get(beurteilung_key, "").strip():
                parts.append(entry[beurteilung_key].strip())
            if parts:
                reports.append("\n\n".join(parts))
                patids.append(entry.get("patid", ""))

        lang = "English" if english else "German"
        print(f"Loaded {len(reports)} {lang} reports from {p}.")
        return reports, patids

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
    reports, patids = load_reports(args.reports, english=args.english)
    if args.max:
        reports = reports[: args.max]
        patids = patids[: args.max]
        print(f"Limited to {len(reports)} reports.")

    model, tokenizer = load_model(args.model, quantize=args.quantize)

    system_prompt        = SYSTEM_PROMPT_ENGLISH      if args.english else SYSTEM_PROMPT
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
