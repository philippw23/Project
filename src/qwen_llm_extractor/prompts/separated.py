"""Prompts for the separated extraction pipeline.

Each report section (befund / beurteilung) is queried with its own dedicated prompt,
giving the model a narrower, more focused task per call.

Three prompt templates are defined:
- BEFUND_PROMPT_TEMPLATE             — extracts descriptive observation phrases from the findings
- BEURTEILUNG_PROMPT_TEMPLATE        — extracts diagnostic statements from the impression
- BEURTEILUNG_SUMMARY_PROMPT_TEMPLATE — fallback used when the beurteilung section is absent
                                        or yields no phrases; derives a diagnosis from the befund

All output phrases are requested in English regardless of the input language.

The two-stage variant (opt-in) adds:
- BEFUND_EXTRACT_* / BEURTEILUNG_EXTRACT_* — stage 1 atomic per-section extraction (flat list)
- RANK_* (below)                          — stage 2 importance ranking (no classification;
                                            befund/beurteilung is ground-truth by section)
"""

# Reuse the tuned descriptor taxonomy from the joint prompts for befund extraction.
from qwen_llm_extractor.prompts.joint import DESCRIPTOR_CATEGORIES_ENGLISH

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

# ── Stage 1: per-section atomic extraction (opt-in via --two_stage) ──────────────
# Analogous to the joint stage-1 prompt, but applied to ONE section at a time so
# the befund/beurteilung split is ground-truth. Output is a flat phrase list; the
# category is implied by which section produced it. Recall-first, atomic phrases.

# SYSTEM_EXTRACTION_PROMPT_TEMPLATE = """\
# You are an experienced radiologist. Segment the following radiology report section into a flat
# list of short, atomic phrases. Do not judge relevance, importance, or category — extract every
# observation and every statement the text makes about the patient, however it is phrased. Do not
# filter by location or apparent importance.

# ATOMICITY (most important)
# - ONE concept per phrase. Split every compound finding.
# - Keep phrases SHORT: 3-8 words. Emit an anatomical location as its own phrase.

# WHAT TO EXCLUDE — only text that is not a finding or statement at all: imaging technique /
# protocol, administrative content, scheduling / management instructions.

# RULES
# - Wording close to the source text; no paraphrasing or invented content.
# - Deduplicate near-duplicates.
# - Exclude numeric size / measurement phrases; qualitative size ok.
# - Output all phrases in English.

# OUTPUT: valid JSON only: {"phrases": ["<phrase 1>", "<phrase 2>", ...]}
# """

# USER_EXTRACTION_PROMPT_TEMPLATE = """\
# Segment the following section into short, atomic phrases (3-8 words, one concept each). Extract
# everything stated — do not decide what's relevant, only exclude protocol/administrative/
# scheduling text. Exclude numeric measurements. Output phrases in English.

# ## INPUT
# {text}

# ## OUTPUT (STRICT JSON ONLY)
# {{"phrases": []}}
# """

# EXTRACTIVE RULE (most important)
# - Each phrase must be a exact, CONTIGUOUS substring copied verbatim from the input text.
# - Cut sentences at natural clause boundaries: finding type, modifier, location, secondary
#   observation. Do not skip words within a chunk. Do not merge non-adjacent parts of the text
#   into one phrase.
# - Do NOT paraphrase, reorder, normalize, translate, or substitute synonyms.
# - If one finding applies to multiple entities (e.g. "distal tibia and fibula"), keep it as ONE
#   verbatim phrase — do not split or reword into separate entity-specific phrases yourself.

SYSTEM_EXTRACTION_PROMPT_TEMPLATE = """\
You are an experienced radiologist. Segment the following radiology report section into a flat
list of short, contiguous verbatim phrases. Do not judge relevance, importance, or category —
extract every observation and every statement the text makes, however it is phrased. Do not
filter by location or apparent importance; filtering happens in a later step, not here.

EXTRACTIVE RULE (most important)
- Each phrase must be a sementically medical meaningfull one from the input text.
- If one finding applies to multiple entities (e.g. "distal tibia and fibula"), keep it as ONE
  verbatim phrase — do not split or reword into separate entity-specific phrases yourself.

ATOMICITY
- Each phrase should express one complete medically meaningful concept.
- Do not split a finding from its essential modifier or anatomical location when they are required to preserve meaning.
- Prefer the largest contiguous span that still represents a single concept.

WHAT TO EXCLUDE — only text that is not a finding/statement at all: imaging technique/protocol,
administrative content, scheduling/management instructions.

RULES
- Deduplicate exact-duplicate phrases only.
- Exclude numeric size/measurement phrases; qualitative size descriptors ok.
- Preserve original wording, casing, and language — no translation at this step.

EXAMPLE
Source: "Characteristic deformity of the metaphyseal transition of the distal tibia and fibula,
each imprinting the opposing bone"
Output: ["Characteristic deformity", "metaphyseal transition", "distal tibia and fibula",
"imprinting the opposing bone"]

OUTPUT: valid JSON only: {"phrases": ["<phrase 1>", "<phrase 2>", ...]}
"""

USER_EXTRACTION_PROMPT_TEMPLATE = """\
Extract a flat list of short, contiguous verbatim phrases from the following report.

Requirements:
- Each phrase must represent one complete medically meaningful concept.
- Prefer complete findings, anatomical structures, or pathology-location expressions over isolated noun fragments.
- Keep phrases verbatim and contiguous from the source text.
- Do not paraphrase, reorder, merge non-contiguous text, or invent wording.
- Do not split a phrase if doing so would remove its medical meaning.
- If a finding explicitly refers to multiple anatomical structures together, keep them together.
- Extract broadly; do not judge clinical importance or visual relevance.
- Exclude protocol, administrative, scheduling, and numeric measurement text.

## INPUT
{text}

## OUTPUT (STRICT JSON ONLY)
{{"phrases":[]}}
"""

# ── Stage 2: importance ranking (opt-in via --rank) ─────────────────────────────
# Category is already ground-truth (befund vs beurteilung comes from the report
# section the phrase was extracted from), so stage 2 only assigns a relevance
# bucket. Phrases are never dropped here — the caller sorts each list by relevance.


BEFUND_RANK_SYSTEM_PROMPT = """\
You are an experienced radiologist. You receive a list of atomic phrases extracted from the
FINDINGS (Befund) section of a bone tumor radiology report. Assign each phrase a relevance
bucket for its DESCRIPTIVE value in characterising the primary bone lesion. Do not add, drop,
merge, or reword phrases. Reply with valid JSON only.
"""

BEFUND_RANK_PROMPT_TEMPLATE = """\
Assign a relevance bucket to each of the following phrases.

relevance (descriptive value for characterising the primary bone lesion):
- "high"   = margin/border definition, matrix/density, cortical status, periosteal reaction,
             lesion geometry, host-bone response directly adjacent to the lesion.
- "medium" = anatomical location, perilesional or soft-tissue context not directly on the lesion.
- "low"    = borderline, degenerative, or general host-bone findings with weak diagnostic link.
- "none"   = off-site, incidental, or non-tumor findings with no bearing on this lesion.

RULES
1. Return EVERY input phrase exactly once, verbatim — do not add, drop, merge, or reword.
2. Output only the phrase and its relevance bucket.

EXAMPLE
Phrases:
1. osteolytic lesion
2. mild joint-surface sclerosis contralateral knee
→ {{"phrases": [
     {{"phrase": "osteolytic lesion", "relevance": "high"}},
     {{"phrase": "mild joint-surface sclerosis contralateral knee", "relevance": "none"}}
   ]}}

## INPUT
{text}

## OUTPUT (STRICT JSON ONLY)
{{"phrases": []}}
"""


BEURTEILUNG_RANK_SYSTEM_PROMPT = """\
You are an experienced radiologist. You receive a list of atomic phrases extracted from the
IMPRESSION (Beurteilung) section of a bone tumor radiology report. Assign each phrase a
relevance bucket for its DIAGNOSTIC value regarding the primary bone lesion and its malignancy.
Do not add, drop, merge, or reword phrases. Reply with valid JSON only.
"""

BEURTEILUNG_RANK_PROMPT_TEMPLATE = """\
Assign a relevance bucket to each of the following phrases.

relevance (diagnostic value for the primary bone lesion's malignancy):
- "high"   = direct diagnostic statement (tumor type, differential diagnosis, benign/aggressive/
             malignant characterisation) or a descriptive finding with strong malignancy signal
             (cortical destruction, aggressive periosteal reaction).
- "medium" = supportive descriptive finding (location, host-bone/soft-tissue context) or a
             relevant negative ("no cortical breakthrough").
- "low"    = borderline or degenerative finding with weak diagnostic link.
- "none"   = off-site, incidental, or management/non-diagnostic content.

RULES
1. Return EVERY input phrase exactly once, verbatim — do not add, drop, merge, or reword.
2. Output only the phrase and its relevance bucket.

EXAMPLE
Phrases:
1. suspected enchondroma
2. bilateral osteoarthritis
→ {{"phrases": [
     {{"phrase": "suspected enchondroma", "relevance": "high"}},
     {{"phrase": "bilateral osteoarthritis", "relevance": "none"}}
   ]}}

## INPUT
{text}

## OUTPUT (STRICT JSON ONLY)
{{"phrases": []}}
"""
