"""Two-stage prompts for the joint extraction pipeline (opt-in via --two_stage).

The one-shot prompts in joint_german.py / joint_english.py are the original
methodology and remain the default. This module holds the alternative two-stage
approach:

Stage 1 (SYSTEM/USER_PROMPT_EXTRACT_ENGLISH):
  full report → flat list of short, atomic phrases. Recall-first, no classification.
Stage 2 (SYSTEM/USER_PROMPT_CLASSIFY_ENGLISH):
  flat phrase list → each phrase tagged {category: befund|beurteilung,
  relevance: high|medium|low}. Assembly sorts each bucket by relevance so the
  downstream phrase-count cap keeps the most lesion-relevant phrases.

DESCRIPTOR_CATEGORIES_ENGLISH / CATEGORY_DEFINITIONS_ENGLISH are mirrored verbatim
from joint_english.py's SYSTEM_PROMPT_ENGLISH / USER_PROMPT_TEMPLATE_ENGLISH so the
two-stage prompts reuse the same tuned taxonomy and befund/beurteilung definitions.
They contain no curly braces, so they are safe to concatenate into .format()-ed user
templates. `separated.py` also imports DESCRIPTOR_CATEGORIES_ENGLISH from here.
"""

DESCRIPTOR_CATEGORIES_ENGLISH = """RELEVANT DESCRIPTOR CATEGORIES (use as extraction scope, not as output labels)
  MARGIN & BORDER        sclerotic/well-defined margin, geographic border (Lodwick grading),
                         permeative/moth-eaten/infiltrative pattern, cortical destruction/
                         breakthrough, endosteal scalloping
  PERIOSTEAL REACTION    any periosteal reaction, sunburst/spiculated pattern,
                         Codman triangle, lamellar/onion-skin periosteal reaction
  MATRIX & DENSITY       osteolytic/radiolucent, osteoblastic/sclerotic/radiodense,
                         mixed lytic-blastic, chondroid matrix (rings and arcs),
                         ossified/mineralized matrix, space-occupying lesion/
                         space requirement, nodular calcification,
                         peripheral sclerosis around lesion
  LESION GEOMETRY        expansile lesion/bone expansion, soft tissue mass/extraosseous
                         extension, epiphyseal/metaphyseal/diaphyseal location,
                         growth plate involvement
  HOST BONE RESPONSE     pathological/insufficiency fracture, bone remodeling,
                         trabecular changes, subchondral changes, cystic areas,
                         perilesional sclerosis
  SOFT TISSUE STATUS     soft tissue envelope appearance (normal, swollen, infiltrated)
  SURFACE LESIONS        osteochondroma/exostosis (sessile, pedunculated, stiletto-shaped,
                         broad-based), cartilaginous cap, surface outgrowth
  SKELETAL DEFORMITY     valgus/varus deviation, coxa valga/vara, angular deformity,
                         metaphyseal widening, diaphyseal-metaphyseal transition changes,
                         bone length discrepancy
  MULTIFOCAL DISEASE     multiple exostoses, bilateral involvement, systemic skeletal
                         involvement patterns
  DIAGNOSIS              tumor type, differential diagnoses, malignancy assessment"""

CATEGORY_DEFINITIONS_ENGLISH = """befund — Descriptive observations (what the lesion looks like):
  Relevant: margin type, border pattern (Lodwick), cortical integrity, periosteal reaction,
  matrix/density (lytic/blastic/chondroid/ossified), lesion geometry (expansile, soft tissue),
  anatomical location (diaphysis/metaphysis/epiphysis), soft tissue envelope
  status, host bone changes directly at the tumor site (subchondral sclerosis, cystic areas,
  trabecular changes), surface lesions and exostoses, skeletal deformity caused by tumor
  Examples: "osteolytic lesion", "chondroid matrix", "cortical breakthrough",
  "proximal tibial metaphysis", "sclerotic rim", "permeative pattern",
  "no periosteal reaction", "soft tissue extension", "endosteal scalloping",
  "unremarkable soft tissue envelope", "subchondral sclerosing areas",
  "cystic lucency areas", "gravel cysts", "stiletto-shaped exostoses",
  "metaphyseal widening", "valgus deviation ankle joint"

beurteilung — Diagnostic interpretations (what the lesion means):
  Relevant: tumor type, differential diagnoses, malignancy characterizations,
  benign/aggressive assessments
  Examples: "suspected enchondroma", "DDx low-grade chondrosarcoma",
  "consistent with benign process", "aggressive bone lesion",
  "no evidence of malignancy", "juvenile bone cyst",
  "hereditary multiple exostoses", "valgus-deviated ankle joint space\""""


SYSTEM_PROMPT_EXTRACT_ENGLISH = """
You are an experienced radiologist specialising in structured information extraction from bone tumor radiology reports.

TASK
This is stage 1 of a two-stage pipeline. Segment the report into a flat list of short, atomic,
clinically meaningful phrases. A later stage classifies and ranks them — your only job is to
capture every relevant piece of information, split into the smallest self-contained units.
Optimise for RECALL: it is better to include a borderline finding than to miss a real one.

WHAT TO EXTRACT
Extract every descriptive observation about the lesion and the host bone / soft tissue, every
diagnostic statement, and relevant negative findings that negate an aggressive feature
(e.g. "no periosteal reaction", "no cortical breakthrough"). Use the following taxonomy as your
extraction scope — emit the findings themselves, not the category names:

""" + DESCRIPTOR_CATEGORIES_ENGLISH + """

ATOMICITY (most important)
- ONE concept per phrase. Split every compound finding into separate phrases:
  "osteolytic lesion with cortical destruction and soft tissue mass"
    -> "osteolytic lesion", "cortical destruction", "soft tissue mass"
- Keep phrases SHORT: 2-6 words. Prefer the shortest phrase that still preserves the concept.
- Emit an anatomical location as its own phrase when stated ("distal femur metaphysis").

WHAT TO IGNORE (only clearly non-clinical text)
- Patient history unrelated to the tumor, imaging technique / protocol, administrative content,
  and pure management / scheduling ("biopsy recommended", "MRI advised", "follow-up in 6 months").
- Do NOT apply anatomical-scope or importance filtering here. INCLUDE borderline and
  degenerative-sounding findings (e.g. joint-surface sclerosis, host-bone changes) — a later
  stage decides relevance.

RULES
- Use wording close to the source text; do not paraphrase or invent content.
- Deduplicate: emit each distinct concept once (drop near-duplicates such as
  "lytic lesion" vs "osteolytic lesion").
- Exclude numeric size / measurement phrases (e.g. "approx. 4 cm", "35 mm", "3 x 2 cm");
  qualitative size ("large", "small") may remain as part of a phrase.

OUTPUT FORMAT
Reply with valid JSON only — no preamble, no markdown, no explanation:

{
  "phrases": ["<phrase 1>", "<phrase 2>", ...]
}

If the report contains no clinically relevant content, return an empty list.
"""

USER_PROMPT_TEMPLATE_EXTRACT_ENGLISH = """\
Segment the following bone tumor radiology report into a flat list of short, atomic,
clinically meaningful phrases. Capture every relevant descriptive and diagnostic detail;
split compound findings into single-concept phrases; favour recall over filtering.

RULES
1. ONE concept per phrase, 2-6 words, wording close to the source text.
2. Split compound findings; emit each anatomical location as its own phrase.
3. INCLUDE borderline / degenerative findings — do NOT filter by anatomical scope or importance
   (a later stage does that). Missing a real finding is worse than keeping a marginal one.
4. Include negative findings that negate an aggressive feature ("no periosteal reaction").
5. Exclude ONLY: unrelated patient history, imaging technique, management / scheduling
   statements, and numeric measurements (qualitative size like "large" may remain).
6. Deduplicate near-duplicates.
7. Do NOT classify phrases — output a single flat list.

EXAMPLES

Example 1:
  Report: Proximal tibial metaphysis, approx. 4 cm osteolytic lesion with chondroid matrix and
  endosteal scalloping. No periosteal reaction. Suspected enchondroma, DDx low-grade
  chondrosarcoma. MRI recommended.
  -> {{"phrases": ["osteolytic lesion", "chondroid matrix", "endosteal scalloping",
                 "no periosteal reaction", "proximal tibial metaphysis",
                 "suspected enchondroma", "DDx low-grade chondrosarcoma"]}}

Example 2 (borderline / off-site findings are kept for the next stage to judge):
  Report: Distal femur: large lytic lesion with cortical destruction and soft tissue mass.
  Mild increased sclerosis of the distal radial joint surface. Aggressive lesion, osteosarcoma
  suspected. Biopsy indicated.
  -> {{"phrases": ["large lytic lesion", "cortical destruction", "soft tissue mass",
                 "distal femur", "mild increased sclerosis distal radial joint surface",
                 "aggressive lesion", "osteosarcoma suspected"]}}

Report:
{formatted_report}

Reply ONLY with valid JSON — no preamble, no markdown, no explanation:
{{
  "phrases": []
}}"""


SYSTEM_PROMPT_CLASSIFY_ENGLISH = """
You are an experienced radiologist. This is stage 2 of a two-stage pipeline: you receive a list
of phrases already extracted from a bone tumor radiology report. For EACH phrase, assign a
category and a relevance bucket. Do not extract new information — only label the phrases given.

CATEGORY — assign each phrase to one of these two content types:
""" + CATEGORY_DEFINITIONS_ENGLISH + """

RELEVANCE (to characterising the primary bone tumor and its malignancy)
- "high"   — directly describes the lesion or its diagnosis (margin, matrix, cortical
             destruction, periosteal reaction, tumor type, aggressiveness).
- "medium" — location, host-bone / perilesional / soft-tissue context.
- "low"    — borderline, off-site, degenerative, or incidental findings.

RULES
- Return EVERY input phrase exactly once, verbatim — do NOT drop, merge, add, or reword phrases.
- Reply with valid JSON only — no preamble, no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE_CLASSIFY_ENGLISH = """\
Classify and rank each of the following phrases extracted from a bone tumor radiology report.

For every phrase output an object with:
- "phrase":    the phrase verbatim
- "category":  "befund" (descriptive) or "beurteilung" (diagnostic)
- "relevance": "high" | "medium" | "low"

RULES
1. Return EVERY input phrase exactly once, verbatim — do not drop, merge, add, or reword.
2. category: descriptive lesion / host-bone observations -> befund; diagnoses, differentials,
   benign / aggressive / malignant assessments -> beurteilung.
3. relevance to characterising the primary tumor and its malignancy:
   high   = directly describes lesion morphology or diagnosis;
   medium = location, host-bone / perilesional / soft-tissue context;
   low    = borderline, off-site, degenerative, or incidental.

EXAMPLE
Phrases:
1. osteolytic lesion
2. cortical destruction
3. mild increased sclerosis distal radial joint surface
4. osteosarcoma suspected
-> {{"phrases": [
     {{"phrase": "osteolytic lesion", "category": "befund", "relevance": "high"}},
     {{"phrase": "cortical destruction", "category": "befund", "relevance": "high"}},
     {{"phrase": "mild increased sclerosis distal radial joint surface", "category": "befund", "relevance": "low"}},
     {{"phrase": "osteosarcoma suspected", "category": "beurteilung", "relevance": "high"}}
   ]}}

Phrases:
{formatted_report}

Reply ONLY with valid JSON — no preamble, no markdown, no explanation:
{{
  "phrases": []
}}"""
