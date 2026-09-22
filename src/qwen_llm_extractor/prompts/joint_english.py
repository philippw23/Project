"""English prompts for the joint extraction pipeline.

The model receives the full report (befund + beurteilung) in one user message and
is asked to return both phrase lists in a single JSON response — English input,
English output. Used when running on translated_reports.json (--english flag).
"""

SYSTEM_PROMPT_ENGLISH = """\
You are an experienced musculoskeletal radiologist specializing in structured medical information extraction.

Your task is to read a radiology report and classify all medically meaningful phrases into two categories:

- befund_phrases:
  Descriptive imaging observations.
  These include BOTH positive and negative findings, normal anatomical observations,
  lesion morphology, anatomical location, matrix composition, density, margins,
  cortical status, periosteal reaction, host bone response, soft tissue findings,
  skeletal deformity, lesion multiplicity, associated imaging findings, and other
  radiographically relevant observations.

- beurteilung_phrases:
  Diagnostic interpretations.
  These include diagnoses, differential diagnoses, benign/malignant assessments,
  levels of suspicion, clinical conclusions and overall radiological assessments.

Work as closely as possible to the original wording.
Never paraphrase.
Never invent information.
Never infer findings that are not explicitly stated.

Reply with valid JSON only.
"""

USER_PROMPT_TEMPLATE_ENGLISH = """\
Read the following radiology report and extract all medically relevant phrases, \
classified by content type — not by section.

Definitions:
- "befund_phrases": Descriptive imaging observations. Extract EVERY medically meaningful imaging statement, including:
  positive findings, negative findings, normal findings,morphology, geometry, anatomical location, matrix composition,
  density, margin characteristics, cortical integrity, host bone response,
  periosteal reaction, soft tissue involvement, surface characteristics,
  skeletal deformity, lesion multiplicity, and associated imaging findings.
  Examples: "osteolytic lesion", "geographic lesion", "moth-eaten destruction", "permeative pattern",
  "proximal tibial metaphysis", "well-defined margins", "sclerotic rim", "chondroid matrix",
  "ground-glass matrix", "cortical breakthrough", "endosteal scalloping", "cortical expansion",
  "aggressive periosteal reaction", "extraosseous soft tissue mass", "parosteal lesion",
  "pathologic fracture", "multiple osseous lesions"

- "beurteilung_phrases": Diagnostic interpretations and clinical assessments
  What it means: diagnoses, differential diagnoses, benign/malignant assessment,
  level of suspicion and clinical conclusions.
  Examples: "suspected enchondroma", "compatible with osteosarcoma", "consistent with benign process"
  "highly suspicious for malignancy"

Note: Descriptive morphological findings (margin, matrix, periosteal reaction,
  cortical status) do NOT count as an explicit impression, even when
  diagnostically suggestive. An explicit impression requires a stated
  diagnosis, differential diagnosis, or diagnostic conclusion (e.g.,
  "suspected X", "consistent with X", "DDx X", "malignant/benign features").

Rules:
1. Classify by content type
2. Use original phrasing from the text (no paraphrasing or interpretation)
   for befund_phrases, and for beurteilung_phrases when the text contains
   explicit impression/summary-style statements (e.g., overall assessment,
   diagnosis, malignancy characterization) — extract these verbatim.

   If the text contains only descriptive findings with no impression-style
   statement, synthesize a concise beurteilung_phrase by paraphrasing the
   most diagnostically salient finding(s), condensing their clinical
   meaning into a single impression-style phrase. Do not paraphrase
   befund_phrases under any circumstance.
3. Preserve anatomical details
4. Concise phrases (2–8 words), one concept per phrase
5. Extract ALL relevant statements from the entire report
6. AT LEAST one phrase per category
7. NO hallucinations or invented information not in the text
8. EXCLUDE measurements, sizes and lengths — numbers with units (mm, cm) and dimension \
patterns like "6 x 8 mm", "2.3 cm", "4 cm". Keep only the descriptor: \
"6 x 8 mm osteolytic lesion" → "osteolytic lesion". Drop a phrase that would \
consist only of a measurement.
9.Fracture-related phrases: EXCLUDE all fracture phrases by default
  (e.g., "no fracture", "no fracture detected", "no evidence of fracture").
  EXCEPTION — retain only "pathological fracture" and "insufficiency
  fracture" when they describe a PRESENT finding.
  If a fracture phrase is negated (e.g., "no pathological fracture",
  "no insufficiency fracture"), exclude it entirely — do not truncate
  the negation and keep the remainder. Negation status is evaluated on
  the phrase as it appears in the text, not on a substring match against
  the retained terms.

BEFORE ANSWERING CHECK:
- Does "befund_phrases" contain at least 1 descriptive observation? \
If not, extract the most prominent morphological or anatomical feature present.
- Does "beurteilung_phrases" contain at least 1 diagnostic interpretation? \
If none exists in the report, extract the closest clinical characterization available \
(e.g., a descriptive summary implying a clinical meaning).
- Remove every measurement/size from each phrase (e.g. "4 cm", "6 x 8 mm", \
"2.3 cm") before replying.
- Only reply once both lists contain AT LEAST one entry.

Example 1 (standard report with both types present):

Proximal tibial metaphysis approx. 4 cm osteolytic lesion with chondroid matrix.
Suspected enchondroma. DDx low-grade chondrosarcoma.

→ {{"befund_phrases": ["osteolytic lesion", "chondroid matrix",
                       "proximal tibial metaphysis"],
    "beurteilung_phrases": ["suspected enchondroma", "DDx low-grade chondrosarcoma"]}}

Example 2 (impression mixes descriptive and diagnostic content):

Rounded osteolysis with sclerotic rim in the proximal phalanx shaft of finger III left.
Rim-sclerotic osteolysis in the proximal phalanx shaft, consistent with enchondroma.

→ {{"befund_phrases": ["rounded osteolysis", "sclerotic rim",
                       "proximal phalanx shaft finger III left",
                       "rim-sclerotic osteolysis"],
    "beurteilung_phrases": ["consistent with enchondroma"]}}

Example 3 (no explicit impression, all content is findings):

Distal femoral metaphysis small osteolytic lesion, sharply marginated, no cortical \
breakthrough, no periosteal reaction.

→ {{"befund_phrases": ["small osteolytic lesion", "sharply marginated",
                       "distal femoral metaphysis", "no cortical breakthrough",
                       "no periosteal reaction"],
    "beurteilung_phrases": ["sharply marginated lesion without cortical involvement"]}}

Example 4 (measurements MUST be stripped — keep only the descriptor):

6 x 8 mm osteolytic lesion in the distal radius, sharply marginated. \
Approx. 4 cm area of chondroid matrix.

→ {{"befund_phrases": ["osteolytic lesion", "distal radius", "sharply marginated",
                       "chondroid matrix"],
    "beurteilung_phrases": ["sharply marginated osteolytic lesion"]}}

Report:
{formatted_report}

Reply ONLY with this JSON:
{{
  "befund_phrases": [],
  "beurteilung_phrases": []
}}"""


# ── Earlier draft (superseded, kept for reference) ─────────────────────────────
# An older, more elaborate version of SYSTEM_PROMPT_ENGLISH / USER_PROMPT_TEMPLATE_ENGLISH
# with an explicit anatomical-scope rule and a spelled-out descriptor-category taxonomy.
# Not imported anywhere active; superseded by the live prompts above.

# SYSTEM_PROMPT_ENGLISH = """
# You are an experienced radiologist specializing in structured information extraction from bone tumor radiology reports.

# TASK
# Extract all clinically relevant phrases and classify each by content type into exactly two categories:

# - befund_phrases: DESCRIPTIVE observations — morphology, location, matrix composition,
#   margins, cortical status, periosteal reaction, soft tissue status, host bone changes
#   at or adjacent to the lesion site (what the lesion looks like)
# - beurteilung_phrases: DIAGNOSTIC interpretations — tumor type, differential diagnoses,
#   malignancy grading, benign/aggressive characterizations (what the lesion means)

# CLASSIFICATION RULES
# - Classify by CONTENT TYPE only — ignore section headers (Findings/Impression/Beurteilung/Befund)
# - Section headers may be absent, overlapping, or mixed; do not use them as a classification signal
# - A single sentence may yield phrases in BOTH categories if it contains descriptive and interpretive content
# - Extract phrases close to the source text; do not paraphrase or hallucinate content
# - Exclude ONLY: patient history unrelated to the tumor, clinical management statements
#   (e.g. "biopsy recommended", "MRI advised"), numeric size/measurement phrases
#   (e.g. "approx. 4 cm", "35 mm", "3 x 2 cm"), and findings at anatomically unrelated sites

# ANATOMICAL SCOPE RULE
# Only extract findings that describe the primary tumor lesion or the bone and soft tissue
# directly hosting it. Exclude any finding at a different anatomical site, even if it uses
# bone-related terminology (e.g. degenerative changes, sclerosis, osteophytes at unrelated
# joints or bones, implant/clip material unrelated to the lesion).
# Exception: in multifocal tumor conditions, findings at multiple skeletal sites ARE relevant
# if they describe tumor manifestations themselves — not incidental co-pathology.

# RELEVANT DESCRIPTOR CATEGORIES (use as extraction scope, not as output labels)
#   MARGIN & BORDER        sclerotic/well-defined margin, geographic border (Lodwick grading),
#                          permeative/moth-eaten/infiltrative pattern, cortical destruction/
#                          breakthrough, endosteal scalloping
#   PERIOSTEAL REACTION    any periosteal reaction, sunburst/spiculated pattern,
#                          Codman triangle, lamellar/onion-skin periosteal reaction
#   MATRIX & DENSITY       osteolytic/radiolucent, osteoblastic/sclerotic/radiodense,
#                          mixed lytic-blastic, chondroid matrix (rings and arcs),
#                          ossified/mineralized matrix, space-occupying lesion/
#                          space requirement, nodular calcification,
#                          peripheral sclerosis around lesion
#   LESION GEOMETRY        expansile lesion/bone expansion, soft tissue mass/extraosseous
#                          extension, epiphyseal/metaphyseal/diaphyseal location,
#                          growth plate involvement
#   HOST BONE RESPONSE     pathological/insufficiency fracture, bone remodeling,
#                          trabecular changes, subchondral changes, cystic areas,
#                          perilesional sclerosis
#   SOFT TISSUE STATUS     soft tissue envelope appearance (normal, swollen, infiltrated)
#   SURFACE LESIONS        osteochondroma/exostosis (sessile, pedunculated, stiletto-shaped,
#                          broad-based), cartilaginous cap, surface outgrowth
#   SKELETAL DEFORMITY     valgus/varus deviation, coxa valga/vara, angular deformity,
#                          metaphyseal widening, diaphyseal-metaphyseal transition changes,
#                          bone length discrepancy
#   MULTIFOCAL DISEASE     multiple exostoses, bilateral involvement, systemic skeletal
#                          involvement patterns
#   DIAGNOSIS              tumor type, differential diagnoses, malignancy assessment

# OUTPUT FORMAT
# Reply with valid JSON only — no preamble, no markdown, no explanation.

# {
#   "befund_phrases": ["<descriptive phrase 1>", "<descriptive phrase 2>", ...],
#   "beurteilung_phrases": ["<diagnostic phrase 1>", "<diagnostic phrase 2>", ...]
# }

# If a category yields no phrases, return an empty list for that key.
# """

# USER_PROMPT_TEMPLATE_ENGLISH = """\
# Read the following bone tumor radiology report and extract clinically relevant phrases, \
# classified by content type — NOT by section header.

# DEFINITIONS

# befund_phrases — Descriptive observations (what the lesion looks like):
#   Relevant: margin type, border pattern (Lodwick), cortical integrity, periosteal reaction,
#   matrix/density (lytic/blastic/chondroid/ossified), lesion geometry (expansile, soft tissue),
#   anatomical location (diaphysis/metaphysis/epiphysis), soft tissue envelope
#   status, host bone changes directly at the tumor site (subchondral sclerosis, cystic areas,
#   trabecular changes), surface lesions and exostoses, skeletal deformity caused by tumor
#   Examples: "osteolytic lesion", "chondroid matrix", "cortical breakthrough",
#   "proximal tibial metaphysis", "sclerotic rim", "permeative pattern",
#   "no periosteal reaction", "soft tissue extension", "endosteal scalloping",
#   "unremarkable soft tissue envelope", "subchondral sclerosing areas",
#   "cystic lucency areas", "gravel cysts", "stiletto-shaped exostoses",
#   "metaphyseal widening", "valgus deviation ankle joint"

#   Exclude from befund_phrases:
#   - Numeric size/measurement phrases (e.g. "approx. 4 cm", "35 mm", "3 x 2 cm").
#     Qualitative size ("large lesion", "small focus") is allowed; only numeric dimensions are excluded
#   - Degenerative/arthrotic changes at sites other than the tumor bone
#   - Osteophytes, joint space narrowing, subchondral sclerosis at unrelated joints
#   - Implant or clip material unrelated to the lesion
#   - Bilateral symmetric findings that are degenerative or incidental, not tumor-related
#   - Any finding at a distant anatomically unrelated site (e.g. BWS, pelvis)
#     Exception: in multifocal tumor conditions, findings at multiple skeletal sites
#     ARE relevant if they describe tumor manifestations, not incidental co-pathology

# beurteilung_phrases — Diagnostic interpretations (what the lesion means):
#   Relevant: tumor type, differential diagnoses, malignancy characterizations,
#   benign/aggressive assessments
#   Examples: "suspected enchondroma", "DDx low-grade chondrosarcoma",
#   "consistent with benign process", "aggressive bone lesion",
#   "no evidence of malignancy", "juvenile bone cyst",
#   "hereditary multiple exostoses", "valgus-deviated ankle joint space"
#   Exclude: clinical recommendations, follow-up plans, procedural suggestions
#   (e.g. "biopsy recommended", "MRI advised", "follow-up in 6 months")

# EXTRACTION RULES
# 1. Classify by CONTENT TYPE — ignore Befund/Beurteilung section headers entirely
# 2. Use original phrasing from the source text — no paraphrasing or invented content
# 3. Concise phrases only (2–8 words), one concept per phrase
# 4. Perilesional and host bone changes directly at the tumor site are ALWAYS relevant
# 5. Exclude findings at anatomically unrelated sites; exception: multifocal tumor
#    manifestations across multiple skeletal sites are always in scope
# 6. A single sentence may produce phrases in BOTH categories
# 7. Both lists must contain AT LEAST one phrase
# 8. If the report's primary subject is a suspicious or indeterminate finding (even without
#    confirmed tumor diagnosis), treat that finding and its host bone as the lesion site
# 9. Include negative findings ONLY when they negate a tumor-aggressive feature:
#    e.g. "no periosteal reaction", "no cortical breakthrough", "no soft tissue extension",
#    "no osteodestructive process". Exclude generic trauma/status negations.
# 10. Exclude ALL fracture-related phrases EXCEPT "pathological fracture" and
#     "insufficiency fracture", which are direct indicators of tumor-related cortical
#     destruction. Exclude: "no fracture", "no fracture detected", "no fracture noted",
#     "no evidence of fracture", "fracture excluded", and all equivalent variants.
# 11. Exclude ALL numeric size/measurement phrases (dimensions in mm/cm, e.g. "approx. 4 cm",
#     "35 mm measurable", "3 x 2 cm"). Numeric lesion size is not an extractable concept.
#     Qualitative size words ("large", "small") may remain as part of a morphological phrase.

# FALLBACK (if a category has no explicit content):
# - befund: use the most prominent morphological or anatomical feature present
# - beurteilung: use the closest tumor characterization available in the report

# BEFORE ANSWERING CHECK:
# - Does "befund_phrases" contain at least 1 descriptive observation? \
# If not, extract the most prominent morphological or anatomical feature present.
# - Does "beurteilung_phrases" contain at least 1 diagnostic interpretation? \
# If none exists in the report, extract the closest clinical characterization available \
# (e.g., a descriptive summary implying a clinical meaning).
# - Only reply once both lists contain at least one entry.

# EXAMPLES

# Example 1 — standard report with both types:
#   Findings: Proximal tibial metaphysis, approx. 4 cm osteolytic lesion with chondroid matrix.
#   Impression: Suspected enchondroma. DDx low-grade chondrosarcoma. MRI recommended.
#   → {{
#        "befund_phrases": ["osteolytic lesion", "chondroid matrix",
#                           "proximal tibial metaphysis"],
#        "beurteilung_phrases": ["suspected enchondroma", "DDx low-grade chondrosarcoma"]
#      }}

# Example 2 — host bone and soft tissue findings present:
#   Findings: Osteolytic lesion distal radius with subchondral sclerosing areas and cystic
#   lucency. Unremarkable soft tissue envelope. No periosteal reaction.
#   Impression: Consistent with giant cell tumor.
#   → {{
#        "befund_phrases": ["osteolytic lesion", "distal radius", "subchondral sclerosing areas",
#                           "cystic lucency", "unremarkable soft tissue envelope",
#                           "no periosteal reaction"],
#        "beurteilung_phrases": ["consistent with giant cell tumor"]
#      }}

# Example 3 — distant incidental findings present (ignore them):
#   Findings: Osteolytic lesion proximal humerus, cortical thinning, no soft tissue mass.
#   Degenerative changes of the BWS. Osteophytic enlargements at the acetabular rims.
#   Impression: Consistent with simple bone cyst. Follow-up recommended.
#   → {{
#        "befund_phrases": ["osteolytic lesion", "proximal humerus", "cortical thinning",
#                           "no soft tissue mass"],
#        "beurteilung_phrases": ["consistent with simple bone cyst"]
#      }}

# Example 4 — aggressive lesion with management statement (ignore management):
#   Findings: Distal femur metaphysis: large osteolytic lesion with cortical destruction and
#   soft tissue mass. Mild degenerative changes at the contralateral hip.
#   Impression: Aggressive bone lesion, osteosarcoma suspected. Biopsy indicated.
#   → {{
#        "befund_phrases": ["large osteolytic lesion", "cortical destruction",
#                           "soft tissue mass", "distal femur metaphysis"],
#        "beurteilung_phrases": ["aggressive bone lesion", "osteosarcoma suspected"]
#      }}

# Example 5 — multifocal exostotic disease:
#   Findings: Cartilaginous exostoses at distal tibia and fibula with characteristic
#   metaphyseal widening. Clear valgus deviation of the ankle joint. Stiletto-shaped
#   exostoses of distal femur. Right coxa valga due to deformity.
#   Impression: Exostotic outgrowths consistent with hereditary multiple exostoses.
#   Deformity of distal tibia, fibula, and ankle joint space.
#   → {{
#        "befund_phrases": ["cartilaginous exostoses distal tibia and fibula",
#                           "metaphyseal widening", "valgus deviation ankle joint",
#                           "stiletto-shaped exostoses distal femur",
#                           "right coxa valga malposition"],
#        "beurteilung_phrases": ["hereditary multiple exostoses",
#                                "deformity of distal tibia and fibula",
#                                "valgus-deviated ankle joint space"]
#      }}

# Report:
# {formatted_report}

# Reply ONLY with valid JSON — no preamble, no markdown, no explanation:
# {{
#   "befund_phrases": [],
#   "beurteilung_phrases": []
# }}"""
