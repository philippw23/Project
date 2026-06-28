"""Prompts for the joint extraction pipeline.

The model receives the full report (befund + beurteilung) in one user message and
is asked to return both phrase lists in a single JSON response.

Four prompt variants are defined:
- SYSTEM_PROMPT / USER_PROMPT_TEMPLATE         — German input, German output
- SYSTEM_PROMPT_ENGLISH / USER_PROMPT_TEMPLATE_ENGLISH — English input, English output

The English variants are used when running on translated_reports.json (--english flag).
"""

SYSTEM_PROMPT = """\
Du bist ein erfahrener Radiologe und Experte für strukturierte medizinische Informationsextraktion.

Arbeite präzise, textnah und ohne Halluzination.
Unterscheide zwischen deskriptiven (Befund) und diagnostischen (Beurteilung) Informationen.

Alle Ausgaben müssen medizinisch korrekt sein.
Antworte ausschließlich mit validem JSON.
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

# SYSTEM_PROMPT_ENGLISH = """\
# You are an experienced radiologist and expert in structured medical information extraction.

# Work precisely, close to the source text, and without hallucination.
# Distinguish between descriptive (Findings) and diagnostic (Impression) information.

# All outputs must be medically correct.
# Reply with valid JSON only.
# """

# USER_PROMPT_TEMPLATE_ENGLISH = """\
# Read the following radiology report and extract the medically important phrases \
# separated by source section.

# CRITICAL RULE:
# - "befund_phrases": Extract ONLY from the FINDINGS section
# - "beurteilung_phrases": Extract ONLY from the IMPRESSION section (if present)
#   If IMPRESSION is missing: extract the most important descriptive features from FINDINGS

# Definitions:
# - "befund_phrases": Individual descriptive observations from the findings
#   Examples: "osteolytic lesion", "chondroid matrix", "cortical breakthrough", \
#   "proximal tibial metaphysis", "approx. 4 cm diameter", "sclerotic rim", \
#   "well-defined margins", "soft tissue involvement"

# - "beurteilung_phrases": ALL clinically relevant statements from the impression
#   This includes:
#   • Diagnoses and differential diagnoses: "enchondroma", "DDx chondrosarcoma"
#   • Descriptive summaries: "rim-sclerotic osteolysis", "well-circumscribed lesion"
#   • Clinical assessments: "consistent with benign process", "radiologically unremarkable"
#   • Negative findings: "no periosteal reaction", "no soft tissue mass"
#   • Recommendations: "follow-up recommended", "biopsy indicated"

# Rules:
# 1. Use original phrasing from the text (no paraphrasing or interpretation)
# 2. Preserve anatomical details and measurements
# 3. Concise phrases (2–8 words), one concept per phrase
# 4. AT LEAST one phrase per category
# 5. Extract ALL relevant statements from the impression
# 6. NO hallucinations or invented information not explicitly stated in the text

# BEFORE ANSWERING CHECK:
# - Does "befund_phrases" contain at least 1 phrase? If not, extract at least the most \
# prominent lesion, location, or morphological feature from the findings.
# - Does "beurteilung_phrases" contain at least 1 phrase? If not, extract the most \
# important clinical statement from the findings section.
# - Only reply once both lists contain at least one entry.

# Example 1 (with impression):
# FINDINGS:
# Proximal tibial metaphysis approx. 4 cm osteolytic lesion with chondroid matrix.

# IMPRESSION:
# Suspected enchondroma. DDx low-grade chondrosarcoma.

# → {{"befund_phrases": ["osteolytic lesion", "chondroid matrix", "approx. 4 cm lesion",
#                        "proximal tibial metaphysis"],
#     "beurteilung_phrases": ["suspected enchondroma", "DDx low-grade chondrosarcoma"]}}

# Example 2 (with impression):
# FINDINGS:
# Rounded osteolysis with sclerotic rim in the proximal phalanx shaft of finger III left.

# IMPRESSION:
# Rim-sclerotic osteolysis in the proximal phalanx shaft, consistent with enchondroma.

# → {{"befund_phrases": ["rounded osteolysis", "sclerotic rim",
#                        "proximal phalanx shaft finger III left"],
#     "beurteilung_phrases": ["rim-sclerotic osteolysis", "consistent with enchondroma"]}}

# Example 3 (no impression):
# FINDINGS:
# Distal femoral metaphysis small osteolytic lesion, sharply marginated, no cortical breakthrough, \
# no periosteal reaction.

# [No impression available]

# → {{"befund_phrases": ["small osteolytic lesion", "sharply marginated",
#                        "distal femoral metaphysis", "no cortical breakthrough",
#                        "no periosteal reaction"],
#     "beurteilung_phrases": ["sharply marginated osteolytic lesion", "no cortical involvement"]}}

# Report:
# {formatted_report}

# Reply ONLY with this JSON:
# {{
#   "befund_phrases": [],
#   "beurteilung_phrases": []
# }}"""

# SYSTEM_PROMPT_ENGLISH_BACKUP = """\
# You are an experienced radiologist specializing in structured medical information extraction.

# Your task: read a radiology report and classify all medically relevant phrases into two categories:

# - befund_phrases: DESCRIPTIVE observations — morphology, location, size, matrix, margins,
#   cortical status, periosteal reaction (what the lesion looks like)
# - beurteilung_phrases: DIAGNOSTIC interpretations — diagnoses, differentials, clinical
#   assessments, recommendations, benign/malignant characterizations (what the lesion means)

# Section headers (Findings/Impression) may overlap, be absent, or mix both types.
# Ignore section boundaries — classify every phrase by its CONTENT TYPE only.
# Work close to the source text. No hallucinations. Reply with valid JSON only.
# """

# USER_PROMPT_TEMPLATE_ENGLISH_BACKUP = """\
# Read the following radiology report and extract all medically relevant phrases, \
# classified by content type — not by section.

# Definitions:
# - "befund_phrases": Descriptive observations about the lesion
#   What it looks like: morphology, location, size, matrix type, margin characteristics,
#   cortical integrity, periosteal reaction, soft tissue involvement
#   Examples: "osteolytic lesion", "chondroid matrix", "cortical breakthrough",
#   "proximal tibial metaphysis", "approx. 4 cm diameter", "sclerotic rim",
#   "well-defined margins", "no periosteal reaction", "soft tissue involvement"

# - "beurteilung_phrases": Diagnostic interpretations and clinical assessments
#   What it means: diagnoses, differential diagnoses, benign/malignant characterizations,
#   clinical conclusions, recommendations
#   Examples: "suspected enchondroma", "DDx low-grade chondrosarcoma",
#   "consistent with benign process", "radiologically unremarkable",
#   "biopsy indicated", "follow-up recommended"

# Rules:
# 1. Classify by content type — ignore section headers entirely
# 2. Use original phrasing from the text (no paraphrasing or interpretation)
# 3. Preserve anatomical details and measurements
# 4. Concise phrases (2–8 words), one concept per phrase
# 5. Extract ALL relevant statements from the entire report
# 6. AT LEAST one phrase per category
# 7. NO hallucinations or invented information not in the text

# BEFORE ANSWERING CHECK:
# - Does "befund_phrases" contain at least 1 descriptive observation? \
# If not, extract the most prominent morphological or anatomical feature present.
# - Does "beurteilung_phrases" contain at least 1 diagnostic interpretation? \
# If none exists in the report, extract the closest clinical characterization available \
# (e.g., a descriptive summary implying a clinical meaning).
# - Only reply once both lists contain at least one entry.

# Example 1 (standard report with both types present):
# FINDINGS:
# Proximal tibial metaphysis approx. 4 cm osteolytic lesion with chondroid matrix.
# IMPRESSION:
# Suspected enchondroma. DDx low-grade chondrosarcoma.

# → {{"befund_phrases": ["osteolytic lesion", "chondroid matrix", "approx. 4 cm lesion",
#                        "proximal tibial metaphysis"],
#     "beurteilung_phrases": ["suspected enchondroma", "DDx low-grade chondrosarcoma"]}}

# Example 2 (impression mixes descriptive and diagnostic content):
# FINDINGS:
# Rounded osteolysis with sclerotic rim in the proximal phalanx shaft of finger III left.
# IMPRESSION:
# Rim-sclerotic osteolysis in the proximal phalanx shaft, consistent with enchondroma.

# → {{"befund_phrases": ["rounded osteolysis", "sclerotic rim",
#                        "proximal phalanx shaft finger III left",
#                        "rim-sclerotic osteolysis"],
#     "beurteilung_phrases": ["consistent with enchondroma"]}}

# Example 3 (no explicit impression, all content is findings):
# FINDINGS:
# Distal femoral metaphysis small osteolytic lesion, sharply marginated, no cortical \
# breakthrough, no periosteal reaction.

# → {{"befund_phrases": ["small osteolytic lesion", "sharply marginated",
#                        "distal femoral metaphysis", "no cortical breakthrough",
#                        "no periosteal reaction"],
#     "beurteilung_phrases": ["sharply marginated lesion without cortical involvement"]}}

# Report:
# {formatted_report}

# Reply ONLY with this JSON:
# {{
#   "befund_phrases": [],
#   "beurteilung_phrases": []
# }}"""


SYSTEM_PROMPT_ENGLISH = """
You are an experienced radiologist specializing in structured information extraction from bone tumor radiology reports.

TASK
Extract all clinically relevant phrases and classify each by content type into exactly two categories:

- befund_phrases: DESCRIPTIVE observations — morphology, location, size, matrix composition,
  margins, cortical status, periosteal reaction, soft tissue status, host bone changes
  at or adjacent to the lesion site (what the lesion looks like)
- beurteilung_phrases: DIAGNOSTIC interpretations — tumor type, differential diagnoses,
  malignancy grading, benign/aggressive characterizations (what the lesion means)

CLASSIFICATION RULES
- Classify by CONTENT TYPE only — ignore section headers (Findings/Impression/Beurteilung/Befund)
- Section headers may be absent, overlapping, or mixed; do not use them as a classification signal
- A single sentence may yield phrases in BOTH categories if it contains descriptive and interpretive content
- Extract phrases close to the source text; do not paraphrase or hallucinate content
- Exclude ONLY: patient history unrelated to the tumor, clinical management statements
  (e.g. "biopsy recommended", "MRI advised"), and findings at anatomically unrelated sites

ANATOMICAL SCOPE RULE
Only extract findings that describe the primary tumor lesion or the bone and soft tissue
directly hosting it. Exclude any finding at a different anatomical site, even if it uses
bone-related terminology (e.g. degenerative changes, sclerosis, osteophytes at unrelated
joints or bones, implant/clip material unrelated to the lesion).
Exception: in multifocal tumor conditions, findings at multiple skeletal sites ARE relevant
if they describe tumor manifestations themselves — not incidental co-pathology.

RELEVANT DESCRIPTOR CATEGORIES (use as extraction scope, not as output labels)
  MARGIN & BORDER        sclerotic/well-defined margin, geographic border (Lodwick grading),
                         permeative/moth-eaten/infiltrative pattern, cortical destruction/
                         breakthrough, endosteal scalloping
  PERIOSTEAL REACTION    any periosteal reaction, sunburst/spiculated pattern,
                         Codman triangle, lamellar/onion-skin periosteal reaction
  MATRIX & DENSITY       osteolytic/radiolucent, osteoblastic/sclerotic/radiodense,
                         mixed lytic-blastic, chondroid matrix (rings and arcs),
                         ossified/mineralized matrix
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
  DIAGNOSIS              tumor type, differential diagnoses, malignancy assessment

OUTPUT FORMAT
Reply with valid JSON only — no preamble, no markdown, no explanation.

{
  "befund_phrases": ["<descriptive phrase 1>", "<descriptive phrase 2>", ...],
  "beurteilung_phrases": ["<diagnostic phrase 1>", "<diagnostic phrase 2>", ...]
}

If a category yields no phrases, return an empty list for that key.
"""

USER_PROMPT_TEMPLATE_ENGLISH = """\
Read the following bone tumor radiology report and extract clinically relevant phrases, \
classified by content type — NOT by section header.

DEFINITIONS

befund_phrases — Descriptive observations (what the lesion looks like):
  Relevant: margin type, border pattern (Lodwick), cortical integrity, periosteal reaction,
  matrix/density (lytic/blastic/chondroid/ossified), lesion geometry (expansile, soft tissue),
  anatomical location (diaphysis/metaphysis/epiphysis), lesion size, soft tissue envelope
  status, host bone changes directly at the tumor site (subchondral sclerosis, cystic areas,
  trabecular changes), surface lesions and exostoses, skeletal deformity caused by tumor
  Examples: "osteolytic lesion", "chondroid matrix", "cortical breakthrough",
  "proximal tibial metaphysis", "approx. 4 cm", "sclerotic rim", "permeative pattern",
  "no periosteal reaction", "soft tissue extension", "endosteal scalloping",
  "unremarkable soft tissue envelope", "subchondral sclerosing areas",
  "cystic lucency areas", "gravel cysts", "stiletto-shaped exostoses",
  "metaphyseal widening", "valgus deviation ankle joint"

  Exclude from befund_phrases:
  - Degenerative/arthrotic changes at sites other than the tumor bone
  - Osteophytes, joint space narrowing, subchondral sclerosis at unrelated joints
  - Implant or clip material unrelated to the lesion
  - Bilateral symmetric findings that are degenerative or incidental, not tumor-related
  - Any finding at a distant anatomically unrelated site (e.g. BWS, pelvis)
    Exception: in multifocal tumor conditions, findings at multiple skeletal sites
    ARE relevant if they describe tumor manifestations, not incidental co-pathology

beurteilung_phrases — Diagnostic interpretations (what the lesion means):
  Relevant: tumor type, differential diagnoses, malignancy characterizations,
  benign/aggressive assessments
  Examples: "suspected enchondroma", "DDx low-grade chondrosarcoma",
  "consistent with benign process", "aggressive bone lesion",
  "no evidence of malignancy", "juvenile bone cyst",
  "hereditary multiple exostoses", "valgus-deviated ankle joint space"
  Exclude: clinical recommendations, follow-up plans, procedural suggestions
  (e.g. "biopsy recommended", "MRI advised", "follow-up in 6 months")

EXTRACTION RULES
1. Classify by CONTENT TYPE — ignore Befund/Beurteilung section headers entirely
2. Use original phrasing from the source text — no paraphrasing or invented content
3. Concise phrases only (2–8 words), one concept per phrase
4. Perilesional and host bone changes directly at the tumor site are ALWAYS relevant
5. Exclude findings at anatomically unrelated sites; exception: multifocal tumor
   manifestations across multiple skeletal sites are always in scope
6. A single sentence may produce phrases in BOTH categories
7. Both lists must contain AT LEAST one phrase
8. If the report's primary subject is a suspicious or indeterminate finding (even without
   confirmed tumor diagnosis), treat that finding and its host bone as the lesion site
9. Include negative findings ONLY when they negate a tumor-aggressive feature:
   e.g. "no periosteal reaction", "no cortical breakthrough", "no soft tissue extension",
   "no osteodestructive process". Exclude generic trauma/status negations.
10. Exclude ALL fracture-related phrases EXCEPT "pathological fracture" and
    "insufficiency fracture", which are direct indicators of tumor-related cortical
    destruction. Exclude: "no fracture", "no fracture detected", "no fracture noted",
    "no evidence of fracture", "fracture excluded", and all equivalent variants.

FALLBACK (if a category has no explicit content):
- befund: use the most prominent morphological or anatomical feature present
- beurteilung: use the closest tumor characterization available in the report

BEFORE ANSWERING CHECK:
- Does "befund_phrases" contain at least 1 descriptive observation? \
If not, extract the most prominent morphological or anatomical feature present.
- Does "beurteilung_phrases" contain at least 1 diagnostic interpretation? \
If none exists in the report, extract the closest clinical characterization available \
(e.g., a descriptive summary implying a clinical meaning).
- Only reply once both lists contain at least one entry.

EXAMPLES

Example 1 — standard report with both types:
  Findings: Proximal tibial metaphysis, approx. 4 cm osteolytic lesion with chondroid matrix.
  Impression: Suspected enchondroma. DDx low-grade chondrosarcoma. MRI recommended.
  → {{
       "befund_phrases": ["osteolytic lesion", "chondroid matrix", "approx. 4 cm",
                          "proximal tibial metaphysis"],
       "beurteilung_phrases": ["suspected enchondroma", "DDx low-grade chondrosarcoma"]
     }}

Example 2 — host bone and soft tissue findings present:
  Findings: Osteolytic lesion distal radius with subchondral sclerosing areas and cystic
  lucency. Unremarkable soft tissue envelope. No periosteal reaction.
  Impression: Consistent with giant cell tumor.
  → {{
       "befund_phrases": ["osteolytic lesion", "distal radius", "subchondral sclerosing areas",
                          "cystic lucency", "unremarkable soft tissue envelope",
                          "no periosteal reaction"],
       "beurteilung_phrases": ["consistent with giant cell tumor"]
     }}

Example 3 — distant incidental findings present (ignore them):
  Findings: Osteolytic lesion proximal humerus, cortical thinning, no soft tissue mass.
  Degenerative changes of the BWS. Osteophytic enlargements at the acetabular rims.
  Impression: Consistent with simple bone cyst. Follow-up recommended.
  → {{
       "befund_phrases": ["osteolytic lesion", "proximal humerus", "cortical thinning",
                          "no soft tissue mass"],
       "beurteilung_phrases": ["consistent with simple bone cyst"]
     }}

Example 4 — aggressive lesion with management statement (ignore management):
  Findings: Distal femur metaphysis: large osteolytic lesion with cortical destruction and
  soft tissue mass. Mild degenerative changes at the contralateral hip.
  Impression: Aggressive bone lesion, osteosarcoma suspected. Biopsy indicated.
  → {{
       "befund_phrases": ["large osteolytic lesion", "cortical destruction",
                          "soft tissue mass", "distal femur metaphysis"],
       "beurteilung_phrases": ["aggressive bone lesion", "osteosarcoma suspected"]
     }}

Example 5 — multifocal exostotic disease:
  Findings: Cartilaginous exostoses at distal tibia and fibula with characteristic
  metaphyseal widening. Clear valgus deviation of the ankle joint. Stiletto-shaped
  exostoses of distal femur. Right coxa valga due to deformity.
  Impression: Exostotic outgrowths consistent with hereditary multiple exostoses.
  Deformity of distal tibia, fibula, and ankle joint space.
  → {{
       "befund_phrases": ["cartilaginous exostoses distal tibia and fibula",
                          "metaphyseal widening", "valgus deviation ankle joint",
                          "stiletto-shaped exostoses distal femur",
                          "right coxa valga malposition"],
       "beurteilung_phrases": ["hereditary multiple exostoses",
                               "deformity of distal tibia and fibula",
                               "valgus-deviated ankle joint space"]
     }}

Report:
{formatted_report}

Reply ONLY with valid JSON — no preamble, no markdown, no explanation:
{{
  "befund_phrases": [],
  "beurteilung_phrases": []
}}"""