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

SYSTEM_PROMPT_ENGLISH_BACKUP = """\
You are an experienced radiologist specializing in structured medical information extraction.

Your task: read a radiology report and classify all medically relevant phrases into two categories:

- befund_phrases: DESCRIPTIVE observations — morphology, location, size, matrix, margins,
  cortical status, periosteal reaction (what the lesion looks like)
- beurteilung_phrases: DIAGNOSTIC interpretations — diagnoses, differentials, clinical
  assessments, recommendations, benign/malignant characterizations (what the lesion means)

Section headers (Findings/Impression) may overlap, be absent, or mix both types.
Ignore section boundaries — classify every phrase by its CONTENT TYPE only.
Work close to the source text. No hallucinations. Reply with valid JSON only.
"""

USER_PROMPT_TEMPLATE_ENGLISH_BACKUP = """\
Read the following radiology report and extract all medically relevant phrases, \
classified by content type — not by section.

Definitions:
- "befund_phrases": Descriptive observations about the lesion
  What it looks like: morphology, location, size, matrix type, margin characteristics,
  cortical integrity, periosteal reaction, soft tissue involvement
  Examples: "osteolytic lesion", "chondroid matrix", "cortical breakthrough",
  "proximal tibial metaphysis", "approx. 4 cm diameter", "sclerotic rim",
  "well-defined margins", "no periosteal reaction", "soft tissue involvement"

- "beurteilung_phrases": Diagnostic interpretations and clinical assessments
  What it means: diagnoses, differential diagnoses, benign/malignant characterizations,
  clinical conclusions, recommendations
  Examples: "suspected enchondroma", "DDx low-grade chondrosarcoma",
  "consistent with benign process", "radiologically unremarkable",
  "biopsy indicated", "follow-up recommended"

Rules:
1. Classify by content type — ignore section headers entirely
2. Use original phrasing from the text (no paraphrasing or interpretation)
3. Preserve anatomical details and measurements
4. Concise phrases (2–8 words), one concept per phrase
5. Extract ALL relevant statements from the entire report
6. AT LEAST one phrase per category
7. NO hallucinations or invented information not in the text

BEFORE ANSWERING CHECK:
- Does "befund_phrases" contain at least 1 descriptive observation? \
If not, extract the most prominent morphological or anatomical feature present.
- Does "beurteilung_phrases" contain at least 1 diagnostic interpretation? \
If none exists in the report, extract the closest clinical characterization available \
(e.g., a descriptive summary implying a clinical meaning).
- Only reply once both lists contain at least one entry.

Example 1 (standard report with both types present):
FINDINGS:
Proximal tibial metaphysis approx. 4 cm osteolytic lesion with chondroid matrix.
IMPRESSION:
Suspected enchondroma. DDx low-grade chondrosarcoma.

→ {{"befund_phrases": ["osteolytic lesion", "chondroid matrix", "approx. 4 cm lesion",
                       "proximal tibial metaphysis"],
    "beurteilung_phrases": ["suspected enchondroma", "DDx low-grade chondrosarcoma"]}}

Example 2 (impression mixes descriptive and diagnostic content):
FINDINGS:
Rounded osteolysis with sclerotic rim in the proximal phalanx shaft of finger III left.
IMPRESSION:
Rim-sclerotic osteolysis in the proximal phalanx shaft, consistent with enchondroma.

→ {{"befund_phrases": ["rounded osteolysis", "sclerotic rim",
                       "proximal phalanx shaft finger III left",
                       "rim-sclerotic osteolysis"],
    "beurteilung_phrases": ["consistent with enchondroma"]}}

Example 3 (no explicit impression, all content is findings):
FINDINGS:
Distal femoral metaphysis small osteolytic lesion, sharply marginated, no cortical \
breakthrough, no periosteal reaction.

→ {{"befund_phrases": ["small osteolytic lesion", "sharply marginated",
                       "distal femoral metaphysis", "no cortical breakthrough",
                       "no periosteal reaction"],
    "beurteilung_phrases": ["sharply marginated lesion without cortical involvement"]}}

Report:
{formatted_report}

Reply ONLY with this JSON:
{{
  "befund_phrases": [],
  "beurteilung_phrases": []
}}"""

SYSTEM_PROMPT_ENGLISH = """\
You are an experienced radiologist specializing in structured medical information extraction \
for bone tumor radiology reports.

Your task: read a radiology report and extract ONLY phrases that are relevant to \
bone tumor characterization. Classify them into two categories:

- befund_phrases: DESCRIPTIVE observations about the tumor — its morphology, location,
  size, matrix, margins, cortical status, periosteal reaction, soft tissue involvement
  (what the lesion looks like)
- beurteilung_phrases: DIAGNOSTIC interpretations — diagnoses, differentials, clinical
  assessments, recommendations, benign/malignant characterizations (what the lesion means)

Focus exclusively on features relevant to bone tumor assessment. Ignore incidental findings,
patient history unrelated to the tumor, and post-operative context unless it describes
residual tumor features.

The following radiological descriptor categories define what is considered relevant:
  MARGIN & BORDER: sclerotic/well-defined margin, geographic border (Lodwick), permeative/
    moth-eaten/infiltrative pattern, cortical destruction/breakthrough, endosteal scalloping
  PERIOSTEAL REACTION: any periosteal reaction, sunburst/spiculated pattern,
    Codman triangle, lamellar/onion-skin periosteal reaction
  MATRIX & DENSITY: osteolytic/radiolucent, osteoblastic/sclerotic/radiodense,
    mixed lytic-blastic, chondroid matrix (rings and arcs), ossified/mineralized matrix
  LESION GEOMETRY: expansile lesion/bone expansion, soft tissue mass/extraosseous extension,
    epiphyseal involvement/growth plate, diaphyseal location, metaphyseal location
  HOST BONE RESPONSE: pathological/insufficiency fracture, bone remodeling/trabecular changes
  DIAGNOSIS: tumor type, differential diagnoses, malignancy assessment, biopsy/follow-up

Section headers may overlap or be absent. Classify every phrase by CONTENT TYPE only.
Work close to the source text. No hallucinations. Reply with valid JSON only.
"""

USER_PROMPT_TEMPLATE_ENGLISH = """\
Read the following bone tumor radiology report and extract phrases relevant to tumor \
characterization, classified by content type — not by section.

Definitions:
- "befund_phrases": Descriptive observations about the bone tumor
  Focus on: margin type, border pattern, cortical integrity, periosteal reaction,
  matrix/density (lytic/blastic/chondroid/ossified), lesion geometry (expansile, soft tissue),
  anatomical location (diaphysis/metaphysis/epiphysis), size
  Examples: "osteolytic lesion", "chondroid matrix", "cortical breakthrough",
  "proximal tibial metaphysis", "approx. 4 cm diameter", "sclerotic rim",
  "permeative pattern", "no periosteal reaction", "soft tissue extension",
  "endosteal scalloping", "expansile lesion", "geographic border"

- "beurteilung_phrases": Diagnostic interpretations and clinical assessments
  What it means: diagnoses, differential diagnoses, benign/malignant characterizations,
  clinical conclusions, recommendations
  Examples: "suspected enchondroma", "DDx low-grade chondrosarcoma",
  "consistent with benign process", "no evidence of malignancy",
  "biopsy indicated", "follow-up recommended"

Rules:
1. Extract ONLY tumor-relevant features — skip incidental findings and unrelated history
2. Classify by content type — ignore section headers entirely
3. Use original phrasing from the text (no paraphrasing or interpretation)
4. Preserve anatomical details and measurements
5. Concise phrases (2–8 words), one concept per phrase
6. AT LEAST one phrase per category
7. NO hallucinations or invented information not in the text

BEFORE ANSWERING CHECK:
- Does "befund_phrases" contain at least 1 tumor-relevant descriptive observation?
  If not, extract the most prominent morphological or anatomical feature present.
- Does "beurteilung_phrases" contain at least 1 diagnostic interpretation?
  If none exists in the report, extract the closest clinical characterization available.
- Only reply once both lists contain at least one entry.

Example 1 (standard report with both types present):
FINDINGS:
Proximal tibial metaphysis approx. 4 cm osteolytic lesion with chondroid matrix.
IMPRESSION:
Suspected enchondroma. DDx low-grade chondrosarcoma.

→ {{"befund_phrases": ["osteolytic lesion", "chondroid matrix", "approx. 4 cm lesion",
                       "proximal tibial metaphysis"],
    "beurteilung_phrases": ["suspected enchondroma", "DDx low-grade chondrosarcoma"]}}

Example 2 (impression mixes descriptive and diagnostic content):
FINDINGS:
Rounded osteolysis with sclerotic rim in the proximal phalanx shaft of finger III left.
IMPRESSION:
Rim-sclerotic osteolysis in the proximal phalanx shaft, consistent with enchondroma.

→ {{"befund_phrases": ["rounded osteolysis", "sclerotic rim",
                       "proximal phalanx shaft finger III left",
                       "rim-sclerotic osteolysis"],
    "beurteilung_phrases": ["consistent with enchondroma"]}}

Example 3 (no explicit impression, all content is findings):
FINDINGS:
Distal femoral metaphysis small osteolytic lesion, sharply marginated, no cortical \
breakthrough, no periosteal reaction.

→ {{"befund_phrases": ["small osteolytic lesion", "sharply marginated",
                       "distal femoral metaphysis", "no cortical breakthrough",
                       "no periosteal reaction"],
    "beurteilung_phrases": ["sharply marginated lesion without cortical involvement"]}}

Example 4 (report with incidental findings to ignore):
FINDINGS:
Distal femur metaphysis: large osteolytic lesion with cortical destruction and soft tissue
mass. Mild degenerative changes at the knee joint. Patient has known hypertension.
IMPRESSION:
Aggressive bone lesion, cortical breakthrough with extraosseous extension. Osteosarcoma
suspected. Biopsy indicated.

→ {{"befund_phrases": ["large osteolytic lesion", "cortical destruction", "soft tissue mass",
                       "distal femur metaphysis"],
    "beurteilung_phrases": ["aggressive bone lesion", "cortical breakthrough",
                            "extraosseous extension", "osteosarcoma suspected",
                            "biopsy indicated"]}}

Report:
{formatted_report}

Reply ONLY with this JSON:
{{
  "befund_phrases": [],
  "beurteilung_phrases": []
}}"""