"""Prompts for the separated extraction pipeline.

Each report section (befund / beurteilung) is queried with its own dedicated prompt,
giving the model a narrower, more focused task per call.

Three prompt templates are defined:
- BEFUND_PROMPT_TEMPLATE             — extracts descriptive observation phrases from the findings
- BEURTEILUNG_PROMPT_TEMPLATE        — extracts diagnostic statements from the impression
- BEURTEILUNG_SUMMARY_PROMPT_TEMPLATE — fallback used when the beurteilung section is absent
                                        or yields no phrases; derives a diagnosis from the befund

All output phrases are requested in English regardless of the input language.
"""

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
