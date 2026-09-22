"""German prompts for the joint extraction pipeline.

The model receives the full report (befund + beurteilung) in one user message and
is asked to return both phrase lists in a single JSON response — German input,
German output. Used by default (i.e. without --english).
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
  "proximale Tibia metaphysär", "sklerotischer Randsaum"

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
7. KEINE Größen- oder Maßangaben extrahieren (numerische Maße wie "ca. 4 cm", "35 mm", \
   "3 x 2 cm"); Maße sind kein zu extrahierendes Konzept. Qualitative Größen ("große Läsion", \
   "kleiner Herd") sind erlaubt.

VOR DEM ANTWORTEN PRÜFE:
- Enthält "befund_phrases" mindestens 1 Phrase? Wenn nein, extrahiere mindestens die auffälligste Läsion, Lokalisation oder Eigenschaft aus dem Befund.
- Enthält "beurteilung_phrases" mindestens 1 Phrase? Wenn nein, leite mindestens eine diagnostische Kurzbewertung aus dem Befund ab.
- Antworte erst, wenn beide Listen mindestens einen Eintrag enthalten.

Beispiel 1 (Beurteilung mit Diagnose):
BEFUND:
Proximale Tibia metaphysär ca. 4 cm große osteolytische Läsion mit chondroider Matrix.

BEURTEILUNG:
V.a. Enchondrom. DD niedriggradiges Chondrosarkom.

→ {{"befund_phrases": ["osteolytische Läsion", "chondroide Matrix",
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
