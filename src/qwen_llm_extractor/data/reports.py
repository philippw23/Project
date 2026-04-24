import json
from pathlib import Path


def load_reports_joint(source: str, english: bool = False) -> tuple[list[str], list[str]]:
    """Load reports for joint extraction. Returns (reports, patids).

    Concatenates befund + beurteilung into a single string per report.
    english=True  → use befund_en / beurteilung_en (translated_reports.json)
    english=False → use befund / beurteilung        (sanitized_reports.json)
    """
    p = Path(source)
    if p.suffix != ".json":
        raise ValueError(f"Unsupported source: {source}")

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


def load_reports_separated(source: str) -> tuple[list[str], list[str], list[str]]:
    """Load reports for separated extraction. Returns (befunds, beurteilungs, patids).

    Keeps befund and beurteilung as separate parallel lists (German only).
    """
    p = Path(source)
    if p.suffix != ".json":
        raise ValueError(f"Unsupported source: {source}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON must be a list.")

    befunds, beurteilungs, patids = [], [], []
    for entry in data:
        befund      = (entry.get("befund") or "").strip()
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
