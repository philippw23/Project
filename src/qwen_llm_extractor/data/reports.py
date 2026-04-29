"""Report loading utilities for the two extraction modes (joint and separated).

Both functions read from a JSON file that is expected to be a list of report dicts,
each with at minimum a ``patid`` and one or more of the following section fields:
``befund``, ``beurteilung``, ``befund_en``, ``beurteilung_en``.

A legacy format where the JSON is a plain list of strings (no metadata) is also
supported for backward compatibility; patids are returned as empty strings in that case.
"""

import json
from pathlib import Path


def load_reports_joint(source: str, english: bool = False) -> tuple[list[str], list[str]]:
    """Load reports for joint extraction by concatenating befund and beurteilung.

    Parameters
    ----------
    source  : path to a ``.json`` reports file
    english : if True, read ``befund_en`` / ``beurteilung_en`` (translated_reports.json);
              if False, read ``befund`` / ``beurteilung`` (sanitized_reports.json)

    Returns
    -------
    tuple[list[str], list[str]]
        ``(reports, patids)`` — parallel lists of the same length.
        Reports with no non-empty section fields are silently skipped.
    """
    p = Path(source)
    if p.suffix != ".json":
        raise ValueError(f"Unsupported source: {source}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON must be a list.")
    # Legacy format: plain list of strings (no patid metadata)
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
            # Sections are double-newline separated so the LLM sees them as distinct blocks
            reports.append("\n\n".join(parts))
            patids.append(entry.get("patid", ""))

    lang = "English" if english else "German"
    print(f"Loaded {len(reports)} {lang} reports from {p}.")
    return reports, patids


def load_reports_separated(source: str) -> tuple[list[str], list[str], list[str]]:
    """Load reports for separated extraction, keeping sections as parallel lists.

    Unlike ``load_reports_joint``, the two sections are NOT concatenated — they
    are returned as separate lists so each can be passed to its own LLM prompt.
    Only German fields (``befund`` / ``beurteilung``) are supported.

    Parameters
    ----------
    source : path to a ``.json`` reports file

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``(befunds, beurteilungs, patids)`` — three parallel lists of the same length.
        Reports where both sections are empty are skipped. An assertion guards against
        length mismatches introduced by future edits to the loop.
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
