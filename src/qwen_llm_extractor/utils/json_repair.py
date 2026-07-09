"""Utilities for robustly parsing JSON produced by LLMs.

LLMs occasionally emit malformed JSON — unclosed markdown fences, missing commas,
invalid escape sequences, or wrong bracket types. The functions here apply a
sequence of heuristic repairs before falling back to a structured error dict.
"""

import json
import re


def _fix_invalid_escapes(s: str) -> str:
    """Replace backslashes not part of a valid JSON escape sequence with '\\\\'.

    LLMs writing German medical text sometimes emit bare backslashes (e.g. before
    units or German characters) that are illegal in JSON strings. This function
    walks the string character by character and doubles any backslash that is not
    followed by a recognised JSON escape character.

    Parameters
    ----------
    s : raw JSON-like string potentially containing invalid backslash escapes

    Returns
    -------
    str with all invalid backslash occurrences doubled so json.loads can parse them
    """
    valid_escapes = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}
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
    """Apply a sequence of best-effort repairs for common LLM JSON mistakes.

    Applied in order:
    1. Fix invalid backslash escapes (see _fix_invalid_escapes).
    2. Insert missing commas between adjacent string literals on consecutive lines.
    3. Insert missing commas after a closing bracket before the next key.
    4. Replace a closing ``}`` that should be ``]`` to terminate an array.

    Parameters
    ----------
    s : partially-broken JSON string

    Returns
    -------
    str that is more likely to be parseable by json.loads — not guaranteed
    """
    s = _fix_invalid_escapes(s)
    # LLMs sometimes omit commas between list items across lines: "foo"\n"bar" → "foo",\n"bar"
    s = re.sub(r'"\s*\n(\s*)"', '",\n\\1"', s)
    # Missing comma after a closing bracket before the next key: ]\n"key" → ],\n"key"
    s = re.sub(r'([}\]])\s*\n(\s*")', r'\1,\n\2', s)
    # LLM occasionally closes an array with } instead of ]: "item"\n  } → "item"\n  ]
    s = re.sub(r'(")\s*\n(\s+)\}(\s*\n\s*\})', r'\1\n\2]\3', s)
    return s


def _extract_json_objects(s: str) -> list[dict]:
    """Return all top-level JSON objects found in ``s``, in order.

    Uses ``raw_decode`` starting at each successive ``{`` so it tolerates leading /
    trailing prose AND multiple concatenated objects — which plain ``json.loads``
    rejects with "Extra data" when an LLM splits its answer into several objects
    (e.g. one per report section). Non-dict values are skipped.
    """
    decoder = json.JSONDecoder()
    objs: list[dict] = []
    i = 0
    while True:
        start = s.find("{", i)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(s, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(obj, dict):
            objs.append(obj)
        i = max(end, start + 1)
    return objs


def _merge_objects(objs: list[dict]) -> dict:
    """Merge several JSON objects into one, concatenating list-valued keys.

    When the model splits its output into multiple objects, their phrase lists
    (``phrases`` / ``befund_phrases`` / ``beurteilung_phrases``) are concatenated
    so nothing is dropped. Scalar keys keep the first value seen.
    """
    merged: dict = {}
    for o in objs:
        for k, v in o.items():
            if isinstance(v, list):
                if not isinstance(merged.get(k), list):
                    merged[k] = []
                merged[k].extend(v)
            else:
                merged.setdefault(k, v)
    return merged


def _parse_response(raw: str) -> dict:
    """Extract and parse JSON object(s) from a raw LLM response string.

    Handles common failure modes:
    - Markdown code fences (```json ... ```) wrapping the JSON
    - Invalid backslash escapes / missing commas (via _repair_json)
    - MULTIPLE concatenated objects (merged into one; fixes "Extra data" errors)
    - Extra prose before/after the JSON

    Parameters
    ----------
    raw : raw text output from the LLM (may contain fences, extra prose, etc.)

    Returns
    -------
    dict  — parsed (and merged, if the model emitted several objects) JSON on success
    dict  — ``{"error": "...", "raw": raw}`` on any parse failure, so callers can log
            the error without crashing the extraction loop
    """
    # Strip ```json ... ``` fences that the model sometimes wraps around its output
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$",          "", cleaned, flags=re.MULTILINE).strip()

    objs = _extract_json_objects(cleaned)
    if not objs:
        # Retry after best-effort repairs (invalid escapes, missing commas, ...)
        objs = _extract_json_objects(_repair_json(cleaned))
    if not objs:
        if "{" not in cleaned:
            return {"error": "No JSON found in response", "raw": raw}
        return {"error": "JSON parse error: no decodable object", "raw": raw}

    return objs[0] if len(objs) == 1 else _merge_objects(objs)
