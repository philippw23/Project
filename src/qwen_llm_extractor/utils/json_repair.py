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


def _parse_response(raw: str) -> dict:
    """Extract and parse the first JSON object from a raw LLM response string.

    Handles three common failure modes:
    - Markdown code fences (```json ... ```) wrapping the JSON
    - Invalid backslash escapes in string values
    - Missing commas or wrong bracket types (via _repair_json)

    Parameters
    ----------
    raw : raw text output from the LLM (may contain fences, extra prose, etc.)

    Returns
    -------
    dict  — parsed JSON on success
    dict  — ``{"error": "...", "raw": raw}`` on any parse failure, so callers
            can log the error without crashing the extraction loop
    """
    # Strip ```json ... ``` fences that the model sometimes wraps around its output
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$",          "", cleaned, flags=re.MULTILINE).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {"error": "No JSON found in response", "raw": raw}

    json_str = match.group()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return json.loads(_repair_json(json_str))
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "raw": raw}
