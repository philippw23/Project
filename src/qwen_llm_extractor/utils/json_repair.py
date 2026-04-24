import json
import re


def _fix_invalid_escapes(s: str) -> str:
    """Replace backslashes not part of a valid JSON escape with double backslash."""
    valid_escapes = set('"\\\/bfnrtu')
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
    """Best-effort repairs for common LLM JSON mistakes."""
    s = _fix_invalid_escapes(s)
    # Add missing comma between a closing quote and an opening quote on the next line
    s = re.sub(r'"\s*\n(\s*)"', '",\n\\1"', s)
    # Add missing comma between ] or } and the next key
    s = re.sub(r'([}\]])\s*\n(\s*")', r'\1,\n\2', s)
    # Fix } used instead of ] to close an array
    s = re.sub(r'(")\s*\n(\s+)\}(\s*\n\s*\})', r'\1\n\2]\3', s)
    return s


def _parse_response(raw: str) -> dict:
    """Extract JSON from LLM response, robust to markdown fences and invalid escapes."""
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
