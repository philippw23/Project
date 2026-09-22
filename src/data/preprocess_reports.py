"""Filter reports.json and write a sanitized version without unusable findings."""
import json
from pathlib import Path

INPUT_PATH = Path(__file__).parent.parent.parent / "data" / "text" / "reports.json"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "text" / "sanitized_reports.json"

PLACEHOLDER = "Die Bilder wurden bereitgestellt."

with open(INPUT_PATH, 'r', encoding="utf-8") as f:
    reports = json.load(f)

sanitized = [
    r for r in reports
    if r.get("befund", "").strip() and PLACEHOLDER not in r["befund"]
]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(sanitized, f, ensure_ascii=False, indent=2)

removed = len(reports) - len(sanitized)
print(f"Input reports:    {len(reports)}")
print(f"Removed:          {removed}")
print(f"Sanitized reports:{len(sanitized)}")
print(f"Output written to: {OUTPUT_PATH}")
