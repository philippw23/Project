"""Filter reports.json and write a sanitized version without unusable findings.

Usage:
    python src/data/preprocess_reports.py [--input PATH] [--output PATH]
"""
import argparse
import json
from pathlib import Path

DEFAULT_INPUT  = Path(__file__).parent.parent.parent / "data" / "internal_dataset" / "text" / "reports.json"
DEFAULT_OUTPUT = Path(__file__).parent.parent.parent / "data" / "internal_dataset" / "text" / "sanitized_reports.json"

PLACEHOLDER = "Die Bilder wurden bereitgestellt."


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  default=str(DEFAULT_INPUT),
                        help="Path to reports.json (default: %(default)s)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Path to write sanitized_reports.json (default: %(default)s)")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    with open(args.input, "r", encoding="utf-8") as f:
        reports = json.load(f)

    sanitized = [
        r for r in reports
        if r.get("befund", "").strip() and PLACEHOLDER not in r["befund"]
    ]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, ensure_ascii=False, indent=2)

    removed = len(reports) - len(sanitized)
    print(f"Input reports:    {len(reports)}")
    print(f"Removed:          {removed}")
    print(f"Sanitized reports:{len(sanitized)}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main(parse_args())
