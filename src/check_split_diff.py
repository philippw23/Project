"""Check which instances differ between two split JSON files.

Reports image stems that appear in one split but not the other, both overall
and per split key (train / val / test).

Usage:
    python src/check_split_diff.py --split_a data/internal_dataset/split.json \\
                                   --split_b data/internal_dataset/split_clahe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT_A = ROOT_DIR / "data" / "internal_dataset" / "split.json"
DEFAULT_SPLIT_B = ROOT_DIR / "data" / "internal_dataset" / "split_clahe.json"

SPLIT_KEYS = ("train", "val", "test")


def load_split(path: Path) -> dict[str, list[dict]]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: raw[k] for k in SPLIT_KEYS if k in raw}


def stems(samples: list[dict]) -> set[str]:
    return {Path(s["image"]).stem for s in samples}


def report_diff(label: str, only_a: set[str], only_b: set[str], name_a: str, name_b: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    if only_a:
        print(f"  Only in {name_a} ({len(only_a)}):")
        for stem in sorted(only_a):
            print(f"    {stem}")
    else:
        print(f"  Nothing exclusive to {name_a}")

    if only_b:
        print(f"  Only in {name_b} ({len(only_b)}):")
        for stem in sorted(only_b):
            print(f"    {stem}")
    else:
        print(f"  Nothing exclusive to {name_b}")

    if not only_a and not only_b:
        print("  → Identical")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two split JSON files.")
    parser.add_argument("--split_a", default=str(DEFAULT_SPLIT_A))
    parser.add_argument("--split_b", default=str(DEFAULT_SPLIT_B))
    args = parser.parse_args()

    path_a = Path(args.split_a)
    path_b = Path(args.split_b)
    split_a = load_split(path_a)
    split_b = load_split(path_b)
    name_a  = path_a.name
    name_b  = path_b.name

    print(f"\nComparing splits:")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")

    all_a = {stem for key in SPLIT_KEYS for stem in stems(split_a.get(key, []))}
    all_b = {stem for key in SPLIT_KEYS for stem in stems(split_b.get(key, []))}
    print(f"\nTotal instances — A: {len(all_a)}, B: {len(all_b)}, shared: {len(all_a & all_b)}")

    report_diff("ALL SPLITS COMBINED", all_a - all_b, all_b - all_a, name_a, name_b)

    for key in SPLIT_KEYS:
        sa = stems(split_a.get(key, []))
        sb = stems(split_b.get(key, []))
        report_diff(
            f"{key.upper()}  (A: {len(sa)}, B: {len(sb)}, shared: {len(sa & sb)})",
            sa - sb, sb - sa, name_a, name_b,
        )

    print()


if __name__ == "__main__":
    main()
