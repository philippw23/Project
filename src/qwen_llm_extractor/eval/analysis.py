"""Evaluation and analysis utilities for LLM extraction results.

Provides two main functions:
- ``flatten_to_dataframe`` — converts the raw list-of-dicts output into a tidy DataFrame
- ``print_summary``        — prints per-category phrase statistics to stdout
"""

import pandas as pd

CATEGORIES = [
    "befund_phrases",
    "beurteilung_phrases",
]


def flatten_to_dataframe(results: list[dict], patids: list[str]) -> pd.DataFrame:
    """Convert raw LLM extraction results into a tidy DataFrame.

    Each row represents one extracted phrase for one report. Reports that
    produced a parse error get a single row with ``category="error"``.
    Categories that yielded no phrases get a single ``phrase="not reported"``
    placeholder row so every report/category combination is always present.

    Parameters
    ----------
    results : list of dicts returned by ``query_llm``, one per report.
              Expected keys: ``befund_phrases``, ``beurteilung_phrases``, ``patid``.
              May also contain an ``"error"`` key for failed parses.
    patids  : parallel list of patient IDs (used as a fallback if not in result dict)

    Returns
    -------
    pd.DataFrame with columns: ``patid``, ``report_idx``, ``category``, ``phrase``
    """
    rows = []
    for i, result in enumerate(results):
        patid = patids[i] if i < len(patids) else ""
        if "error" in result:
            rows.append({
                "patid":      patid,
                "report_idx": i,
                "category":   "error",
                "phrase":     result.get("error", "unknown"),
            })
            continue

        for cat in CATEGORIES:
            value = result.get(cat, [])
            # The LLM sometimes returns a semicolon-delimited string instead of a JSON array
            if isinstance(value, str):
                phrases = [p.strip() for p in value.split(";") if p.strip()]
            elif isinstance(value, list):
                phrases = [str(p).strip() for p in value if str(p).strip()]
            else:
                phrases = []

            if not phrases:
                rows.append({
                    "patid": patid, "report_idx": i, "category": cat, "phrase": "not reported",
                })
            else:
                for phrase in phrases:
                    # Lowercase for case-insensitive downstream comparison and deduplication
                    rows.append({
                        "patid": patid, "report_idx": i, "category": cat, "phrase": phrase.lower(),
                    })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    """Print per-category phrase counts and top-10 most frequent phrases to stdout."""
    print("\n" + "=" * 65)
    print("LLM EXTRACTION SUMMARY")
    print("=" * 65)

    if df.empty or "category" not in df.columns:
        print("No data extracted.")
        return

    errors = df[df["category"] == "error"]
    if not errors.empty:
        print(f"\nFailed reports : {errors['report_idx'].nunique()}")

    ok = df[(df["category"] != "error") & (df["phrase"] != "not reported")]
    if ok.empty:
        print("No phrases extracted.")
        return

    n_reports = ok["report_idx"].nunique()
    print(f"\nTotal phrases  : {len(ok)}")
    print(f"Unique phrases : {ok['phrase'].nunique()}")
    print(f"Reports OK     : {n_reports}")
    print(f"Avg per report : {len(ok) / n_reports:.1f}")

    for cat in CATEGORIES:
        grp = ok[ok["category"] == cat]
        print(f"\n── {cat} ({len(grp)} phrases, {grp['phrase'].nunique()} unique) ──")
        top = grp["phrase"].value_counts().head(10)
        for phrase, count in top.items():
            print(f"  [{count:>3}×]  {phrase}")
