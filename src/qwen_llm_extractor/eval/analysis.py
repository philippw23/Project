import pandas as pd

CATEGORIES = [
    "befund_phrases",
    "beurteilung_phrases",
]


def flatten_to_dataframe(results: list[dict], patids: list[str]) -> pd.DataFrame:
    """One row per (patid, report_idx, category, phrase)."""
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
            if isinstance(value, str):
                phrases = [p.strip() for p in value.split(";") if p.strip()]
            elif isinstance(value, list):
                phrases = [str(p).strip() for p in value if str(p).strip()]
            else:
                phrases = []

            if not phrases:
                rows.append({"patid": patid, "report_idx": i, "category": cat, "phrase": "not reported"})
            else:
                for phrase in phrases:
                    rows.append({"patid": patid, "report_idx": i, "category": cat, "phrase": phrase.lower()})

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
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


def compare_with_medbert(llm_df: pd.DataFrame, medbert_csv: str) -> None:
    mb = pd.read_csv(medbert_csv)

    llm_terms = set(llm_df[llm_df["category"] != "error"]["phrase"].str.lower())
    mb_terms  = set(mb["phrase"].str.lower())

    overlap  = llm_terms & mb_terms
    llm_only = llm_terms - mb_terms
    mb_only  = mb_terms  - llm_terms

    print("\n" + "=" * 65)
    print("COMPARISON: LLM  vs  medbert/KeyBERT")
    print("=" * 65)
    print(f"  LLM unique phrases    : {len(llm_terms)}")
    print(f"  medbert unique phrases: {len(mb_terms)}")
    print(f"  Overlap               : {len(overlap)}"
          f"  ({100*len(overlap)/max(len(llm_terms),1):.0f}% of LLM terms)")
    print(f"  Only in LLM           : {len(llm_only)}")
    print(f"  Only in medbert       : {len(mb_only)}")

    print("\nTop 15 LLM-only phrases (not found by medbert):")
    freq_llm = llm_df[llm_df["category"] != "error"].groupby("phrase")["report_idx"].nunique()
    for t in freq_llm.reindex(sorted(llm_only)).nlargest(15).index:
        print(f"  [{freq_llm[t]:>3}×]  {t}")

    print("\nTop 15 medbert-only phrases (not found by LLM):")
    freq_mb = mb.groupby("phrase")["report_idx"].nunique()
    for t in freq_mb.reindex(sorted(mb_only)).nlargest(15).index:
        print(f"  [{freq_mb[t]:>3}×]  {t}")
