"""Count entries in the report_accnr column (column G, index 6) of metadata.xlsx."""
import json
from pathlib import Path

import pandas as pd

ROOT_DIR          = Path(__file__).resolve().parent.parent
EXCEL_PATH        = ROOT_DIR / "data" / "internal_dataset" / "metadata.xlsx"
FULL_REPORTS_PATH = ROOT_DIR / "data" / "internal_dataset" / "text" / "full_reports.json"
REPORTS_PATH      = ROOT_DIR / "data" / "internal_dataset" / "text" / "reports.json"


def main() -> None:
    df = pd.read_excel(
        EXCEL_PATH,
        sheet_name="internal_data_matched",
        usecols=[6, 7],
        skiprows=1,
        header=None,
        nrows=1268,
        dtype=str,
        engine="openpyxl",
    )
    df[6] = df[6].fillna("").str.strip()
    df[7] = df[7].fillna("").str.strip()

    total     = len(df)
    non_empty = (df[6] != "").sum()
    empty     = total - non_empty
    unique    = df[6][df[6] != ""].nunique()

    print(f"Total rows          : {total}")
    print(f"  report_accnr non-empty : {non_empty}")
    print(f"  report_accnr empty     : {empty}")
    print(f"  Unique values (non-empty): {unique}")
    # Rows with no report_accnr — check if their patids are all unique
    no_accnr = df[df[6] == ""]
    no_accnr_patids = no_accnr[7].tolist()
    unique_patids = len(set(no_accnr_patids))
    duplicate_patids = [p for p in set(no_accnr_patids) if no_accnr_patids.count(p) > 1]
    print(f"  Rows with empty report_accnr : {len(no_accnr)}")
    print(f"    Unique patids              : {unique_patids}")
    if duplicate_patids:
        print(f"    Duplicate patids           : {duplicate_patids}")
    else:
        print(f"    All patids are unique      : True")
    print()

    accnr_list = df[6][df[6] != ""].tolist()

    def _check(path: Path, label: str) -> None:
        with open(path, encoding="utf-8") as fh:
            entries = json.load(fh)
        accnrs_in_file = {str(e.get("accnr", "")).strip() for e in entries}
        matched   = [a for a in accnr_list if a in accnrs_in_file]
        unmatched = [a for a in accnr_list if a not in accnrs_in_file]
        print(f"  Matched in {label:<25}: {len(matched)}")
        print(f"  Unmatched in {label:<23}: {len(unmatched)}")
        if unmatched:
            print(f"  Unmatched: {unmatched}")

    _check(FULL_REPORTS_PATH, "full_reports.json")
    _check(REPORTS_PATH,      "reports.json")

if __name__ == "__main__":
    main()
