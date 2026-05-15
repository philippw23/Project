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
        usecols=[6],
        skiprows=1,
        header=None,
        nrows=1268,
        dtype=str,
        engine="openpyxl",
    )
    df[6] = df[6].fillna("").str.strip()

    total     = len(df)
    non_empty = (df[6] != "").sum()
    empty     = total - non_empty
    unique    = df[6][df[6] != ""].nunique()

    print(f"Total rows          : {total}")
    print(f"  report_accnr non-empty : {non_empty}")
    print(f"  report_accnr empty     : {empty}")
    print(f"  Unique values (non-empty): {unique}")
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
