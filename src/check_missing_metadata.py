"""Report patient IDs with missing age, sex, or malignancy label in the Excel metadata."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from biomedclip.utils.misc import DEFAULT_EXCEL

def main(excel_path: Path = DEFAULT_EXCEL) -> None:
    df = pd.read_excel(
        excel_path,
        sheet_name="internal_data_matched",
        usecols=[0, 2, 4, 5, 7],
        skiprows=1,
        header=0,
        dtype=str,
        engine="openpyxl",
    )
    df.columns = ["filename", "malignancy", "age", "sex", "patid"]
    df = df.dropna(subset=["filename", "patid"])
    df["filename"]   = df["filename"].str.strip()
    df["patid"]      = df["patid"].str.strip()
    df["malignancy"] = df["malignancy"].fillna("").str.strip().str.lower()
    df["age"]        = df["age"].fillna("")
    df["sex"]        = df["sex"].fillna("")

    def _norm_id(val: str) -> str:
        try:
            return str(int(float(val)))
        except ValueError:
            return val

    df["patid"] = df["patid"].apply(_norm_id)

    def _valid_age(v: str) -> bool:
        try:
            float(v)
            return True
        except (ValueError, TypeError):
            return False

    def _valid_sex(v: str) -> bool:
        return str(v).strip().lower() in ("m", "male", "1", "f", "female", "0")

    bad_age   = ~df["age"].apply(_valid_age)
    bad_sex   = ~df["sex"].apply(_valid_sex)
    bad_label = df["malignancy"] == ""

    any_bad = bad_age | bad_sex | bad_label
    missing = df[any_bad].copy()

    missing["missing_fields"] = (
        bad_age[any_bad].map({True: "age"  , False: ""}) + " " +
        bad_sex[any_bad].map({True: "sex"  , False: ""}) + " " +
        bad_label[any_bad].map({True: "label", False: ""})
    ).str.split().apply(lambda x: ", ".join(x))

    print(f"Total rows (filename+patid present): {len(df)}")
    print(f"  Missing age  : {bad_age.sum()}")
    print(f"  Missing sex  : {bad_sex.sum()}")
    print(f"  Missing label: {bad_label.sum()}")
    print(f"  Any missing  : {any_bad.sum()}\n")

    if missing.empty:
        print("No rows with missing metadata.")
        return

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print(missing[["patid", "filename", "age", "sex", "malignancy", "missing_fields"]].to_string(index=False))


if __name__ == "__main__":
    main()
