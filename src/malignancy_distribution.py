import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ── Dataset defaults ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
INTERNAL_DEFAULT_XLSX = ROOT / "data" / "internal_dataset" / "metadata.xlsx"
INTERNAL_SHEET        = "internal_data_matched"
INTERNAL_SPLIT        = ROOT / "data" / "internal_dataset" / "split.json"
BTXRD_DEFAULT_XLSX    = ROOT / "data" / "BTXRD" / "dataset.xlsx"

# Per-field display config. `column` is the source column name in the respective
# sheet; `order` (optional) fixes the category order in tables/plots.
FIELD_CONFIG = {
    "entity":       {"label": "Entity"},
    "malignancy":   {"label": "Lesion Behaviour",
                     "order": ["benign", "intermediate", "malignant"]},
    "localization": {"label": "Localization"},
    "age":          {"label": "Age"},
    "sex":          {"label": "Sex"},
}

# Field -> source column name, per dataset.
INTERNAL_COLUMNS = {
    "entity": "entity", "malignancy": "malignancy", "localization": "localization",
    "age": "age", "sex": "SEX",
}
BTXRD_COLUMNS = {
    # `malignancy` is derived from one-hot columns (see _load_btxrd).
    "malignancy": "malignancy", "age": "age", "sex": "gender",
}


def build_age_bins(age_data: pd.Series) -> tuple[list[int], list[str]]:
    min_age = int(age_data.min())
    max_age = int(age_data.max())
    start = (min_age // 10) * 10
    end = ((max_age // 10) + 1) * 10
    edges = list(range(start, end + 1, 10))
    labels = [f"{bin_start}-{bin_start + 9}" for bin_start in edges[:-1]]
    return edges, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distribution of a dataset field for the internal split or BTXRD."
    )
    parser.add_argument(
        "--dataset", choices=["internal", "btxrd"], default="internal",
        help="Which dataset to analyze (default: internal).",
    )
    parser.add_argument(
        "--excel-path", default=None,
        help="Override the source Excel file (defaults per --dataset).",
    )
    parser.add_argument(
        "--sheet-name", default=INTERNAL_SHEET,
        help="Worksheet name for the internal dataset (ignored for BTXRD).",
    )
    parser.add_argument(
        "--split-path", default=None,
        help="internal only: path to split.json. When given, only samples in the "
             "split are counted (matched by file name). Pass 'none' to disable. "
             f"Default: {INTERNAL_SPLIT}",
    )
    parser.add_argument(
        "--btxrd-tumor-only", action="store_true", default=True,
        help="BTXRD only: count only tumor images (drop normal/no-tumor). Default: on.",
    )
    parser.add_argument(
        "--btxrd-include-normal", dest="btxrd_tumor_only", action="store_false",
        help="BTXRD only: also include normal (no-tumor) images as a category.",
    )
    parser.add_argument(
        "--field", required=True, choices=FIELD_CONFIG.keys(),
        help="Dataset field to analyze.",
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory to save the plot instead of displaying it.",
    )
    return parser.parse_args()


def _split_file_names(split_path: Path) -> set[str]:
    """Return the set of image file names present in the split (train+val+test)."""
    with open(split_path, encoding="utf-8") as fh:
        split = json.load(fh)
    names: set[str] = set()
    for part in ("train", "val", "test"):
        for sample in split.get(part, []):
            names.add(Path(sample["image"]).name)
    return names


def _load_internal(excel_path: Path, sheet_name: str | None,
                   split_path: Path | None) -> pd.DataFrame:
    try:
        df = pd.read_excel(
            excel_path, sheet_name=sheet_name or 0, engine="openpyxl"
        )
    except FileNotFoundError as exc:
        raise ValueError(f"Excel file not found: {excel_path}") from exc
    except Exception as exc:  # missing sheet, parse errors, …
        raise ValueError(f"Unable to read '{excel_path}': {exc}") from exc

    if split_path is not None:
        split_files = _split_file_names(split_path)
        before = len(df)
        df = df[df["file_name"].astype(str).isin(split_files)]
        print(f"Restricted to split: {len(df)} / {before} rows "
              f"({len(split_files)} split images).")
    return df


def _load_btxrd(excel_path: Path, tumor_only: bool) -> pd.DataFrame:
    try:
        df = pd.read_excel(excel_path, sheet_name=0, engine="openpyxl")
    except FileNotFoundError as exc:
        raise ValueError(f"Excel file not found: {excel_path}") from exc
    except Exception as exc:
        raise ValueError(f"Unable to read '{excel_path}': {exc}") from exc

    # Lesion behaviour from the one-hot benign/malignant columns.
    behaviour = pd.Series("normal", index=df.index, dtype=object)
    behaviour[df["benign"] == 1] = "benign"
    behaviour[df["malignant"] == 1] = "malignant"
    df = df.assign(malignancy=behaviour)

    if tumor_only:
        before = len(df)
        df = df[df["tumor"] == 1]
        print(f"BTXRD tumor-only: {len(df)} / {before} rows "
              f"(dropped {before - len(df)} normal images).")
    return df


def get_field_series(args: argparse.Namespace) -> pd.Series:
    """Load the requested dataset/field as a raw pandas Series."""
    if args.dataset == "internal":
        excel_path = Path(args.excel_path) if args.excel_path else INTERNAL_DEFAULT_XLSX
        if args.split_path is None:
            split_path = INTERNAL_SPLIT if INTERNAL_SPLIT.exists() else None
        elif args.split_path.lower() == "none":
            split_path = None
        else:
            split_path = Path(args.split_path)
        df = _load_internal(excel_path, args.sheet_name, split_path)
        columns = INTERNAL_COLUMNS
    else:  # btxrd
        excel_path = Path(args.excel_path) if args.excel_path else BTXRD_DEFAULT_XLSX
        df = _load_btxrd(excel_path, args.btxrd_tumor_only)
        columns = BTXRD_COLUMNS

    if args.field not in columns:
        raise ValueError(
            f"Field '{args.field}' is not available for dataset '{args.dataset}'. "
            f"Available: {', '.join(sorted(columns))}."
        )
    return df[columns[args.field]]


def compute_counts(raw_series: pd.Series, field_name: str):
    """Return counts (and, for age, the numeric series too)."""
    if field_name == "age":
        numeric = pd.to_numeric(raw_series, errors="coerce").dropna()
        if numeric.empty:
            raise ValueError("No valid numeric age values found.")
        edges, labels = build_age_bins(numeric)
        bins = pd.cut(numeric, bins=edges, labels=labels,
                      include_lowest=True, right=False)
        counts = bins.value_counts(sort=False)
        return numeric, counts[counts > 0]

    series = raw_series.dropna().astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        raise ValueError(f"No valid '{field_name}' values found.")

    counts = series.value_counts()
    order = FIELD_CONFIG[field_name].get("order")
    if order:
        present = [c for c in order if c in counts.index]
        present += [c for c in counts.index if c not in order]  # keep any extras
        counts = counts.reindex(present)
    else:
        counts = counts.sort_values(ascending=False)
    return None, counts


def print_distribution(counts: pd.Series, field_name: str,
                       age_data: pd.Series | None = None) -> None:
    field = FIELD_CONFIG[field_name]
    axis_name = field["label"].lower().replace(" ", "_")
    print(f"\n{field['label']} Distribution\n")
    print(counts.rename_axis(axis_name).to_frame("count").to_string())
    print(f"\nTotal: {int(counts.sum())}")
    if field_name == "age" and age_data is not None and not age_data.empty:
        print(f"Youngest: {int(age_data.min())}  Oldest: {int(age_data.max())}")


def plot_distribution(counts: pd.Series, field_name: str, dataset: str,
                      age_data: pd.Series | None = None,
                      save_dir: Path | None = None) -> None:
    field = FIELD_CONFIG[field_name]
    title = f"{field['label']} Distribution ({dataset})"
    plt.figure(figsize=(12, 7))

    TITLE_FS, LABEL_FS, TICK_FS, VALUE_FS = 24, 20, 18, 20

    if field_name == "age":
        if age_data is None or age_data.empty:
            raise ValueError("No age data available for plotting.")
        edges, _ = build_age_bins(age_data)
        ax = age_data.plot.hist(bins=edges, alpha=0.5, edgecolor="black")
        plt.xlabel(field["label"], fontsize=LABEL_FS); plt.xticks(edges)
        for patch, value in zip(ax.patches, counts.tolist()):
            ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height(),
                    str(value), ha="center", va="bottom", fontsize=VALUE_FS)
    else:
        ax = counts.plot(kind="bar", alpha=0.5, edgecolor="black")
        plt.xlabel(field["label"], fontsize=LABEL_FS); plt.xticks(rotation=0, ha="center")
        # add headroom so the value label above the tallest bar isn't clipped
        ax.set_ylim(top=ax.get_ylim()[1] * 1.08)
        for index, value in enumerate(counts):
            ax.text(index, value, str(value), ha="center", va="bottom", fontsize=VALUE_FS)

    plt.title(title, fontsize=TITLE_FS)
    plt.ylabel("Count", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{dataset}_{field_name}_distribution.png"
        plt.savefig(out_path, dpi=150)
        print(f"Plot saved to: {out_path}")
    else:
        plt.show()
    plt.close()


def main() -> int:
    args = parse_args()
    try:
        raw_series = get_field_series(args)
        age_data, counts = compute_counts(raw_series, args.field)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    save_dir = Path(args.save_dir) if args.save_dir else None
    print_distribution(counts, args.field, age_data=age_data)
    plot_distribution(counts, args.field, args.dataset,
                      age_data=age_data, save_dir=save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
