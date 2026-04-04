import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


START_ROW = 2
END_ROW = 1269
ROW_COUNT = END_ROW - START_ROW + 1
FIELD_CONFIG = {
    "entity": {"column": "B", "label": "Entity"},
    "malignancy": {"column": "C", "label": "Malignancy Type"},
    "localization": {"column": "D", "label": "Localization"},
    "age": {"column": "E", "label": "Age"},
    "sex": {"column": "F", "label": "Sex"},
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
        description="Display the distribution of malignancy types from an Excel file."
    )
    parser.add_argument(
        "--excel-path",
        required=True,
        help="Path to the Excel file containing malignancy types in column C.",
    )
    parser.add_argument(
        "--sheet-name",
        help="Optional worksheet name. If omitted, the first sheet is used.",
        default="internal_data_matched",
    )
    parser.add_argument(
        "--field",
        required=True,
        choices=FIELD_CONFIG.keys(),
        help="Dataset field to analyze.",
    )
    return parser.parse_args()


def load_distribution_counts(
    excel_path: Path, sheet_name: str | None, field_name: str
):
    field = FIELD_CONFIG[field_name]

    try:
        data = pd.read_excel(
            excel_path,
            sheet_name=sheet_name if sheet_name else 0,
            usecols=field["column"],
            skiprows=START_ROW - 1,
            nrows=ROW_COUNT,
            engine="openpyxl",
        )
    except FileNotFoundError as exc:
        raise ValueError(f"Excel file not found: {excel_path}") from exc
    except ValueError as exc:
        # pandas uses ValueError for several read_excel issues, including missing sheets.
        raise ValueError(f"Unable to read worksheet: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Unable to read Excel file '{excel_path}': {exc}") from exc

    raw_series = data.iloc[:, 0]

    if field_name == "age":
        numeric_series = pd.to_numeric(raw_series, errors="coerce").dropna()

        if numeric_series.empty:
            raise ValueError(
                "No valid numeric age values were found in column E for rows 2 through 1269."
            )

        age_bin_edges, age_bin_labels = build_age_bins(numeric_series)
        age_bins = pd.cut(
            numeric_series,
            bins=age_bin_edges,
            labels=age_bin_labels,
            include_lowest=True,
            right=False,
        )
        counts = age_bins.value_counts(sort=False)
        counts = counts[counts > 0]
        return numeric_series, counts

    series = raw_series.dropna().astype(str).str.strip()
    series = series[series != ""]

    if series.empty:
        raise ValueError(
            f"No valid {field['label'].lower()} values were found in column "
            f"{field['column']} for rows 2 through 1269."
        )

    return series.value_counts().sort_values(ascending=False)


def print_distribution(
    counts: pd.Series, field_name: str, age_data: pd.Series | None = None
) -> None:
    field = FIELD_CONFIG[field_name]
    axis_name = field["label"].lower().replace(" ", "_")
    print(f"\n{field['label']} Distribution\n")
    print(counts.rename_axis(axis_name).to_frame("count").to_string())
    if field_name == "age" and age_data is not None and not age_data.empty:
        print(f"\nYoungest person: {int(age_data.min())}")
        print(f"Oldest person: {int(age_data.max())}")


def plot_distribution(
    counts: pd.Series, field_name: str, age_data: pd.Series | None = None
) -> None:
    field = FIELD_CONFIG[field_name]
    plt.figure(figsize=(12, 6))

    if field_name == "age":
        if age_data is None or age_data.empty:
            raise ValueError("No age data available for plotting.")

        age_bin_edges, _ = build_age_bins(age_data)
        ax = age_data.plot.hist(bins=age_bin_edges, alpha=0.5, edgecolor="black")
        plt.title(f"{field['label']} Distribution")
        plt.xlabel(field["label"])
        plt.ylabel("Count")
        plt.xticks(age_bin_edges)
        for patch, value in zip(ax.patches, counts.tolist()):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height(),
                str(value),
                ha="center",
                va="bottom",
            )
        plt.tight_layout()
        plt.show()
        return

    ax = counts.plot(kind="bar", alpha=0.5, edgecolor="black")
    plt.title(f"{field['label']} Distribution")
    plt.xlabel(field["label"])
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    for index, value in enumerate(counts):
        ax.text(index, value, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


def main() -> int:
    args = parse_args()
    excel_path = Path(args.excel_path)
    age_data = None

    try:
        result = load_distribution_counts(
            excel_path=excel_path,
            sheet_name=args.sheet_name,
            field_name=args.field,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.field == "age":
        age_data, counts = result
    else:
        counts = result

    print_distribution(counts, args.field, age_data=age_data)
    plot_distribution(counts, args.field, age_data=age_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
