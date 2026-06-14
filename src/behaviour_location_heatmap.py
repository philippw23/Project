"""Heatmap of lesion behaviour x anatomical location.

Two separate heatmaps (one per dataset):
  internal — behaviour {benign, intermediate, malignant} x 16 body regions,
             restricted to the split (matched by file_name). Single-label.
  btxrd    — behaviour {benign, malignant} x fine-grained bone/joint columns,
             tumor images only. MULTI-LABEL: an image contributes to every
             location it is flagged with, so column sums exceed the image count.

Behaviour is on the y-axis, location on the x-axis (ordered by total frequency,
most common on the left). Cell colour = raw count (sequential blue, zeros white).

Usage:
    python src/behaviour_location_heatmap.py --dataset internal --save-dir results/dist
    python src/behaviour_location_heatmap.py --dataset btxrd    --save-dir results/dist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malignancy_distribution import (
    BTXRD_DEFAULT_XLSX,
    INTERNAL_DEFAULT_XLSX,
    INTERNAL_SHEET,
    INTERNAL_SPLIT,
    _load_btxrd,
    _load_internal,
)

BEHAVIOUR_ORDER = ["benign", "intermediate", "malignant"]

# Fine-grained BTXRD location columns (one-hot). wrist-joint is empty but kept
# so the axis matches the dataset schema.
BTXRD_FINE_LOCATIONS = [
    "hand", "ulna", "radius", "humerus", "foot", "tibia", "fibula", "femur",
    "hip bone", "ankle-joint", "knee-joint", "hip-joint", "wrist-joint",
    "elbow-joint", "shoulder-joint",
]

# Font sizes (presentation scale).
TITLE_FS, LABEL_FS, TICK_FS, CBAR_FS = 22, 18, 15, 16


def build_internal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """behaviour (rows) x localization (cols) raw-count matrix; single-label."""
    beh = df["malignancy"].astype(str).str.strip()
    loc = df["localization"].astype(str).str.strip()
    ct = pd.crosstab(beh, loc)
    rows = [b for b in BEHAVIOUR_ORDER if b in ct.index]
    return ct.reindex(index=rows)


def build_btxrd_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """behaviour (rows) x fine location (cols) raw-count matrix; multi-label."""
    beh = df["malignancy"].astype(str).str.strip()
    rows = [b for b in BEHAVIOUR_ORDER if b in set(beh)]  # benign, malignant
    mat = pd.DataFrame(0, index=rows, columns=BTXRD_FINE_LOCATIONS, dtype=int)
    for loc in BTXRD_FINE_LOCATIONS:
        present = df[loc] == 1
        for b in rows:
            mat.loc[b, loc] = int((present & (beh == b)).sum())

    multi = int((df[BTXRD_FINE_LOCATIONS].sum(axis=1) > 1).sum())
    print(f"BTXRD multi-label note: {multi} / {len(df)} tumor images flag >1 "
          f"location, so column sums exceed the image count.")
    return mat


def order_by_frequency(mat: pd.DataFrame) -> pd.DataFrame:
    """Sort columns by descending total count (most common on the left)."""
    totals = mat.sum(axis=0).sort_values(ascending=False)
    return mat.reindex(columns=totals.index)


def plot_heatmap(mat: pd.DataFrame, dataset: str, save_dir: Path | None) -> None:
    n_rows, n_cols = mat.shape
    fig_w = 0.85 * n_cols + 5.0
    fig_h = 1.05 * n_rows + 2.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Mask zeros so empty combinations render white.
    data = np.ma.masked_equal(mat.to_numpy(dtype=float), 0.0)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")

    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=TICK_FS)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(mat.index, fontsize=TICK_FS)
    ax.set_xlabel("Anatomical Location", fontsize=LABEL_FS)
    ax.set_ylabel("Lesion Behaviour", fontsize=LABEL_FS)
    ax.set_title(f"Lesion Behaviour by Anatomical Location ({dataset})",
                 fontsize=TITLE_FS)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Count", fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=CBAR_FS)

    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{dataset}_behaviour_location_heatmap.png"
        fig.savefig(out_path, dpi=150)
        print(f"Plot saved to: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["internal", "btxrd"], required=True)
    p.add_argument("--excel-path", default=None,
                   help="Override the source Excel file (defaults per --dataset).")
    p.add_argument("--sheet-name", default=INTERNAL_SHEET,
                   help="Internal worksheet name (ignored for BTXRD).")
    p.add_argument("--split-path", default=None,
                   help="internal only: split.json to restrict to. 'none' to "
                        f"disable. Default: {INTERNAL_SPLIT}")
    p.add_argument("--save-dir", default=None,
                   help="Directory to save the plot instead of displaying it.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.dataset == "internal":
            excel = Path(args.excel_path) if args.excel_path else INTERNAL_DEFAULT_XLSX
            if args.split_path is None:
                split_path = INTERNAL_SPLIT if INTERNAL_SPLIT.exists() else None
            elif args.split_path.lower() == "none":
                split_path = None
            else:
                split_path = Path(args.split_path)
            df = _load_internal(excel, args.sheet_name, split_path)
            mat = build_internal_matrix(df)
        else:
            excel = Path(args.excel_path) if args.excel_path else BTXRD_DEFAULT_XLSX
            df = _load_btxrd(excel, tumor_only=True)
            mat = build_btxrd_matrix(df)
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    mat = order_by_frequency(mat)

    print(f"\nLesion Behaviour x Anatomical Location ({args.dataset})\n")
    print(mat.to_string())
    print(f"\nMatrix total: {int(mat.to_numpy().sum())}")

    save_dir = Path(args.save_dir) if args.save_dir else None
    plot_heatmap(mat, args.dataset, save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
