"""Plot a cosine-similarity heatmap of the 21-dim binary descriptor vectors,
sorted by malignancy label (benign / intermediate / malignant).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DESCRIPTORS = ROOT_DIR / "data" / "internal_dataset" / "text" / "descriptor_vectorsv2.json"
DEFAULT_DATASET     = ROOT_DIR / "data" / "internal_dataset" / "dataset_full.json"
DEFAULT_OUT         = ROOT_DIR / "results" / "descriptor_similarity_heatmap.png"

LABEL_ORDER = {"benign": 0, "intermediate": 1, "malignant": 2}


def main(args: argparse.Namespace) -> None:
    with open(args.descriptors, encoding="utf-8") as fh:
        desc = json.load(fh)
    with open(args.dataset, encoding="utf-8") as fh:
        dataset = json.load(fh)

    samples = [(s["image"], s["label"]) for s in dataset if s["image"] in desc]
    samples.sort(key=lambda x: LABEL_ORDER[x[1]])

    imgs   = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    vecs   = np.array([desc[img] for img in imgs], dtype=np.float32)

    # L2-normalise; zero vectors (no features extracted) produce 0 similarity
    norms      = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms_safe = np.where(norms == 0, 1.0, norms)
    vecs_norm  = vecs / norms_safe

    sim = vecs_norm @ vecs_norm.T   # [N, N] cosine similarity

    counts     = [labels.count(l) for l in ["benign", "intermediate", "malignant"]]
    boundaries = [0, counts[0], counts[0] + counts[1], sum(counts)]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=1.0,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=1.5)
        ax.axvline(b - 0.5, color="black", linewidth=1.5)

    tick_pos    = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(3)]
    tick_labels = [
        f"Benign\n(n={counts[0]})",
        f"Interm.\n(n={counts[1]})",
        f"Malignant\n(n={counts[2]})",
    ]
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_title(
        "Descriptor Vector Cosine Similarity\n(sorted by label, zero-norm=0)",
        fontsize=13,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", default=str(DEFAULT_DESCRIPTORS))
    parser.add_argument("--dataset",     default=str(DEFAULT_DATASET))
    parser.add_argument("--out",         default=str(DEFAULT_OUT))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
