"""Print per-class statistics for a descriptor_vectors.json file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DESCRIPTORS = ROOT_DIR / "data" / "internal_dataset" / "text" / "descriptor_vectors.json"
DEFAULT_DATASET     = ROOT_DIR / "data" / "internal_dataset" / "dataset_full.json"

DESCRIPTOR_KEYS = [
    "sclerotic_margin", "geographic_border", "permeative_pattern",
    "cortical_destruction", "endosteal_scalloping", "periosteal_reaction",
    "sunburst_periosteal", "codman_triangle", "lamellar_periosteal",
    "osteolytic", "osteoblastic", "mixed_lytic_blastic", "chondroid_matrix",
    "ossified_matrix", "expansile", "soft_tissue_extension",
    "epiphyseal_involvement", "diaphyseal_location", "metaphyseal_location",
    "pathological_fracture", "bone_remodeling",
]


def analyze(vecs: np.ndarray, label: str) -> None:
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    if len(vecs) == 0:
        print(f"{label:<14}: no samples")
        return
    n_active = vecs.sum(axis=1)
    zeros    = int((n_active == 0).sum())
    print(f"{label:<14}: mean active={n_active.mean():.2f}  std={n_active.std():.2f}  zeros={zeros}/{len(vecs)}")


def main(args: argparse.Namespace) -> None:
    with open(args.descriptors, encoding="utf-8") as fh:
        desc = json.load(fh)
    with open(args.dataset, encoding="utf-8") as fh:
        dataset = json.load(fh)

    # Only keep samples with valid (list) entries
    valid = {img: v for img, v in desc.items() if isinstance(v, list)}
    print(f"Descriptor file : {args.descriptors}")
    print(f"Valid vectors   : {len(valid)}/{len(desc)}\n")

    by_label: dict[str, list] = {"benign": [], "intermediate": [], "malignant": []}
    for sample in dataset:
        img = sample["image"]
        if img in valid:
            by_label[sample["label"]].append(valid[img])

    all_vecs = np.array(list(valid.values()), dtype=np.float32)

    print("── Per-class ──────────────────────────────────────────")
    for label in ["benign", "intermediate", "malignant"]:
        vecs = np.array(by_label[label], dtype=np.float32)
        analyze(vecs, label)

    print()
    analyze(all_vecs, "overall")

    print("\n── Per-feature prevalence (% of samples with feature=1) ──")
    prev = all_vecs.mean(axis=0) * 100
    for name, p in sorted(zip(DESCRIPTOR_KEYS, prev), key=lambda x: -x[1]):
        bar = "█" * int(p / 2)
        print(f"  {name:<30} {p:5.1f}%  {bar}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", default=str(DEFAULT_DESCRIPTORS))
    parser.add_argument("--dataset",     default=str(DEFAULT_DATASET))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
