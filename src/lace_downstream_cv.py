"""K-fold cross-validation orchestrator for LACE downstream evaluation.

Trains the downstream head once per CV fold with a **fixed** set of
hyperparameters (the winning sweep config), evaluating each fold-model on both
its held-out internal `test` split and the frozen external **BTXRD** manifest.
Reports mean ± std across folds — the final generalization result.

All downstream hyperparameters are passed straight through to
`LACE.train.downstream` (same flags as `lace_downstream.py`); this wrapper only
adds the fold loop and the aggregation. `--splits` and `--eval_test` are managed
per fold and must not be passed here.

Usage:
    python src/lace_downstream_cv.py \\
        --version v2 \\
        --checkpoint results/lace_v2_pretrain/run_.../best_retrieval_checkpoint.pt \\
        --cv_dir data/internal_dataset/cv \\
        --btxrd_manifest data/BTXRD/btxrd_downstream_binary.json \\
        --binary --use_mask \\
        --head mlp_no_meta --downstream_visual_mode fg --image_size 224 \\
        --lr 6.35e-05 --dropout 0.3 --weight_decay 0.5 \\
        --loss focal --focal_gamma 2.99 --class_weighting effective \\
        --hidden_dims 256 --batch_size 64 --epochs 100 --patience 10 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from biomedclip.data.datasets import (IDX_TO_LABEL, IDX_TO_LABEL_BINARY,
                                       NUM_CLASSES, NUM_CLASSES_BINARY)
from LACE.train.downstream import (main as run_downstream,
                                   parse_args as parse_downstream_args,
                                   _report_eval)

ROOT_DIR = Path(__file__).resolve().parent.parent

# Managed per fold — reject if the user passes them through.
_RESERVED = {"--splits", "--eval_test", "--run_name"}

# Per-fold metrics summarized as the secondary (stability) block.
_PERFOLD = [
    "test/f1_macro", "test/balanced_acc", "test/acc",
    "btxrd/f1_macro", "btxrd/balanced_acc", "btxrd/acc", "btxrd/loss",
]


def parse_args(argv=None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="K-fold CV orchestrator for LACE downstream (fixed hyperparameters).",
        add_help=True,
    )
    p.add_argument("--cv_dir", default=str(ROOT_DIR / "data" / "internal_dataset" / "cv"),
                   help="Directory containing the fold split files.")
    p.add_argument("--pattern", default="split_binary_fold*.json",
                   help="Glob for fold files inside --cv_dir (sorted).")
    p.add_argument("--summary_json", default=None,
                   help="Where to write the per-fold + aggregate metrics JSON "
                        "(default: <cv_dir>/cv_results.json).")
    known, passthrough = p.parse_known_args(argv)
    return known, passthrough


def _aggregate(fold_results: list[dict]) -> dict:
    """Mean/std across folds for every metric present in *all* folds."""
    common = set(fold_results[0])
    for r in fold_results[1:]:
        common &= set(r)
    agg = {}
    for key in sorted(common):
        vals = np.array([float(r[key]) for r in fold_results], dtype=float)
        agg[key] = {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}
    return agg


def main(argv=None) -> None:
    known, passthrough = parse_args(argv)

    bad = _RESERVED.intersection(passthrough)
    if bad:
        raise SystemExit(f"Do not pass {sorted(bad)} to the CV orchestrator — they are managed per fold.")

    cv_dir = Path(known.cv_dir)
    fold_files = sorted(cv_dir.glob(known.pattern))
    if not fold_files:
        raise SystemExit(f"No fold files matching {known.pattern!r} in {cv_dir}")
    print(f"Found {len(fold_files)} folds in {cv_dir}")

    binary       = "--binary" in passthrough
    idx_to_label = IDX_TO_LABEL_BINARY if binary else IDX_TO_LABEL
    num_classes  = NUM_CLASSES_BINARY if binary else NUM_CLASSES

    fold_results: list[dict] = []
    oof_preds:  list[int] = []   # out-of-fold internal-test predictions (each sample once)
    oof_labels: list[int] = []
    for fold, split_path in enumerate(fold_files):
        print("\n" + "#" * 70)
        print(f"# FOLD {fold} — {split_path.name}")
        print("#" * 70)
        fold_argv = passthrough + [
            "--splits", str(split_path), "--eval_test",
            "--run_name", f"cv_{split_path.stem}",
        ]
        args = parse_downstream_args(fold_argv)
        results = run_downstream(args)
        # peel off the raw per-sample predictions (kept out of scalar aggregation)
        oof_preds  += results.pop("test/_preds",  [])
        oof_labels += results.pop("test/_labels", [])
        fold_results.append(results)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    agg = _aggregate(fold_results)

    # ── Primary: pooled out-of-fold internal metrics ──────────────────────────
    pooled: dict = {}
    if oof_labels:
        pp = np.array(oof_preds)
        pl = np.array(oof_labels)
        pooled = _report_eval(
            "POOLED OUT-OF-FOLD TEST (primary — every internal sample scored once)",
            pp, pl, float("nan"), idx_to_label, num_classes, prefix="test_pooled",
        )
        pooled.pop("test_pooled/loss", None)  # no single loss across folds
        pooled["test_pooled/n"] = int(len(pl))

    # ── Secondary: per-fold stability + external BTXRD ────────────────────────
    print("\n" + "=" * 70)
    print(f"CROSS-VALIDATION SUMMARY  ({len(fold_files)} folds)")
    print("=" * 70)
    if pooled:
        print("Pooled OOF internal (primary):")
        print(f"  {'macro-F1':<16} {pooled['test_pooled/f1_macro']:.4f}   (n={pooled['test_pooled/n']})")
        print(f"  {'balanced acc':<16} {pooled['test_pooled/balanced_acc']:.4f}")
        print(f"  {'accuracy':<16} {pooled['test_pooled/acc']:.4f}")
    print("\nPer-fold (secondary, mean ± std across folds):")
    for key in _PERFOLD:
        if key in agg:
            a = agg[key]
            per_fold = ", ".join(f"{v:.3f}" for v in a["values"])
            tag = "BTXRD external" if key.startswith("btxrd/") else "internal per-fold"
            print(f"  {key:<22} {a['mean']:.4f} ± {a['std']:.4f}   [{per_fold}]   ({tag})")

    summary_path = Path(known.summary_json) if known.summary_json else cv_dir / "cv_results.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({
            "n_folds": len(fold_files),
            "fold_files": [str(f) for f in fold_files],
            "pooled_oof_internal": pooled,
            "per_fold": fold_results,
            "aggregate": agg,
        }, fh, indent=2)
    print(f"\nWrote CV results to {summary_path}")


if __name__ == "__main__":
    main()
