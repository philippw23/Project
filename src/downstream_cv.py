"""K-fold cross-validation orchestrator for LACE downstream evaluation.

Trains the downstream head once per CV fold with a **fixed** set of
hyperparameters (the winning sweep config), evaluating each fold-model on both
its held-out internal `test` split and the frozen external **BTXRD** manifest.
Reports mean ± std across folds — the final generalization result.

All downstream hyperparameters are passed straight through to
`LACE.train.downstream` (same flags as `LACE/train/downstream.py`); this wrapper only
adds the fold loop and the aggregation. `--splits` and `--eval_test` are managed
per fold and must not be passed here.

Usage: one pretrain checkpoint per fold, from CV-mode pretraining, which
writes fold<N>/{split.json,best_retrieval_checkpoint.pt} under the run dir.
Point --cv_dir/--pattern at those fold<N> dirs; each fold's checkpoint is
picked up next to its split file (see --checkpoint_filename to override the
filename looked up):
    python src/downstream_cv.py \\
        --baseline lace \\
        --version v2 \\
        --cv_dir results/lace_v2_pretrain/run_.../ \\
        --pattern "fold*/split.json" \\
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
import os
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from biomedclip.data.datasets import (IDX_TO_LABEL, IDX_TO_LABEL_BINARY,
                                       NUM_CLASSES, NUM_CLASSES_BINARY)
from biomedclip.train.downstream import (main as run_biomedclip_downstream,
                                         parse_args as parse_biomedclip_downstream_args)
from biomedclip.train.img_text_downstream import (main as run_biomedclip_img_text_downstream,
                                                   parse_args as parse_biomedclip_img_text_downstream_args)
from chexfound.train.downstream import (main as run_chexfound_downstream,
                                        parse_args as parse_chexfound_downstream_args)
from gloria_downstream import (main as run_gloria_downstream,
                               parse_args as parse_gloria_downstream_args)
from imagenet_img.train.downstream import (main as run_imagenet_downstream,
                                           parse_args as parse_imagenet_downstream_args)

from biomedclip.utils.downstream_eval import report_eval
from LACE.train.downstream import (main as run_lace_downstream,
                                   parse_args as parse_lace_downstream_args)
from LACE.train.img_text_downstream import (main as run_lace_img_text_downstream,
                                            parse_args as parse_lace_img_text_downstream_args)

ROOT_DIR = Path(__file__).resolve().parent.parent

# Managed per fold — reject if the user passes them through.
# {"a", "b"} with no ":" is a *set* literal, not a dict (a dict would be {"a": 1}).
_RESERVED = {"--splits", "--eval_test", "--run_name", "--checkpoint"}

# Per-fold metrics summarized as the secondary (stability) block.
_PERFOLD = [
    "val/f1_macro", "val/balanced_acc", "val/acc", "val/auroc",
    "test/f1_macro", "test/balanced_acc", "test/acc", "test/auroc",
    "btxrd/f1_macro", "btxrd/balanced_acc", "btxrd/acc", "btxrd/auroc", "btxrd/loss",
]


def parse_args(argv=None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="K-fold CV orchestrator for LACE downstream (fixed hyperparameters).",
        add_help=True,
    )
    p.add_argument("--baseline", default="lace",
                   choices=["lace", "lace_img_text", "biomedclip", "biomedclip_img_text",
                            "chexfound", "gloria", "imagenet"],
                   help="Baseline for which downstream cv is done in a loop")
    p.add_argument("--frozen", action="store_true",
                   help="Frozen-encoder mode: no per-fold pretrain checkpoint is looked up or "
                        "injected (--cv_dir/--pattern just need to resolve to the raw fold split "
                        "files, e.g. data/internal_dataset/cv_binary). Pass whatever flag "
                        "selects the frozen encoder for --baseline through as normal: "
                        "--freezed_biomedclip for biomedclip, --checkpoint none "
                        "--chexfound_weights <path> for chexfound, --checkpoint <fixed ckpt> "
                        "for gloria (no vanilla-weights option). imagenet has no checkpoint "
                        "concept at all — always frozen ImageNet-1k weights, --frozen still "
                        "required so the orchestrator doesn't try to inject one.")
    p.add_argument("--cv_dir", default=str(ROOT_DIR / "data" / "internal_dataset" / "cv_binary"),
                   help="Directory containing the fold split files.")
    p.add_argument("--pattern", default="split_binary_fold*.json",
                   help="Glob for fold files inside --cv_dir (sorted).")
    p.add_argument("--summary_json", default=None,
                   help="Where to write the per-fold + aggregate metrics JSON "
                        "(default: <cv_dir>/cv_results.json).")
    p.add_argument("--checkpoint_filename", default="best_retrieval_checkpoint.pt",
                   help="Filename looked up next to each fold's split file, as "
                        "<split_path's dir>/<checkpoint_filename>. Matches the layout CV-mode "
                        "pretraining produces (one fold<N>/ dir per fold, holding both "
                        "split.json and the checkpoint) — point --cv_dir at that pretrain "
                        "run dir with --pattern 'fold*/split.json'.")
    # parse_known_args returns (Namespace, list): flags this parser recognizes
    # (--cv_dir, --pattern, ...) go into `known`; everything else — the
    # downstream-training hyperparameters this script doesn't define itself —
    # is left over as `passthrough`, a plain list of strings like
    # ["--binary", "--lr", "5e-5", ...], forwarded untouched to LACE.train.downstream.
    # known = flags this parser defines; passthrough = everything else, as a list of strings
    known, passthrough = p.parse_known_args(argv)
    return known, passthrough


def _aggregate(fold_results: list[dict]) -> dict:
    """Mean/std across folds for every metric present in *all* folds."""
    # Gives the keys of the first fold
    common = set(fold_results[0])
    # Loop over all folds and only keep keys that are in every fold, e.g test/f1_macro
    for r in fold_results[1:]:
        common &= set(r)
    # Aggregate the values for each key across folds, computing mean and std.
    agg = {}
    for key in sorted(common):
        vals = np.array([float(r[key]) for r in fold_results], dtype=float)
        agg[key] = {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}
    return agg


def main(argv=None) -> None:
    known, passthrough = parse_args(argv)

    # reserved flags the user passed through by mistake. In --frozen mode there is no per-fold
    # checkpoint for the orchestrator to inject, so --checkpoint is the caller's to set (e.g.
    # chexfound's --checkpoint none for its frozen baseline) rather than reserved.
    reserved = _RESERVED - {"--checkpoint"} if known.frozen else _RESERVED
    bad = reserved.intersection(passthrough)
    if bad:
        raise SystemExit(f"Do not pass {sorted(bad)} to the CV orchestrator — they are managed per fold.")

    # one split file per fold, sorted for a stable fold0, fold1, ... order
    cv_dir = Path(known.cv_dir)
    fold_files = sorted(cv_dir.glob(known.pattern))
    if not fold_files:
        raise SystemExit(f"No fold files matching {known.pattern!r} in {cv_dir}")
    print(f"Found {len(fold_files)} folds in {cv_dir}")

    # Determine whether this is a binary or multi-class downstream task, to pick the right label
    # mapping and number of classes. LACE's --binary is a bare store_true flag, but biomedclip's
    # --binary takes an explicit value (--binary true/false), so presence alone isn't enough —
    # check the following token too when there is one.
    if "--binary" in passthrough:
        i = passthrough.index("--binary")
        nxt = passthrough[i + 1] if i + 1 < len(passthrough) else ""
        binary = nxt.lower() not in ("false", "0", "no")
    else:
        binary = False
    idx_to_label = IDX_TO_LABEL_BINARY if binary else IDX_TO_LABEL
    num_classes  = NUM_CLASSES_BINARY if binary else NUM_CLASSES

    # Loop over folds, running the downstream evaluation for each fold's split and checkpoint.
    # Unique per invocation (SLURM job ID, else PID + start timestamp) so two CV runs sharing
    # --out_dir / --pattern never overwrite each other's per-fold checkpoint file.
    run_tag = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    fold_results: list[dict] = []
    oof_preds:  list[int] = []   # out-of-fold internal-test predictions (each sample once)
    oof_labels: list[int] = []
    oof_probs:  list[list[float]] = []
    for fold, split_path in enumerate(fold_files):
        print("\n" + "#" * 70)
        print(f"# FOLD {fold} — {split_path.name}")
        print("#" * 70)
        # Pass through the user-specified hyperparameters, plus the fold-specific flags.
        fold_argv = passthrough + [
            "--splits", str(split_path), "--eval_test",
            "--run_name", f"cv_fold{fold}_{split_path.stem}_{run_tag}",
        ]
        if known.frozen:
            print("  encoder: frozen (no per-fold checkpoint)")
        else:
            ckpt = split_path.parent / known.checkpoint_filename
            if not ckpt.exists():
                raise SystemExit(f"No per-fold checkpoint at {ckpt}")
            print(f"  checkpoint: {ckpt}")
            fold_argv += ["--checkpoint", str(ckpt)]
        # Parse the fold-specific args and run the downstream evaluation for this fold.
        if known.baseline == "lace":
            args = parse_lace_downstream_args(fold_argv)
            results = run_lace_downstream(args)
        elif known.baseline == "biomedclip":
            args = parse_biomedclip_downstream_args(fold_argv)
            results = run_biomedclip_downstream(args)
        elif known.baseline == "chexfound":
            args = parse_chexfound_downstream_args(fold_argv)
            results = run_chexfound_downstream(args)
        elif known.baseline == "biomedclip_img_text":
            args = parse_biomedclip_img_text_downstream_args(fold_argv)
            results = run_biomedclip_img_text_downstream(args)
        elif known.baseline == "lace_img_text":
            args = parse_lace_img_text_downstream_args(fold_argv)
            results = run_lace_img_text_downstream(args)
        elif known.baseline == "gloria":
            args = parse_gloria_downstream_args(fold_argv)
            results = run_gloria_downstream(args)
        elif known.baseline == "imagenet":
            args = parse_imagenet_downstream_args(fold_argv)
            results = run_imagenet_downstream(args)
        # peel off the raw per-sample predictions (kept out of scalar aggregation)
        oof_preds  += results.pop("test/_preds",  [])
        oof_labels += results.pop("test/_labels", [])
        oof_probs  += results.pop("test/_probs",  [])
        # drop non-numeric entries (e.g. {prefix}/roc_curve, a wandb CustomChart
        # object added by report_eval(..., use_wandb=True)) — not aggregable and
        # not JSON-serializable, so they can't go into fold_results/summary_json.
        results = {k: v for k, v in results.items()
                   if isinstance(v, (int, float, np.integer, np.floating))}
        fold_results.append(results)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    agg = _aggregate(fold_results)

    # ── Primary: pooled out-of-fold internal metrics ──────────────────────────
    pooled: dict = {}
    if oof_labels:
        pp = np.array(oof_preds)
        pl = np.array(oof_labels)
        pb = np.array(oof_probs)
        pooled = report_eval(
            "POOLED OUT-OF-FOLD TEST (primary — every internal sample scored once)",
            pp, pl, float("nan"), idx_to_label, num_classes, prefix="test_pooled", probs=pb,
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
    # Loop over the metrics we want to report per fold, printing the mean ± std and the individual fold values.
    for key in _PERFOLD:
        if key in agg:
            a = agg[key]
            per_fold = ", ".join(f"{v:.3f}" for v in a["values"])
            tag = "BTXRD external" if key.startswith("btxrd/") else "internal per-fold"
            print(f"  {key:<22} {a['mean']:.4f} ± {a['std']:.4f}   [{per_fold}]   ({tag})")
    # Write the summary JSON with the fold files, pooled OOF metrics, per-fold metrics, and aggregate metrics.
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
