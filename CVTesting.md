# Cross-Validation Testing

Notes on how the 10-fold cross-validation methodology works end to end: fold
creation, CV-mode pretraining (per baseline), and the downstream CV
orchestrator that trains/evaluates a head per fold and aggregates results.
Full design rationale (metric choice, why 10 folds, OOF pooling) lives in
[src/LACE/CV_EVAL_PLAN.md](src/LACE/CV_EVAL_PLAN.md); this file is the
practical "how it fits together across baselines" reference.

---

## 1. Fold creation

[src/create_cv_splits.py](src/create_cv_splits.py) turns one existing
`train`/`val`/`test` split manifest (see [SplitCreation.md](SplitCreation.md))
into a pool of **10 fold files** with no patient leakage:

- The given `train` split is divided into **8** stratified, patient-grouped
  parts via `StratifiedGroupKFold` (grouped by `patid`, stratified by
  `label`). The given `val` and `test` splits are kept as-is and become parts
  9 and 10 — reusing the already-carved-out val/test rather than
  re-stratifying them.
- Fold `i`: `test` = part `i`, `val` = the part preceding it, `train` = the
  remaining 8 parts. The last fold (`test`=part 10, `val`=part 9) reproduces
  the original split exactly.
- Every fold is asserted patient-disjoint across train/val/test.
- Output: `split_binary_fold{0..9}.json` (naming is historical — used for
  both binary and 3-class pools) under `--out_dir`, each with the same
  sample-dict schema as the input manifest.

Two pools exist in practice, generated from the two split manifests:
`data/internal_dataset/cv/` (3-class, from `split_final.json`) and
`data/internal_dataset/cv_binary/` (binary, from a binary split manifest).
Wrapper: [sbatch/data/run_create_cv_splits.sh](sbatch/data/run_create_cv_splits.sh).

`data/BTXRD/` is a separate, fixed **external** test set (built once by
[src/data/build_btxrd_downstream.py](src/data/build_btxrd_downstream.py)) — it is never
folded, and is excluded from all pretraining so it stays an unbiased
generalization check.

---

## 2. CV-mode pretraining

Only the contrastive/self-supervised baselines have a pretraining stage to
run per fold; frozen/no-pretrain baselines (ImageNet, GLoRIA's fixed
CheXpert checkpoint) skip straight to downstream CV (§3) against the raw fold
pool.

Baselines with CV-mode pretraining: **LACE v2**
([src/LACE/train/pretrain_v2.py](src/LACE/train/pretrain_v2.py)), **BiomedCLIP**
([src/biomedclip/train/pretrain.py](src/biomedclip/train/pretrain.py)), and **CheXFound**
([src/chexfound/train/pretrain.py](src/chexfound/train/pretrain.py)). All three
share the same shape:

- `--cv_dir <dir>` + `--cv_pattern <glob>` (default `split_binary_fold*.json`)
  run one full pretraining pass **per fold file** instead of a single run,
  writing results under `run_<timestamp>/fold0/`, `fold1/`, ... each holding
  its own checkpoint (e.g. `best_retrieval_checkpoint.pt` for LACE) trained
  only on that fold's `train`/`val`.
- Mutually exclusive with `--splits` (single-run mode).
- Only `train`/`val` from each fold file are touched during pretraining — the
  fold's `test` split is never seen by pretraining, only by downstream CV
  eval (§3).
- Each fold is trained back-to-back **within a single SLURM job**, so job
  `--time` must be bumped to roughly `N_folds × (single-run time budget)`.

Wrappers: [sbatch/lace/run_lace_pretrain_v2.sh](sbatch/lace/run_lace_pretrain_v2.sh)
(`CV_DIR`/`CV_PATTERN` vars, mutually exclusive with `SPLITS`),
[sbatch/biomedclip/run_biomedclip_pretrain.sh](sbatch/biomedclip/run_biomedclip_pretrain.sh)
(same pattern), [sbatch/chexfound/run_chexfound_pretrain_cv.sh](sbatch/chexfound/run_chexfound_pretrain_cv.sh).

---

## 3. Downstream CV orchestrator

[src/downstream_cv.py](src/downstream_cv.py) is the shared entry point for
downstream CV across **all** baselines (`--baseline
lace|biomedclip|biomedclip_img_text|chexfound|gloria|imagenet`) — it trains
one downstream head per fold with a **fixed** set of hyperparameters (the
winning sweep config from [PretrainSweeps.md](PretrainSweeps.md)-adjacent
downstream sweeps — no per-fold re-tuning) and aggregates the results.

*(There is also an older, LACE-only version of this,
[src/lace_downstream_cv.py](src/lace_downstream_cv.py), superseded by the
generic `downstream_cv.py`.)*

### 3.1 Per-fold flow

For each fold split file matching `--cv_dir`/`--pattern` (sorted for a stable
`fold0, fold1, ...` order):

- **Checkpoint lookup** (skipped in `--frozen` mode — see below): the
  checkpoint is expected **next to** the fold's split file, i.e.
  `<split_path's dir>/<checkpoint_filename>`. This matches the CV-mode
  pretraining layout (`run_dir/fold<N>/{split.json, checkpoint}`) — point
  `--cv_dir`/`--pattern` at the pretrain run's `fold*/split.json` files
  directly.
- The fold-specific `--splits`, `--eval_test`, `--run_name`, and (unless
  `--frozen`) `--checkpoint` flags are injected automatically; passing any of
  these through yourself raises an error (`_RESERVED`) — everything else is
  forwarded verbatim to the baseline's own `<name>_downstream.py` args.
- The downstream head is trained/early-stopped on the fold's `train`/`val`
  (on `val_loss`, unchanged from single-split training), then scored on the
  fold's held-out `test` split and, if `--btxrd_manifest` is set, on the full
  external BTXRD manifest too.

**`--frozen` mode**: for baselines with no per-fold pretrain checkpoint to
inject — ImageNet (always vanilla frozen weights), GLoRIA (one fixed
CheXpert checkpoint shared across all folds, passed via `--checkpoint`
directly since it isn't fold-managed), or BiomedCLIP run
`--freezed_biomedclip`. In this mode `--cv_dir` points straight at the raw
fold-split pool (`data/internal_dataset/cv[_binary]/`) rather than a pretrain
run's `fold<N>/` dirs, since there's no per-fold checkpoint to sit next to.

### 3.2 Metric selection (why `val/f1_macro`, not `val_loss`, across runs)

- **Within a fold**: checkpoint selection / early stopping still use
  `val_loss` — inside one run, loss confounds (`focal_gamma`, class
  weighting, ...) are fixed, so it stays a valid, smooth, monotone signal.
- **Across sweep runs / for reporting**: `val/f1_macro` (argmax-based) is
  used instead, since it's invariant to those confounds and matches the
  metric reported on held-out test. The winning sweep config is picked
  manually from W&B, ranked by `val/f1_macro`, tie-broken by
  `val/balanced_acc`.
- Sweep runs never touch `test` or BTXRD (`--eval_test` defaults off) — the
  best config is chosen blind to both, so the eventual CV numbers stay an
  unbiased generalization estimate.

### 3.3 Aggregation & reporting

Two complementary views, both written to `cv_results.json` (`--summary_json`,
default `<cv_dir>/cv_results.json`):

- **Primary — pooled out-of-fold (OOF) internal metrics**
  (`pooled_oof_internal`, prefix `test_pooled/`): every fold's `test`
  predictions are concatenated into one vector spanning the *entire* dataset
  (each sample scored exactly once, by a model that never trained on it),
  then macro-F1/balanced-acc/accuracy are computed once over the pooled set.
  Preferred over averaging 10 small per-fold F1 scores because macro-F1 is
  nonlinear and per-fold sample counts are small/noisy; pooling is not
  contamination since each prediction is still out-of-fold.
- **Secondary — per-fold mean ± std** (`aggregate`, `per_fold`): stability
  indicator across folds for `test/f1_macro`, `test/balanced_acc`,
  `test/acc`, and, when `--btxrd_manifest` is set, `btxrd/f1_macro`,
  `btxrd/balanced_acc`, `btxrd/acc`, `btxrd/loss`.
- **BTXRD** (external, fixed test set) is *only* ever reported as mean ± std
  over the 10 fold-heads — it can't be pooled OOF since every fold scores the
  *same* BTXRD samples, not a disjoint slice. Ensembling the 10 fold-heads
  (soft-voting) was considered but not pursued.

### 3.4 Wrappers

One `run_<baseline>_downstream_cv.sh` per baseline under `sbatch/<baseline>/`
(e.g. [sbatch/lace/run_lace_downstream_cv_full.sh](sbatch/lace/run_lace_downstream_cv_full.sh),
[sbatch/biomedclip/run_biomedclip_downstream_cv.sh](sbatch/biomedclip/run_biomedclip_downstream_cv.sh),
[sbatch/chexfound/run_chexfound_downstream_cv.sh](sbatch/chexfound/run_chexfound_downstream_cv.sh),
[sbatch/gloria/run_gloria_downstream_cv.sh](sbatch/gloria/run_gloria_downstream_cv.sh) /
`run_gloria_downstream_cv_binary.sh`,
[sbatch/imagenet_img/run_imagenet_img_downstream_cv.sh](sbatch/imagenet_img/run_imagenet_img_downstream_cv.sh) /
`run_imagenet_img_downstream_cv_binary.sh`,
[sbatch/biomedclip/run_biomedclip_img_text_downstream_cv.sh](sbatch/biomedclip/run_biomedclip_img_text_downstream_cv.sh)).
Each pins the fixed downstream hyperparameters (winning sweep config) in a
`# ── Fixed hyperparameters ──` block, sets `--baseline`/`--frozen`/`CV_DIR`/
`PATTERN`/`CHECKPOINT_FILENAME` appropriately for that encoder, and forwards
everything to `src/downstream_cv.py`.
