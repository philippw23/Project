# Downstream Evaluation Plan: Sweep Metric + K-Fold CV + BTXRD External Test

Status: **implemented** (2026-07-09). This file records the methodology decisions
(agreed during design) and the concrete implementation steps. Verified: BTXRD
manifest builds (1867 samples, masks aligned), 5 CV folds generate with no
patient leakage, downstream imports + per-fold arg wiring work. Not yet run:
full SLURM training (needs GPU + pretrain checkpoint).

---

## 1. Methodology decisions (settled)

### 1.1 Two different jobs, two different metrics
- **Within-run** (which epoch's checkpoint to keep, when to early-stop):
  keep using **`val_loss`**. Inside a single run the loss confounds
  (`focal_gamma`, `class_weighting`, `label_smoothing`, `batch_size`) are all
  **fixed**, so `val_loss` is a smooth, valid, monotone-ish signal. It is
  continuous (reflects confidence), unlike macro-F1 which is discrete/noisy on a
  ~96-sample val set. **No change** to checkpoint/early-stopping logic.
- **Cross-run** (which sweep config wins) and **reporting**:
  use **`val/f1_macro`** (maximize). It is computed from argmax predictions, so it
  is **invariant** to `focal_gamma` / class weighting / label smoothing / batch
  size — exactly the confounds that make raw `val_loss` **not comparable across
  sweep runs**. It also matches the metric we report on test.

  Rejected alternative: ranking runs by *val_loss improvement since epoch 1*
  (Δ = loss₁ − loss_best). The additive smoothing offset cancels but the
  multiplicative `focal_gamma` scaling does **not**, so the delta is still
  γ-tainted; worse, "biggest improvement" rewards the run that *started worst*,
  not the one that *ends best*. Discarded.

### 1.2 Sweep → best config selection
- Sweep `metric` becomes `val/f1_macro`, goal `maximize`.
- Sweep runs **do not touch** the test set or BTXRD (see `--eval_test` default off).
- Best **run/config** chosen **manually** afterward (pull runs via wandb API),
  ranked by best `val/f1_macro`, **tie-break by `val/balanced_acc`** for runs
  within ~1 flip (~0.01–0.02) of the top, then `val_loss` as last resort.
- Because BTXRD never enters selection, the eventual **BTXRD numbers are an
  unbiased estimate of generalization** for the chosen config. The sweep's
  select-the-max optimism inflates only the internal-val estimate, not BTXRD.

### 1.3 Final evaluation = 10-fold CV on internal + fixed external BTXRD test
- **BTXRD is excluded from LACE pretraining** (user will enforce) → clean
  external held-out test, no representation leakage.
- Take the winning hyperparameters from the sweep and **hold them fixed** across
  all 10 folds (no per-fold re-sweeping — may revisit later as nested CV).
- **10-fold** (not 5): test fold = 1/k, so 10 folds give test ≈ 0.1, val ≈ 0.1,
  train ≈ 0.8 — preserving the current split ratio (user's priority: keep train
  large). 5-fold would have forced test = 0.2 / train = 0.7. Each fold file has
  its own `train` / `val` / `test`; `val` drives within-fold early stopping +
  checkpoint (`val_loss`); `test` = the held-out internal fold. The 10 test folds
  **tile the whole dataset** (verified: 921 samples, each tested exactly once).
- For **each** of the 10 fold-models, evaluate on **both** its internal `test`
  split **and** the single frozen **BTXRD** manifest.
- **Internal reporting — PRIMARY = pooled out-of-fold (OOF) macro-F1**:
  concatenate all 10 folds' test predictions into one 921-length vector (each
  sample predicted once, by a model that never trained on it) → a single
  macro-F1 over all 921. This is more stable than averaging 10 small-fold F1s
  (macro-F1 is nonlinear) and is *not* contamination (OOF ⇒ no train/test
  overlap per prediction). SECONDARY = per-fold macro-F1 **mean ± std** as a
  stability indicator.
- **BTXRD reporting** = **mean ± std** over the 10 fold-heads (each scores all of
  BTXRD). Ensemble (soft-voting) considered but **not pursued for now**.
- Residual caveat (accepted): hyperparameters were tuned on `split_binary.json`'s
  val, now pooled into CV → mild flat-CV optimism, equal for pooled and per-fold.
  Full fix = nested CV (deferred).

---

## 2. Implementation

### 2.1 `src/LACE/train/downstream.py` (modify)
1. **Log `val/f1_macro` every epoch** (from the `val_preds`/`val_labels` that
   `evaluate` already returns): `f1_score(val_labels, val_preds, average="macro")`.
   Also log `val/balanced_acc` (already logged) — used as tie-break.
2. **Set a wandb summary** `val/best_f1_macro` (running max) so runs are sortable
   across the sweep by the selection metric.
3. **Keep checkpoint + early stopping on `val_loss`** (unchanged).
4. **`--eval_test` flag** (`action="store_true"`, default **False**):
   - Only build the `test` dataset, pre-compute test representations, and run the
     final test evaluation **if set**. Sweep runs omit it → test untouched.
5. **`--btxrd_manifest PATH`** (optional): if given (and `--eval_test`), also
   evaluate the BTXRD manifest as an extra external test set, using the **fold's
   train age mean/std** for age normalization and the same `label_to_idx`.
   Logs/prints `btxrd/*` metrics analogous to `test/*`.
6. **Refactor `main()` to return a metrics dict** (`{"val/...": ..., "test/...":
   ..., "btxrd/...": ...}`) so the CV orchestrator can aggregate. Single-run CLI
   behavior (printing the TEST RESULTS block) unchanged.

### 2.2 `src/LACE/train/sweep_downstream_v2.yaml` (modify)
- `metric.name: val/f1_macro`, `metric.goal: maximize`.
- (Command still omits `--eval_test`, so sweep never evaluates test/BTXRD.)

### 2.3 `src/build_btxrd_downstream.py` (new)
- Read `data/BTXRD/dataset.xlsx`: keep rows with `tumor==1` and exactly one of
  `benign==1` / `malignant==1` → `label ∈ {benign, malignant}`.
- `age` from `age`; `sex` from `gender` (M→1.0, F→0.0, matching internal).
- `image` → `data/BTXRD/preprocessed_images/<image_id>.png`.
- `mask`: rasterize `Annotations/<id>.json` polygons via existing
  `rasterize_shapes` + the square-pad logic from `BTXRDOrthoDataset`
  ([datasets.py:509-517](data/../src/LACE/data/datasets.py#L509-L517)); save PNG
  to `data/BTXRD/downstream_masks/<id>.png` so the internal `DownstreamDataset`
  (`use_mask=True`) can `crop_around_mask` consistently. Rows without an
  annotation → `mask` path that does not exist (dataset falls back to full image).
- Emit `data/BTXRD/btxrd_downstream_binary.json`: a flat list of sample dicts with
  keys `{image, mask, label, age, sex, patid}` (schema compatible with the
  internal split samples).

### 2.4 `src/create_cv_splits.py` (new)
- Stratified **5-fold** over the internal dataset (binary), seed 42.
- For each fold `k`: `test` = fold `k`; from the remaining 4 folds carve a `val`
  (stratified, ~same size as current 0.1) and `train` = rest.
- Write `data/internal_dataset/cv/split_binary_fold{0..4}.json`, each with
  `train`/`val`/`test` in the existing sample-dict schema. Reuse the label
  filtering + patient-grouping conventions from `build_stratified_splits`
  (avoid patient leakage across folds if `patid` grouping is used there).

### 2.4b Pooled OOF wiring (added after 10-fold decision)
- `downstream.main` returns the internal-test `preds`/`labels` (`test/_preds`,
  `test/_labels`) alongside scalar metrics so folds can be pooled.
- `lace_downstream_cv.py` peels those off each fold, concatenates them, and
  computes the **pooled** `test_pooled/*` metrics (primary) via `report_eval`;
  per-fold scalars still aggregate to mean ± std (secondary); BTXRD stays
  mean ± std. All three written to `cv_results.json`
  (`pooled_oof_internal`, `per_fold`, `aggregate`).

### 2.5 `src/lace_downstream_cv.py` (new — the k-CV orchestrator)
- Args: `--checkpoint`, `--cv_dir` (or explicit `--splits` list of 5),
  `--btxrd_manifest`, all the **fixed** hyperparameters (lr, dropout,
  hidden_dims, loss, focal_gamma, class_weighting, weight_decay, batch_size,
  head, downstream_visual_mode, image_size, version, binary, use_mask, epochs,
  patience, seed), `--wandb`.
- For each of the 5 fold splits: call the refactored `downstream` training
  routine with `--eval_test` and `--btxrd_manifest` set → collect the returned
  metrics dict.
- Aggregate **mean ± std** across folds for `test/f1_macro`, `test/balanced_acc`,
  `btxrd/f1_macro`, `btxrd/balanced_acc`, etc. Print a summary table and
  optionally log one wandb summary run.

### 2.6 sbatch wrappers (new, mirror existing style)
- `sbatch/lace/run_build_btxrd_downstream.sh`
- `sbatch/data/run_create_cv_splits.sh`
- `sbatch/lace/run_lace_downstream_cv.sh`

---

## 3. Open / deferred
- Per-fold hyperparameter re-tuning (nested CV) — deferred; fixed hyperparameters
  for now.
- Ensemble-of-folds BTXRD number (option B) — not pursued; mean ± std only.
- Patient-level grouping for BTXRD (if multiple images per patient) — verify
  BTXRD `image_id` uniqueness; BTXRD has no patient id, treat each image
  independently.

---

## 4. Build / verify order
1. `downstream.py` (`--eval_test`, `val/f1_macro`, `--btxrd_manifest`, return dict).
2. `sweep_downstream_v2.yaml` metric.
3. `build_btxrd_downstream.py` → generate manifest, spot-check counts/labels.
4. `create_cv_splits.py` → generate 5 folds, check stratification + no test leak.
5. `lace_downstream_cv.py` → dry-run 1 fold locally, then full 5-fold on SLURM.
