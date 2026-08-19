# Dataset Split Creation

Notes on how the train/val/test split for the downstream malignancy classifier was created. Implementation: [`build_stratified_splits`](src/biomedclip/data/splits.py) in [src/biomedclip/data/splits.py](src/biomedclip/data/splits.py), invoked via [src/create_split.py](src/create_split.py). Output: `split.json` (3-class) or `split_binary.json` (benign/malignant only, via `--binary`).

## 1. Sample filtering

Starting from `dataset_full.json` (the assembled dataset manifest), an entry is only included if it has:
- a non-empty report,
- a malignancy label,
- a segmentation mask that actually contains foreground pixels (`_mask_has_pixels`),
- (binary mode only) a label other than `intermediate` — intermediate cases are excluded entirely rather than merged into benign/malignant.

Unknown age/sex values are encoded numerically (age → imputed later with the training-set mean; sex → 0.5) rather than dropped.

## 2. Patient-grouped split — no leakage across patients

**Images belonging to the same patient (`patid`) are kept together** — all of a patient's images end up in the same split (train, val, *or* test), never spread across more than one. This is done by grouping samples by `patid` first (`_group_by_patient`) and splitting on the resulting *groups*, not on individual images. Splitting on individual images instead would let two X-rays of the same lesion/patient end up on both sides of the train/test boundary, letting the model implicitly memorize patient-specific appearance rather than learning malignancy features — this split avoids that leakage.

## 3. Stratified split

The split is **stratified by label**, using `sklearn.model_selection.train_test_split(..., stratify=labels)`. Since splitting happens at the patient-group level, the label used per group is the **majority label** across that patient's images (`_majority_label`) — a patient can have multiple images, occasionally with disagreeing per-image labels, so the group is stratified by whichever label is most common among them.

The split is done in two stratified steps to get three sets from one fraction-based config:
1. `test_frac` is split off first: `patient_groups` → `train_val_groups` + `test_groups`.
2. The remainder is split again into `train_groups` / `val_groups`, using `val_frac = downstream_val_frac / (downstream_train_frac + downstream_val_frac)` so the final proportions match `--downstream_train_frac`, `--downstream_val_frac`, `--test_frac` (which must sum to 1.0).

If stratification fails (too few samples in some class for a given split ratio), it falls back to a plain random split with a warning (`_stratified_split_two`).

## 4. Reproducibility

Both split steps use the same `--seed`, so the split is deterministic given the same `dataset_full.json` and fraction config.

## 5. Output

`train`, `val`, `test` sample lists (flattened back from patient groups) are written to a single JSON manifest (`split.json` / `split_binary.json`) alongside per-split statistics (image count, patient count, per-class distribution) printed to stdout for a quick sanity check that stratification held up in practice.
