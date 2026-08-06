# Image Preprocessing Pipeline

This document describes exactly what happens to an image (and, where relevant, its
lesion segmentation mask) between the raw file on disk and the tensor that enters
the network. It covers the internal dataset and the external BTXRD dataset
separately, since BTXRD requires an extra mask-construction step that the internal
dataset does not.

There is **no offline preprocessing step used by the actual training runs** —
[`src/preprocess_images.py`](src/preprocess_images.py) (offline square-padding into
a `preprocessed_images/` directory) is not called by any of the training data
loaders. All cropping/padding happens on-the-fly, per sample, inside
`Dataset.__getitem__`. The one exception is BTXRD, where the downstream-manifest
builder ([`src/build_btxrd_downstream.py`](src/build_btxrd_downstream.py)) reads
from `data/BTXRD/preprocessed_images/`, so that directory must exist (produced by
running `preprocess_images.py --dataset btxrd` once) before building the BTXRD
manifest — see §2.

---

## 1. Internal dataset

Raw images and masks are loaded directly from `data/internal_dataset/images/` and
`data/internal_dataset/segmentations/` (paths as stored in `split.json`), not from
any preprocessed copy.

### 1.1 Mask-guided crop: `crop_around_mask` / `crop_around_mask_pair`

Implemented in [`src/biomedclip/data/transforms.py:36-155`](src/biomedclip/data/transforms.py).
Both functions use identical geometry; `crop_around_mask` returns only the cropped
image (used by plain BiomedCLIP), `crop_around_mask_pair` returns the image **and**
the mask cropped with the same coordinates (used by LACE, since it needs the mask
to derive per-patch lesion-coverage labels).

Given a raw image and its binary mask:

1. **Skip condition.** If cropping is disabled, or the mask is empty (no
   foreground pixels), the raw image is returned unchanged (`crop_around_mask`) —
   the no-mask/empty-mask case then falls through to plain square-padding, see
   §1.2.
2. **Lesion bounding box.** Binarize the mask (`mask > 0`) and take the tight
   bounding box: `r_min, r_max, c_min, c_max` over the nonzero pixels.
   `box_h = r_max - r_min + 1`, `box_w = c_max - c_min + 1`.
3. **Context margin.** `context_px = ceil(min(H, W) * context_fraction)`, where
   `H, W` are the *full raw image* dimensions and `context_fraction` defaults to
   **0.15**. Because this margin is a fixed number of pixels derived from the
   whole image (not from the lesion size), **small lesions end up with
   proportionally more surrounding context than large lesions**.
4. **Crop side.** The crop is always square:
   `side = max(box_h, box_w) + 2 * context_px`.
5. **Centering on the lesion.** The crop window is centered on the lesion
   bounding-box centroid `(cy, cx) = ((r_min+r_max)/2, (c_min+c_max)/2)`.
6. **Per-axis placement, including the edge case** —
   `_compute_crop_1d` ([`transforms.py:11-33`](src/biomedclip/data/transforms.py)),
   applied independently to the row axis and the column axis:
   - **If the crop fits inside the image** (`side <= dim_size`): a window of
     length `side` is centered on the lesion centroid, then **shifted (clamped)
     toward the interior** if it would run off the image edge —
     `start = max(0, min(round(center - side/2), dim_size - side))`. In other
     words, if the lesion sits close to the image border, the crop window slides
     inward just enough to stay fully inside the image, rather than padding. No
     black border is added in this branch, and the lesion is consequently
     **not perfectly centered** in the output when this shift occurs.
   - **If the crop is larger than the image axis** (`side > dim_size`, e.g. a
     very large lesion or a small raw image): the whole axis is used
     (`0` to `dim_size`), and the missing pixels are made up with zero-padding.
     The padding is split between "before" and "after" so that the lesion
     centroid lands as close as possible to the crop's center:
     `pad_total = side - dim_size`,
     `ideal_pad_before = round(side/2 - center)`, clamped to `[0, pad_total]`.
7. **Slice + pad.** The image (and, for `crop_around_mask_pair`, the mask) is
   sliced with the resulting row/column start-end indices, then any needed
   padding is applied with `np.pad` — image padding uses `constant` mode with
   `pad_value` (default `0.0`, i.e. black), mask padding always uses
   `constant_values=0` regardless of the image's pad mode.
8. **Output.** A square crop of side `side`. Note `side` varies per sample
   (it depends on lesion size, not a fixed target resolution) — resizing to a
   fixed network input size happens later in the `preprocess`/train-transform
   step (§1.3).

### 1.2 No-mask fallback: `pad_to_square`

[`transforms.py:94-104`](src/biomedclip/data/transforms.py). If a sample has no
mask file, or the mask is empty, or `context_fraction < 0` (cropping disabled),
the raw image is instead **symmetrically zero-padded** to a square
(`side = max(H, W)`, padding split evenly — `pad_before = pad // 2`,
`pad_after = pad - pad_before` — on each axis), with no lesion-centering logic
at all. This is the on-the-fly equivalent of what the (unused) offline
`preprocess_images.py` script does.

### 1.3 Resize, normalize, and train-time augmentation

This stage is arguably no longer "preprocessing" in the geometric-alignment sense
above — it's the standard resize/normalize + augmentation stage shared by (almost)
every vision pipeline, but it's included here since it still transforms the image
before the network sees it, and for LACE it is coupled to the mask.

**Base pipeline** (BiomedCLIP `preprocess`/`preprocess_val`, from
`open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")`):
`Resize(224, bicubic) → CenterCrop(224, 224) → ToTensor → Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])`
(standard CLIP normalization statistics). Used as-is for validation/test, and as
the base for LACE's non-training-time crops.

**Plain BiomedCLIP training augmentation** — `build_train_transform`
([`transforms.py:227-253`](src/biomedclip/data/transforms.py)):
`RandomResizedCrop(224, scale=(0.8, 1.0)) → RandomRotation(±10°) → RandomAdjustSharpness(factor=2, p=0.3) → GaussianBlur(kernel=3, sigma=(0.1, 1.0)) → ToTensor → Normalize`.
No horizontal/vertical flips — deliberately, since left/right laterality is
clinically meaningful in these radiographs.

**LACE training augmentation — `synchronized_train_transform`**
([`src/LACE/data/transforms.py:109-154`](src/LACE/data/transforms.py)). LACE needs
the mask to remain spatially registered to the augmented image so it can compute
per-patch lesion-coverage labels for its mask-decoder loss, so the geometric
augmentation is applied identically to image and mask:
- One `RandomResizedCrop.get_params(scale=(0.8, 1.0), ratio=(3/4, 4/3))` draw,
  applied via `TF.resized_crop` to both the image (BICUBIC) and mask (NEAREST)
  with the same `(i, j, h, w)`.
- One rotation angle sampled uniformly in `[-10°, 10°]`, applied via `TF.rotate`
  to both (image BILINEAR, mask NEAREST, `fill=0`).
- `RandomAdjustSharpness` and `GaussianBlur` applied to the image only
  (photometric — mask-independent).
- Image: `ToTensor → Normalize`. Mask: thresholded at `>127` (0/255 → 0/1),
  then patchified.

**Patchification** — `patchify_mask_224`
([`LACE/data/transforms.py:95-106`](src/LACE/data/transforms.py)): the (augmented)
mask is reshaped into the ViT's `16×16` patch grid
(`P = (image_size // 16)**2`, e.g. 196 patches at 224px), and each patch's label
is its **fraction of mask coverage** (mean over the 16×16 pixels), with patches
below 10% coverage zeroed out (`min_coverage=0.10`). At eval time the same
patchification is derived from an unaugmented mask via `mask_to_patch_labels`
(resize + center-crop with NEAREST interpolation, matching the eval preprocess
pipeline, then patchify).

### 1.4 Dual crop for LACE (`InternalTripleDataset`)

[`src/LACE/data/datasets.py:336-397`](src/LACE/data/datasets.py). Unlike plain
BiomedCLIP (single crop per sample), LACE produces **two independently-cropped
views** of each sample, both derived from the same raw image/mask pair via
`crop_around_mask_pair` (§1.1):

| View | `context_fraction` | Purpose |
|---|---|---|
| `global_crop` | 0.4 (`global_context_fraction`) | wide view, used for global image-text alignment (`L_ITA`) |
| `crop_image`  | 0.15 (`context_fraction`)        | tight lesion-centered view, used for local lesion-phrase alignment (`L_sim`) and the mask-decoder loss |

If there is no valid mask, both views fall back to the full raw image
(`pad_to_square`), and `has_mask=False` signals the training loop to skip
`L_sim`/the internal `L_ortho` term for that sample. At train time
(`is_train=True`), each view independently goes through
`synchronized_train_transform` (§1.3, its own random crop/rotation draw per view);
at eval time both go through the plain `preprocess` pipeline, with patch labels
recomputed directly from the (already geometrically-cropped) mask via
`mask_to_patch_labels`.

---

## 2. BTXRD (external test set)

BTXRD ships as raw JPEG images plus LabelImg-style polygon/rectangle annotation
JSONs (`data/BTXRD/Annotations/<id>.json`) — there is no pre-existing lesion
segmentation PNG the way the internal dataset has one. Building a usable BTXRD
manifest is therefore a two-step process: an offline square-padding pass, then a
one-off script that rasterizes the polygon annotations into binary masks and
assembles the downstream JSON manifest. After that, BTXRD samples are consumed by
the exact same `DownstreamDataset`/`crop_around_mask` logic as the internal
dataset (§1.1–1.3).

### 2.1 Offline square-padding (`preprocess_images.py --dataset btxrd`)

This is the one place `preprocess_images.py` is actually a prerequisite: BTXRD's
manifest builder reads images from `data/BTXRD/preprocessed_images/` by default.
Each raw `.jpeg` is opened, converted to RGB, and symmetrically pad-to-square with
black borders, centered (`_square_pad`,
[`preprocess_images.py:34-42`](src/preprocess_images.py)) — same padding logic as
§1.2's `pad_to_square`, just run once offline and cached as PNGs rather than
per-sample. No mask padding happens here (BTXRD has no segmentation PNGs at this
stage).

### 2.2 Binary segmentation mask construction

[`src/build_btxrd_downstream.py`](src/build_btxrd_downstream.py), using
`rasterize_shapes` from
[`src/LACE/data/transforms.py:53-78`](src/LACE/data/transforms.py):

1. For each sample with a matching `Annotations/<id>.json`, the annotation's
   `shapes` list (polygons and/or rectangles, in the **original, un-padded**
   image's pixel coordinates) is rasterized onto a blank `(img_h, img_w)` canvas:
   each `polygon` shape is filled via `PIL.ImageDraw.polygon`, each `rectangle`
   shape via `ImageDraw.rectangle` — producing a binary mask (255 inside any
   annotated region, 0 elsewhere).
2. **Square-pad to match the preprocessed image.** Since the corresponding image
   was already square-padded in step 2.1, the freshly rasterized mask (still in
   original, non-square dimensions) is padded the same way — centered,
   zero-filled — so it lines up pixel-for-pixel with the padded image:
   `side = max(img_h, img_w)`, `pad_top = (side-img_h)//2`,
   `pad_left = (side-img_w)//2`. (This re-implements the same centered-padding
   logic as `pad_to_square`/`_square_pad` rather than calling them directly, but
   produces the identical result.)
3. If the rasterized+padded mask is entirely empty (no annotated region), no
   mask file is written and the sample's `mask` field is set to `""` — such
   samples fall back to the full preprocessed image at load time (no crop
   applied, since `DownstreamDataset` treats a missing/empty mask the same as
   the internal dataset's no-mask case, §1.2).
4. The resulting mask PNG is saved to `data/BTXRD/downstream_masks/<id>.png`.

Only rows that are tumor-positive with exactly one of `{benign, malignant}` set
and valid numeric `age`/`gender` are kept (binary-only, matching the internal
`--binary` split setup); everything else is dropped and counted in the printed
summary (`no_tumor`, `ambiguous_label`, `bad_age_sex`, `missing_image`).

The output manifest (`btxrd_downstream_binary.json`) is a flat list of
`{"image", "mask", "label", "age", "sex", "patid"}` dicts consumed directly by
`DownstreamDataset` / the k-fold CV orchestrator via `--btxrd_manifest`, applying
the same `crop_around_mask` logic (§1.1, `context_fraction=0.15` by default) as
the internal dataset whenever a mask is present.

### 2.3 `BTXRDOrthoDataset` (pretraining-time L_ortho auxiliary set)

A separate, simpler path used only during LACE pretraining to supply the
orthogonality regularization term `L_ortho`
([`src/LACE/data/datasets.py:474-528`](src/LACE/data/datasets.py)) — not the
downstream-eval path above. For each annotated BTXRD image it rasterizes the mask
on the fly (`rasterize_shapes`, same as §2.2), pads it to match the square
preprocessed image (same centered-padding, computed inline), and derives patch
labels (`mask_to_patch_labels`) — but applies **no lesion-centered crop**: the
full square image goes through `self.preprocess` unchanged. This dataset is only
used for the mask-decoder auxiliary loss, not for image-text alignment.
