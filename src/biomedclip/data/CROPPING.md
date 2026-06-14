# Mask-Guided Lesion Cropping

This document describes the lesion-cropping logic used across all baselines
(BiomedCLIP, GLoRIA, LACE, …). It is implemented in
[`transforms.py`](transforms.py) and enabled per run via the `--use_mask` flag.

When `use_mask` is set, each dataset `__getitem__` crops the X-ray to a square
region centred on the lesion **before** the model's resize/normalize pipeline
runs:

```python
image_arr = np.array(image.convert("L"), dtype=float)
mask_arr  = np.array(Image.open(mask_path).convert("L"), dtype=float)
cropped   = crop_around_mask(image_arr, mask_arr)
image     = Image.fromarray(cropped.astype(np.uint8)).convert("RGB")
```

The goal is to feed the encoder a tight, lesion-centred view instead of the full
radiograph, so most of the input pixels carry diagnostically relevant signal.

---

## 1. Inputs

- `image_arr` — the X-ray as a 2-D (grayscale) or 3-D (H×W×C) array.
- `mask` — the segmentation mask, same H×W. Any pixel `> 0` is treated as lesion.

If the mask is **empty** (no positive pixels), cropping is skipped and the
original image is returned unchanged. This makes the crop safe for samples that
lack a usable segmentation.

---

## 2. Computing the crop window

The crop is always a **square**, sized from the lesion bounding box plus a
fixed-pixel context margin.

```
binary_mask = mask > 0
rows, cols  = where(binary_mask)
box_h = rows.max() - rows.min() + 1          # bounding-box height
box_w = cols.max() - cols.min() + 1          # bounding-box width

context_px = ceil(min(H, W) * context_fraction)   # default context_fraction = 0.15
side       = max(box_h, box_w) + 2 * context_px    # square side length
```

Two design choices matter here:

1. **The side is `max(box_h, box_w)`**, so the window is square and fully
   contains the lesion bounding box regardless of its aspect ratio.

2. **`context_px` is a fixed number of pixels derived from the *image* size, not
   the lesion size.** Because the same absolute margin is added to every lesion,
   **small lesions automatically get proportionally more surrounding context**
   than large ones — a small tumour ends up with a wider relative border, which
   preserves anatomical context the model needs to interpret it.

The window is centred on the bounding-box centre:

```
cy = (rows.min() + rows.max()) / 2
cx = (cols.min() + cols.max()) / 2
```

---

## 3. Placing the window (`_compute_crop_1d`)

Each axis is handled independently by `_compute_crop_1d(center, size, dim_size)`,
which returns `(slice_start, slice_end, pad_before, pad_after)`.

**Case A — window fits inside the axis (`size <= dim_size`):**
The window is centred on `center`, then **shifted as far as needed to stay fully
inside `[0, dim_size)`**. This maximises real pixels and means no padding is
required:

```python
start = round(center - size / 2)
start = max(0, min(start, dim_size - size))   # clamp into valid range
return start, start + size, 0, 0
```

So if the lesion sits near an edge, the window slides inward rather than running
off the image.

**Case B — window is larger than the axis (`size > dim_size`):**
The whole axis is used and the shortfall is added as **padding**, split to keep
`center` as close to the output centre as possible:

```python
pad_total        = size - dim_size
ideal_pad_before = round(size / 2 - center)
pad_before       = max(0, min(pad_total, ideal_pad_before))
return 0, dim_size, pad_before, pad_total - pad_before
```

This happens when the lesion (plus margin) is bigger than the image dimension —
e.g. a very large lesion on a narrow radiograph.

---

## 4. Cropping and padding

The computed slices are applied, and any padding from Case B is added:

```python
crop = image_arr[r_start:r_end, c_start:c_end]

if any padding:
    crop = np.pad(crop, pad_width, mode="constant", constant_values=pad_value)
```

- `pad_mode` defaults to `"constant"` with `pad_value = 0.0` → **black border**,
  consistent with how X-rays are letterboxed elsewhere.
- Padding is only ever needed on an axis where the window exceeded the image
  (Case B). When everything fits (Case A), the output is a pure crop with no
  padding.

The result is always a square array of side `side`, lesion-centred, with real
pixels maximised and black padding only where unavoidable.

---

## 5. Function variants

[`transforms.py`](transforms.py) exposes the same geometry in three forms:

| Function | Returns | Use |
|----------|---------|-----|
| `crop_around_mask(image, mask)` | cropped **image** | the standard path used by the datasets |
| `crop_around_mask_pair(image, mask)` | cropped **image + mask** (identical geometry) | when a downstream step also needs the mask cropped to the same window (e.g. local lesion-phrase alignment) |
| `compute_crop_box(image, mask)` | `(r_start, r_end, c_start, c_end)` in original coordinates | when only the box coordinates are needed (e.g. to map something back to the full image); clamps to the image extent instead of padding |

All three share the bbox + fixed-pixel-context geometry described above, so a
crop and its corresponding box/mask always line up.

---

## 6. Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `context_fraction` | `0.15` | margin around the bbox, as a fraction of `min(H, W)` |
| `pad_mode` | `"constant"` | padding mode used when the window exceeds the image |
| `pad_value` | `0.0` | fill value for constant padding (black) |
| `enable_crop` | `True` | `crop_around_mask` only; when `False`, returns the image unchanged |

---

## 7. Summary

1. Empty mask → return image unchanged.
2. Compute lesion bounding box from `mask > 0`.
3. `side = max(box_h, box_w) + 2 * ceil(min(H, W) * context_fraction)` → square window, small lesions get relatively more context.
4. Centre the window on the bbox centre; slide it inward to stay on-image, or pad with black if it is larger than the image.
5. Apply the crop (and identical-geometry mask crop / box if requested).
6. Hand the lesion-centred square to the model's normal resize + normalize transform.
