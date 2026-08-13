# LACE Loss Functions

Reference notes for the thesis write-up on LACE's training objective. LACE v2 (the version with the `MaskTokenDecoder`, [src/LACE/train/pretrain_v2.py](src/LACE/train/pretrain_v2.py)) combines **four** loss terms:

$$\mathcal{L}_\text{total} = \lambda_\text{ita}\,\mathcal{L}_\text{ITA} + \lambda_\text{sim}\,\mathcal{L}_\text{sim} + \lambda_\text{ortho}\,\mathcal{L}_\text{ortho} + \lambda_\text{dice}\,\mathcal{L}_\text{dice}$$

All four weights $\lambda_*$ can be fixed hyperparameters or, with `--learn_loss_weights`, unconstrained log-scale parameters $\log\lambda_*$ optimized jointly with the network (initialised to $\log(\lambda_\text{arg})$, updated by AdamW). Implementation: [src/LACE/loss/objectives.py](src/LACE/loss/objectives.py). Curriculum: by default $\mathcal{L}_\text{ortho}$ and $\mathcal{L}_\text{dice}$ are active from stage 1 (mask-decoder warm-up), while $\mathcal{L}_\text{ITA}$ and $\mathcal{L}_\text{sim}$ only switch on in stage 2 (`--stage1_epochs`, default 10 of 40 total epochs), fully overridable per-loss via `--loss_stages`.

Note: earlier iterations of v2 also carried a reconstruction loss ($\mathcal{L}_\text{rec}$) and an evidential loss ($\mathcal{L}_\text{evid}$); both were removed (commit `ae3f875`, "Removed L_rec and L_evid"). The four losses below are the current active set.

---

## 1. $\mathcal{L}_\text{ITA}$ — Image–Text Alignment

**Purpose:** global-level contrastive alignment between the image CLS embedding and the *Beurteilung* (radiological assessment) phrase embeddings.

**Function:** `symmetric_soft_semantic_loss` ([objectives.py:81](src/LACE/loss/objectives.py)).

Given one image embedding $z_\text{img}$ per valid phrase and phrase embeddings $\phi_j = \text{cls\_proj}(\text{CLS}_j)$ (both L2-normalized, 512-dim):

**I2T (image → text), soft KL:**

$$s_{ij} = z_i \cdot \phi_j / \tau, \qquad q^\text{phrase}_{ij} = \text{softmax}_j\!\left(\phi_i \cdot \phi_j / \tau_{s,\text{beur}}\right)$$
$$\mathcal{L}_\text{ITA}^\text{i2t} = \text{KL}\big(\text{softmax}(s_{i\cdot}) \,\|\, q^\text{phrase}_{i\cdot}\big)$$

The soft target is anchored on **text-text similarity**, not image-image similarity — motivated by the high visual heterogeneity of bone-tumor X-rays, where phrase semantics are a more reliable notion of "similar case" than raw pixel similarity.

**T2I (text → image)** — direction and target controlled by `--t2i_mode`:

| Mode | Target $q^\text{t2i}$ |
|---|---|
| `image_image` (default) | $\text{softmax}(z_i \cdot z_j / \tau_{s,\text{img}})$ |
| `text_text` | same soft target as I2T ($q^\text{phrase}$) |
| `infonce` | hard one-hot InfoNCE, phrase → its source image (cross-entropy instead of KL) |

A fourth mode, `descriptor` (cosine similarity of 21-dim binary lesion-descriptor vectors, MedCLIP eq. 4 style), exists in the code but was not empirically evaluated — omitted here.

$$\mathcal{L}_\text{ITA} = \mathcal{L}_\text{ITA}^\text{i2t} + \lambda_\text{t2i}\,\mathcal{L}_\text{ITA}^\text{t2i}$$

**Modifiers:**
- `--same_image_boost` $b$: adds $b$ to logits between phrases from the same image before the softmax that builds $q^\text{phrase}$, so within-image phrases dominate the target.
- `--reweight_by_n_phrases`: reweights each phrase by $1/(n_\text{phrases of its image} \cdot B)$ so images with many phrases don't dominate the batch loss.

**Temperatures:** learned contrastive $\tau$ (clamped to $[0.01, 0.5]$), soft-target $\tau_{s,\text{beur}}$ (`--tau_s_beur`, default 0.015 in v2 / 0.04 in v1), $\tau_{s,\text{img}}$ (`--tau_s_img_full`, default 0.07).

**Curriculum:** stage 2 only by default (`--loss_stages ita:2`).

---

## 2. $\mathcal{L}_\text{sim}$ — Local Phrase–Patch Alignment (GLoRIA-style)

**Purpose:** fine-grained alignment between individual *Befund* (findings) phrases and localized image regions, so the model learns which patches a given phrase refers to.

**Function:** `gloria_local_loss` ([objectives.py:203](src/LACE/loss/objectives.py)). Computed only for samples with both a segmentation mask and Befund phrases (`has_mask & has_befund`), requires ≥2 valid samples per batch.

**Attention-weighted context vector** for phrase $j$ over image $i$'s projected patches $P^{(i)} \in \mathbb{R}^{196\times512}$:

$$\alpha_{jm}^{(i)} = \text{softmax}_m\!\left(p_m^{(i)} \cdot \phi_j / \tau_2\right), \qquad c_j^{(i)} = \text{normalize}\Big(\textstyle\sum_m \alpha_{jm}^{(i)}\,p_m^{(i)}\Big)$$

If a lesion patch mask is available, attention is restricted to lesion patches (background masked to $-\infty$ before the softmax); images with no lesion patch fall back to attending everywhere.

**I2T (soft KL):** own-image attention feature $c_j^{(i)}$ vs. all phrases in the batch, anchored by the same phrase-phrase soft target construction as $\mathcal{L}_\text{ITA}$ (temperature $\tau_{s,\text{bef}}$, default 0.07).

**T2I:** cross-image attention features $c_j^{(k)}$ for every image $k$ in the batch, then, controlled by `--t2i_mode` (same options as $\mathcal{L}_\text{ITA}$: `infonce` default here, `text_text`, `image_image`).

$$\mathcal{L}_\text{sim} = \mathcal{L}_\text{sim}^\text{i2t} + \lambda_\text{t2i}\,\mathcal{L}_\text{sim}^\text{t2i}$$

**Attention temperature** $\tau_2$: `--sim_attn_tau`, default 0.07.

**Curriculum:** stage 2 only by default.

---

## 3. $\mathcal{L}_\text{ortho}$ — Lesion–Background Orthogonality

**Purpose:** regularizer that pushes the raw ViT feature representation of lesion patches away from background patches, so the encoder's patch space itself becomes lesion-discriminative (independent of the text branch).

**Function:** `ortho_loss` ([objectives.py:488](src/LACE/loss/objectives.py)), operating in the raw 768-dim ViT space (pre-projection). Uses **soft patch weighting**: `patch_labels` are coverage fractions in $[0,1]$ (fraction of a 16×16 patch covered by the lesion mask), not hard 0/1 labels — a patch that is 20% lesion contributes 20% of its embedding to the lesion prototype and 80% to the background prototype.

$$v_\text{lesion}^{(i)} = \frac{\sum_m w^\text{les}_m\,h_m^{(i)}}{\sum_m w^\text{les}_m}, \qquad v_\text{bg}^{(i)} = \frac{\sum_m (1-w^\text{les}_m)\,h_m^{(i)}}{\sum_m (1-w^\text{les}_m)}$$

$$\mathcal{L}_\text{ortho} = \frac{1}{N}\sum_i \big(\cos(v_\text{lesion}^{(i)}, v_\text{bg}^{(i)}) + 1\big)$$

The $+1$ shifts the range from $[-1,1]$ to $[0,2]$ (non-negative loss) without changing gradients; 0 = maximally separated, 2 = identical. Samples where all weight falls on one side (no lesion or no background) are excluded, returning a differentiable 0 if none remain.

An earlier hard-label variant, `ortho_loss_old`, is kept in the codebase for reference — it used binary patch labels and a `min_lesion_patches` threshold instead of soft coverage weighting.

**BTXRD supplementation:** the external BTXRD dataset (`BTXRDOrthoDataset`, polygon/rectangle mask annotations) is cycled in via `itertools.cycle`-style infinite iteration alongside the internal batch, adding an extra $\mathcal{L}_\text{ortho}$ (and $\mathcal{L}_\text{dice}$) term computed on BTXRD patches — extra lesion/background supervision beyond the internal dataset's mask coverage.

**Curriculum:** active in both stage 1 and stage 2 by default (`--loss_stages ortho:1,2`).

---

## 4. $\mathcal{L}_\text{dice}$ — Mask-Token Segmentation Loss (v2 only)

**Purpose:** trains the `MaskTokenDecoder` + `MaskPredictionHead` ([src/LACE/models/mask_tokens.py](src/LACE/models/mask_tokens.py)) to predict a lesion segmentation heatmap over the 14×14 (196) patch grid, grounding a subset of learned mask tokens in the actual lesion location before the contrastive losses (which depend on lesion localization for $\mathcal{L}_\text{sim}$) switch on.

**Architecture producing the input:** `n_mask_tokens` (default 4) learnable tokens cross-attend to ViT patch tokens (with optional Gaussian smoothing on the attention weights, `--gauss_sigma`, to encourage spatially contiguous focus regions on the patch grid), then self-attend among each other. `MaskPredictionHead` turns the refined tokens + patch tokens into per-token, per-patch logits `mask_logits` $\in \mathbb{R}^{B\times N\times196}$ ($N$ = `n_mask_tokens`).

**Function:** `compute_l_dice_ce` ([objectives.py:372](src/LACE/loss/objectives.py)).

**Token selection:** rather than supervising all $N$ mask tokens, the token whose *detached* sigmoid overlap with the ground-truth coverage is highest is selected as the foreground (fg) token per sample:

$$\text{fg}_i = \arg\max_n \sum_p \sigma(\text{logits}_{i,n,p})\cdot\text{gt}_{i,p}$$

This gives a stable target — the token partially covering the lesion keeps receiving gradient to cover it better, instead of the assignment jumping between tokens across batches. `select_fg_token` provides an alternative, label-free selection heuristic (peak-to-mean activation ratio) used at inference/visualization time when no GT is available.

**Loss terms, all computed on the fg token's logits** (samples with all-zero GT are excluded):

1. **Soft Dice** — ratio-based overlap against soft coverage-fraction ground truth:
$$\mathcal{L}_\text{dice} = 1 - \frac{2\sum_p \hat p_p\, g_p}{\sum_p \hat p_p + \sum_p g_p + \epsilon}$$
2. **Global BCE** — dense per-patch binary cross-entropy over all 196 patches, for stable gradient signal everywhere (Dice alone gives a sparse/ratio-only signal).
3. **Hard-negative BCE** — an additional penalty on the top-`hard_neg_k` (default 10) most-activated *background* patches, weighted by `hard_neg_weight` (default 0.5). Targets leakage onto adjacent bone structure that global BCE alone under-penalizes because easy correctly-classified background patches dilute its gradient.

$$\mathcal{L}_\text{dice}^\text{total} = \mathcal{L}_\text{dice} + \mathcal{L}_\text{BCE} + \text{hard\_neg\_weight}\cdot\mathcal{L}_\text{BCE}^\text{hard-neg}$$

(Confusingly the combined term is what the code and this document call $\mathcal{L}_\text{dice}$ / `l_dice` throughout training — the individual Dice-only value is logged separately as `l_dice_only`.)

**BTXRD supplementation:** as with $\mathcal{L}_\text{ortho}$, an extra $\mathcal{L}_\text{dice}$ term is computed on cycled-in BTXRD batches and added to the internal-dataset term.

**Curriculum:** active in both stage 1 and stage 2 by default — it is the loss stage 1 is warming up.

---

## Summary Table

| Loss | Level | Modality | Default stage(s) | Default weight $\lambda$ | Key temperature(s) |
|---|---|---|---|---|---|
| $\mathcal{L}_\text{ITA}$ | Global | Image ↔ Beurteilung phrases | 2 | 1.0 | $\tau$, $\tau_{s,\text{beur}}$=0.015, $\tau_{s,\text{img}}$=0.07 |
| $\mathcal{L}_\text{sim}$ | Local (patch) | Image patches ↔ Befund phrases | 2 | 1.0 | $\tau$, $\tau_{s,\text{bef}}$=0.07, $\tau_2$=0.07 |
| $\mathcal{L}_\text{ortho}$ | Local (patch) | Image only (uses mask) | 1, 2 | 0.1 | — |
| $\mathcal{L}_\text{dice}$ | Local (patch) | Image only (uses mask) | 1, 2 | 1.0 | mask_head_tau=1.0 |

All four are combined in two backward passes per batch in v2 (`train_one_epoch`, [pretrain_v2.py:226](src/LACE/train/pretrain_v2.py)): pass 1 optimizes $\mathcal{L}_\text{ITA}$ alone (when in-stage), pass 2 jointly optimizes $\mathcal{L}_\text{sim} + \mathcal{L}_\text{ortho} + \mathcal{L}_\text{dice}$ (when in-stage) via a single `loss_seg` backward.
