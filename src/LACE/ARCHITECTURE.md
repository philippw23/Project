# LACE v2 Architecture

**L**esion-**A**ligned **C**ontrastive **E**mbedding — pretraining framework for bone-tumor X-ray / radiology-report alignment.

---

## 1. Overview

LACE v2 adapts an image encoder (BiomedCLIP ViT-B/16 by default, or CheXFound ViT-L/16) to a private internal bone-tumor X-ray dataset via a joint objective with four components, plus a learned `MaskTokenDecoder` that predicts a lesion segmentation mask directly from patch tokens:

$$\mathcal{L}_\text{total} = \lambda_\text{ita} \cdot \mathcal{L}_\text{ITA} + \lambda_\text{sim} \cdot \mathcal{L}_\text{sim} + \lambda_\text{ortho} \cdot \mathcal{L}_\text{ortho} + \lambda_\text{dice} \cdot \mathcal{L}_\text{dice}$$

Unlike v1 (all losses active from epoch 1), v2 runs a **configurable 2-stage curriculum**: by default, stage 1 trains only $\mathcal{L}_\text{dice} + \mathcal{L}_\text{ortho}$ to ground the mask decoder before the image-text losses ($\mathcal{L}_\text{ITA}, \mathcal{L}_\text{sim}$) switch on in stage 2. Per-loss stage assignment is fully overridable via `--loss_stages` (e.g. `dice:1,2 ortho:1,2 ita:2 sim:none`). The weights $\lambda_*$ are either fixed or learned as unconstrained log-scale parameters optimized jointly with the rest of the network.

---

## 2. Model Architecture

### 2.1 Visual Encoder — `SharedViT` / `CheXFoundSharedViT`

| Component | Detail |
|-----------|--------|
| Backbone (`--image_encoder biomedclip`, default) | BiomedCLIP ViT-B/16 trunk (768-dim), frozen base weights |
| Backbone (`--image_encoder chexfound`) | CheXFound ViT-L/16 trunk (1024-dim) |
| Adapter — LoRA (default) | Last `--lora_layers` transformer blocks, rank `--lora_r`, scale `--lora_alpha` |
| Adapter — full unfreeze | `--unfreeze_layers N` (biomedclip only) fully fine-tunes the last N blocks + final norm instead of LoRA, overriding `--lora_layers` |
| CLS projection (`img_proj`) | Linear vit_dim → `--embed_dim` (512), L2-normalized → global image embedding $z_\text{img}$ |
| Patch projection (`patch_proj`) | Linear vit_dim → `--embed_dim`, L2-normalized → patch embeddings $\{p_m\}_{m=1}^{196}$ |
| `--warm_start_projections` | Copies `img_proj`/`patch_proj` weights from BiomedCLIP's pretrained `visual.head.proj` (biomedclip encoder only) |

Only **one crop** is computed per sample (unlike v1's separate global + tight crop): `InternalDatasetV2` crops around the lesion mask using `--context_fraction` (`-1.0` = full image, `0.0` = tight bbox, `>0` = bbox + context margin) and `--context_mode` (`image` = margin relative to full image size; `lesion` = margin relative to the lesion bbox, floored by `--min_crop_size`). Images without a mask always fall back to the full (padded-to-square) image. All losses — $\mathcal{L}_\text{ITA}$, $\mathcal{L}_\text{sim}$, $\mathcal{L}_\text{ortho}$, $\mathcal{L}_\text{dice}$ — operate on this single crop's patch grid.

### 2.2 Mask Token Decoder — `MaskTokenDecoder` + `MaskPredictionHead`

A set of `--n_mask_tokens` learnable tokens cross-attends to the 196 raw ViT patch tokens (manual Q/K/V projections, `--n_mask_heads` heads), with an optional Gaussian smoothing kernel (`--gauss_sigma`, std-dev in patch-grid units; `0` disables it) applied to the cross-attention weights between softmax and value-weighting — this encourages spatially compact, contiguous attention regions rather than scattered patches. The tokens then self-attend among themselves (no spatial smoothing, since tokens have no grid layout).

The kernel (`_make_gauss_kernel`) is a separable 2D Gaussian, size $k = \min(2 \cdot \text{round}(2.5\sigma) + 1,\ 13)$ (odd, capped at 13×13), built once as a registered buffer at construction time. Each head's `[N_tokens, 196]` softmax attention map is reshaped to the 14×14 patch grid and depthwise-convolved with this kernel (`same` padding), then renormalized to sum to 1 per token before value-weighting — so smoothing redistributes attention mass spatially without changing its total. `MaskPredictionHead` applies no further smoothing (see below), so this is the only spatial regularization the mask logits — and hence $\mathcal{L}_\text{dice}$ — receive.

`MaskPredictionHead` projects the refined tokens and computes cosine similarity against the raw patch tokens, scaled by a fixed temperature `--mask_head_tau`, producing `mask_logits` $\in \mathbb{R}^{B \times N_\text{tokens} \times 196}$ (raw logits, no sigmoid).

**Foreground token selection** is detached (no gradient) and picks, per sample, the token whose sigmoid-activated overlap with the GT patch labels is highest (`compute_l_dice_ce`); at inference time (no GT available) the same selection instead uses each token's peak-to-mean activation ratio (`select_fg_token`) — the token most spatially concentrated on a small region, since a lesion-focused token has high peak activation relative to its mean while a background-firing token is closer to uniform.

### 2.3 Text Encoder — `BiomedCLIPTextEncoder`

| Component | Detail |
|-----------|--------|
| Backbone | PubMedBERT transformer (768-dim hidden states), fully frozen |
| CLS projection (`cls_proj`) | 2-layer MLP 768 → 640 → `--embed_dim`, GELU, L2-normalized |
| `--warm_start_projections` | Also copies `cls_proj` from BiomedCLIP's pretrained `text.proj` |

**Text encoding modes** (controlled by `--text_mode`; v2 supports 3 of v1's 4 modes — `concat` was dropped):

| Mode | Beurteilung ($\mathcal{L}_\text{ITA}$) | Befund ($\mathcal{L}_\text{sim}$) |
|------|------|--------|
| `phrase` (default) | Up to `--max_beur_phrases` (16) individual phrases | Up to `--max_bef_phrases` (16) individual phrases |
| `full` | Full text as single sequence (`--max_text_len`) | Full text as single sequence |
| `mixed` | Full text as single sequence (`--max_beur_text_len`, 256) | Up to `--max_bef_phrases` individual phrases |

Each phrase is tokenized to `phrase_tok_len=32` tokens; sequences are padded and masked. Samples missing the phrases required by the active `--text_mode` are dropped at dataset construction (`InternalDatasetV2` logs how many).

---

## 3. Loss Functions

### 3.1 $\mathcal{L}_\text{ITA}$ — Image–Text Alignment

Global-level contrastive alignment between CLS image embeddings (full crop) and Beurteilung (assessment) phrase/text embeddings. Computed once per batch, independently of `has_mask` — every sample contributes.

**Image-to-text (I2T) direction:**

$$\mathcal{L}_\text{ITA}^\text{i2t} = \text{KL}\!\left(\log\sigma\!\left(\frac{s_{ij}}{\tau}\right) \;\middle\|\; q^\text{phrase}_{ij}\right), \qquad s_{ij} = z_\text{img}^{(i)} \cdot \phi_j$$

with the **soft phrase-phrase target** anchored on text-text similarity (more reliable than image-image similarity given the high visual heterogeneity of bone tumors):

$$q^\text{phrase}_{ij} = \frac{\exp(\phi_i \cdot \phi_j / \tau_{s,\text{beur}})}{\sum_k \exp(\phi_i \cdot \phi_k / \tau_{s,\text{beur}})}$$

**Text-to-image (T2I) direction** — controlled by `--t2i_mode`:

| Mode | Target distribution $q^\text{t2i}$ |
|------|-------------------------------------|
| `image_image` (default) | $\text{softmax}(z_i \cdot z_j / \tau_{s,\text{img\_full}})$ — image-image similarity |
| `text_text` | Same soft targets as I2T |
| `descriptor` | Cosine similarity of 21-dim binary descriptor vectors |
| `infonce` | Hard one-hot: phrase → its source image (standard InfoNCE) |

$$\mathcal{L}_\text{ITA} = \mathcal{L}_\text{ITA}^\text{i2t} + \lambda_\text{t2i} \cdot \mathcal{L}_\text{ITA}^\text{t2i}$$

**Optional modifiers:** `--same_image_boost` adds a fixed logit boost to same-image phrase pairs in the soft target; `--reweight_by_n_phrases` normalizes each image's contribution equally regardless of its phrase count.

---

### 3.2 $\mathcal{L}_\text{sim}$ — Local Phrase–Patch Alignment (GLoRIA-style, GT-mask-restricted)

Computed only for the subset of the batch with **both** a valid GT segmentation mask and Befund phrases (`has_mask & has_befund`, `sim_valid`; skipped if fewer than 2 such samples). Runs on the same crop/patch grid as $\mathcal{L}_\text{dice}$ (no separate tight crop, unlike v1).

**GT-mask attention restriction:** phrase→patch attention is always restricted to patches the ground-truth mask labels as lesion (`plabels[sim_valid].bool()`, passed as `patch_mask`) — background patches are masked to $-\infty$ before the softmax below, in both I2T and T2I directions. This uses the **ground-truth mask directly**, never the `MaskTokenDecoder`'s predicted mask — the two are trained independently and only interact through the shared backbone gradient. Images with no lesion patch fall back to attending all patches (avoids a degenerate all-$-\infty$ softmax row).

**I2T — per-image attention-weighted context vector**, phrase $j$ of image $i$ attending only over image $i$'s own (lesion-restricted) patches:

$$\alpha_{jm}^{(i)} = \frac{\exp(p_m^{(i)} \cdot \phi_j / \tau_2)}{\sum_{m' \in \mathcal{M}_i} \exp(p_{m'}^{(i)} \cdot \phi_j / \tau_2)}, \qquad c_j^{(i)} = \text{normalize}\!\left(\sum_{m \in \mathcal{M}_i} \alpha_{jm}^{(i)} \, p_m^{(i)}\right)$$

$$\mathcal{L}_\text{sim}^\text{i2t} = \text{KL}\!\left(\log\sigma\!\left(\frac{c_j^{(i)} \cdot \phi_j}{\tau}\right) \;\middle\|\; q^\text{phrase}_{ij}\right)$$

**T2I direction** — same `--t2i_mode` options as $\mathcal{L}_\text{ITA}$, using cross-image attention features restricted the same way (each image's background patches blocked before softmax, if it has any lesion patches):

$$s^\text{t2i}_{jk} = \phi_j \cdot c_j^{(k)}, \qquad \mathcal{L}_\text{sim}^\text{t2i} = f(s^\text{t2i}, q^\text{t2i})$$

$$\mathcal{L}_\text{sim} = \mathcal{L}_\text{sim}^\text{i2t} + \lambda_\text{t2i} \cdot \mathcal{L}_\text{sim}^\text{t2i}$$

**Attention temperature** `--sim_attn_tau` ($\tau_2$, default 0.07).

---

### 3.3 $\mathcal{L}_\text{ortho}$ — Lesion–Background Orthogonality

Pushes lesion patch embeddings away from background patch embeddings in raw ViT feature space (before projection), computed for all samples with a valid mask (`has_mask`).

Each patch contributes to both the lesion and background mean proportionally to its **soft coverage fraction** (a patch 20% covered by the mask contributes 20% to $v_\text{lesion}$ and 80% to $v_\text{bg}$; unlike v1's hard 0/1 patch labels):

$$v_\text{lesion}^{(i)} = \frac{\sum_m w_m^{(i)} h_m^{(i)}}{\sum_m w_m^{(i)}}, \qquad v_\text{bg}^{(i)} = \frac{\sum_m (1-w_m^{(i)}) h_m^{(i)}}{\sum_m (1-w_m^{(i)})}, \qquad w_m^{(i)} \in [0,1]$$

$$\mathcal{L}_\text{ortho} = \frac{1}{N}\sum_{i=1}^{N}\left(\cos\!\left(v_\text{lesion}^{(i)},\, v_\text{bg}^{(i)}\right) + 1\right)$$

The $+1$ offset shifts the loss range from $[-1, 1]$ to $[0, 2]$ without affecting gradients. Samples where all weight falls on one side are excluded.

**BTXRD supplementation:** unless `--no_btxrd` is set, an external dataset (`BTXRDOrthoDataset`, polygon/rectangle annotations) is cycled in parallel (`--btxrd_batch_size`) to provide an additional $\mathcal{L}_\text{dice}$ + $\mathcal{L}_\text{ortho}$ term each step.

---

### 3.4 $\mathcal{L}_\text{dice}$ — Mask Decoder Segmentation Loss

Trains `MaskTokenDecoder` + `MaskPredictionHead` against the ground-truth patch coverage labels, for all samples with a valid mask (`compute_l_dice_ce`). The foreground token is selected per-sample (detached) as the token with highest sigmoid-overlap against GT, then three terms are computed on that token's logits:

- **Dice** — ratio-based overlap between predicted probability and GT coverage, shapes the prediction globally.
- **Global BCE** — dense per-patch gradient over all 196 patches for stable learning.
- **Hard-negative BCE** — additional penalty on the top `hard_neg_k=10` most-activated *background* patches (GT = 0), weighted by `hard_neg_weight=0.5`. Targets leakage onto adjacent bone that global BCE alone under-penalizes because easy correct patches dilute its gradient.

$$\mathcal{L}_\text{dice} = \mathcal{L}_\text{dice-ratio} + \mathcal{L}_\text{BCE} + 0.5 \cdot \mathcal{L}_\text{hard-neg}$$

Logged separately as `l_dice_only`, `l_bce`, `l_hard_neg` alongside the combined `l_dice`.

The predicted `mask_logits` feeding into this loss inherit their spatial smoothness entirely from `MaskTokenDecoder`'s Gaussian-smoothed cross-attention (§2.2, `--gauss_sigma`) — no smoothing is applied downstream in `MaskPredictionHead` or in the loss itself. Larger `--gauss_sigma` biases predicted masks toward compact, contiguous blobs before dice/BCE are even computed; setting it to `0` disables the kernel and lets attention (and thus the mask logits) scatter freely across patches.

---

## 4. Training Procedure

### 4.1 Two-Pass Optimizer Step

Each batch is split into **two independent backward passes** (separate `optimizer.zero_grad()` / `scaler.step()` calls):

1. **Pass 1 — $\mathcal{L}_\text{ITA}$** (only if `"ita"` is active in the current stage): one forward pass through the ViT CLS token + text encoder.
2. **Pass 2 — $\mathcal{L}_\text{sim} + \mathcal{L}_\text{ortho} + \mathcal{L}_\text{dice}$**: a second, separate forward pass (`vit.forward_all`) computing patch tokens once, shared by the mask decoder, `L_sim`, and `L_ortho`; plus the optional BTXRD batch's `L_dice`/`L_ortho` contribution.

### 4.2 Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, two param groups (weight-decay vs. no-decay: biases, norm weights) |
| Learning rate | `--lr` (default 5×10⁻⁵) |
| Weight decay | `--weight_decay` (default 0.2), 0.0 for norm/bias params |
| Betas | (0.9, 0.98) |
| Epsilon | 1×10⁻⁶ |
| Gradient clipping | max norm = 1.0, applied separately after each pass |
| Precision | Mixed (FP16 autocast forward, FP32 backward via `GradScaler`) |

**LR schedule — reset at the stage 1 → stage 2 transition:** stage 1 gets its own warmup (`max(1, stage1_epochs // 5)`) and schedule over `stage1_epochs`; when stage 2 begins, a **fresh** scheduler is built with warmup `max(1, (epochs - stage1_epochs) // 5)` over the remaining epochs. `--scheduler cosine` anneals $\eta_t = 0.1 + 0.9 \cdot 0.5(1 + \cos(\pi \cdot \text{progress}))$ after warmup; default (`constant`) holds at `lr` after warmup.

### 4.3 Curriculum

| Arg | Behaviour |
|-----|-----------|
| `--stage1_epochs N` (default 10) | Epochs 1..N run only the losses assigned to stage 1. `N=0` disables the curriculum — every stage-2-assigned loss runs from epoch 1. |
| `--loss_stages` | Explicit per-loss stage assignment, tokens `NAME:STAGES` over `{ita, sim, ortho, dice}` and stages `{1,2}`/`0`/`none`. Overrides the default entirely; unlisted losses are off in every stage. |
| Default (no `--loss_stages`) | `dice:{1,2}`, `ortho:{1,2}`, `ita:{2}`, `sim:{2}` — mask decoder warms up on dice+ortho before the image-text losses switch on. |

### 4.4 Loss Weight Options

| Flag | Behaviour |
|------|-----------|
| Fixed $\lambda_*$ (default) | `--lambda_ita`, `--lambda_sim`, `--lambda_ortho`, `--lambda_dice` set as hyperparameters |
| `--learn_loss_weights` | $\log\lambda_*$ (all four) are unconstrained learnable parameters, updated by AdamW alongside the network |

### 4.5 Checkpointing & Early Stopping

| Checkpoint | Trigger |
|------------|---------|
| `best_checkpoint.pt` | Lowest validation loss — reset at the stage 1→2 transition, since the loss scale changes once sim+ita activate |
| `best_retrieval_checkpoint.pt` | Highest mean R@1 (I2T + T2I average) — **primary metric**, tracked globally across both stages (not reset at the transition, since retrieval scale is stable) |
| `final_checkpoint.pt` | End of training (optimizer state stripped) |

Early stopping: patience = `--patience` (default 20) epochs without mean R@1 improvement; the counter resets when stage 2 begins.

### 4.6 Per-Epoch Evaluation

Beyond the four losses (val split), every epoch also computes:
- **Retrieval** (`evaluate_retrieval_lace`): I2T/T2I R@1, R@5, median rank, and their mean — the early-stopping metric.
- **kNN probe** (`evaluate_knn_probe`): leave-one-out kNN (k=5, k=20) on CLS embeddings against malignancy labels, reporting balanced accuracy and macro-F1 — a cheap proxy for downstream classification quality without training a head.
- **Heatmap visualization**: for 10 fixed validation samples with GT masks, saves the foreground mask token's predicted heatmap overlaid on the image + GT contour + GT patch bounding boxes, to `run_dir/heatmaps/epoch_NNN/` (and W&B if enabled).

---

## 5. Data

### 5.1 Internal Dataset — `InternalDatasetV2`

Each sample provides:
- `input_image` $\in \mathbb{R}^{3 \times 224 \times 224}$ — the single crop (see §2.1)
- `patch_labels` $\in [0,1]^{196}$ — soft per-patch lesion coverage fraction (zeros if `has_mask=False`)
- `has_mask`, `has_befund` (+ `has_beurteilung` in `full` mode) — validity flags
- Tokenized phrase/text arrays per `--text_mode` (see §2.3)
- `label` — malignancy class, for the kNN probe only (not used by any pretraining loss)

Samples missing text required by the active `--text_mode` are dropped at construction. Splits are loaded from `--splits` (a `split.json` with `train`/`val` keys), or per-fold in CV mode (§5.3).

### 5.2 BTXRD Dataset — `BTXRDOrthoDataset`

External public dataset used for the auxiliary $\mathcal{L}_\text{dice}$ + $\mathcal{L}_\text{ortho}$ term (§3.3, §3.4). Disabled by `--no_btxrd`. Provides polygon/rectangle mask annotations; cycled indefinitely (reshuffling every wraparound) via `--btxrd_batch_size` (default 128).

### 5.3 Cross-Validation Mode — `--cv_dir`

Passing `--cv_dir <folder>` (e.g. `data/internal_dataset/cv_binary`) runs one full pretraining pass per fold split file matching `--cv_pattern` (default `split_binary_fold*.json`), instead of a single run. Each fold gets its own model/optimizer instance and writes checkpoints + heatmaps to `run_<timestamp>/fold<N>/`; the BTXRD loader is fold-independent and built once, reused across folds. Mutually exclusive with `--splits`.

---

## 6. Key Hyperparameters Summary

| Arg | Default | Notes |
|-----|---------|-------|
| `--image_encoder` | `biomedclip` | `biomedclip` \| `chexfound` |
| `--lora_layers` / `--lora_r` / `--lora_alpha` | 4 / 8 / 16.0 | Ignored if `--unfreeze_layers > 0` |
| `--unfreeze_layers` | 0 | biomedclip only; full fine-tune of last N blocks instead of LoRA |
| `--embed_dim` | 512 | |
| `--n_mask_tokens` / `--n_mask_heads` | 4 / 8 | Mask decoder capacity |
| `--gauss_sigma` | 1.5 | Cross-attention spatial smoothing; `0` disables |
| `--mask_head_tau` | 1.0 | Mask logit temperature |
| `--sim_attn_tau` | 0.07 | $\mathcal{L}_\text{sim}$ attention softmax temperature |
| `--batch_size` / `--btxrd_batch_size` | 128 / 128 | |
| `--text_mode` | `phrase` | `full` \| `phrase` \| `mixed` |
| `--t2i_mode` | `image_image` | `image_image` \| `text_text` \| `descriptor` \| `infonce` |
| `--context_fraction` / `--context_mode` / `--min_crop_size` | 0.15 / `image` / 224 | Single-crop geometry (§2.1) |
| `--stage1_epochs` | 10 | `0` disables the curriculum |
| `--epochs` | 40 | Total (stage1 + stage2) |
| `--lr` / `--weight_decay` | 5e-5 / 0.2 | |
| `--patience` | 20 | On retrieval mean R@1 |
| `--tau_s_beur` / `--tau_s_bef` / `--tau_s_img_full` | 0.015 / 0.07 / 0.07 | Soft-target temperatures |
| `--lambda_ita` / `--lambda_sim` / `--lambda_ortho` / `--lambda_dice` | 1.0 / 1.0 / 0.1 / 1.0 | Overridden by `--learn_loss_weights` |
| `--loss_stages` | (default curriculum) | See §4.3 |
| `--no_btxrd` | off | Disables BTXRD entirely |
| `--cv_dir` / `--cv_pattern` | off | See §5.3 |

Hyperparameter search: W&B Bayes optimization (`sweep_pretrain_v2.yaml`, `sweep_pretrain_v2_full.yaml`), maximising `retrieval/mean_r1` on the validation set.

---

## 7. File Map

| File | Contents |
|------|----------|
| `train/pretrain_v2.py` | Training loop, curriculum/stage logic, loss orchestration, checkpointing, heatmap visualization |
| `models/encoders.py` | `SharedViT` (biomedclip), `CheXFoundSharedViT`, `BiomedCLIPTextEncoder`, `ProjectionHead` |
| `models/lora.py` | `inject_lora_vit`, `unfreeze_last_n_vit` |
| `models/mask_tokens.py` | `MaskTokenDecoder`, `MaskPredictionHead`, `select_fg_token` |
| `loss/objectives.py` | `symmetric_soft_semantic_loss` ($\mathcal{L}_\text{ITA}$), `gloria_local_loss` ($\mathcal{L}_\text{sim}$), `ortho_loss`, `compute_l_dice_ce` |
| `data/datasets.py` | `InternalDatasetV2`, `BTXRDOrthoDataset` |
| `data/transforms.py` | Image augmentation + `crop_around_mask_pair`, `mask_to_patch_labels` |
| `eval/retrieval.py` | `evaluate_retrieval_lace` — R@1, R@5, median rank |
| `eval/knn_probe.py` | `evaluate_knn_probe` — leave-one-out kNN balanced accuracy / macro-F1 |
| `train/sweep_pretrain_v2.yaml`, `sweep_pretrain_v2_full.yaml` | W&B Bayes sweep configurations |

**v1** (`train/pretrain.py`, tight-crop-based $\mathcal{L}_\text{sim}$, no mask decoder, no curriculum) is superseded by v2 and no longer actively developed; see git history for its design if needed.
