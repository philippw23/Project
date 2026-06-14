# LACE v1 Architecture

**L**esion-**A**ligned **C**ontrastive **E**mbedding — pretraining framework for bone-tumor X-ray / radiology-report alignment.

---

## 1. Overview

LACE v1 adapts BiomedCLIP (ViT-B/16 + PubMedBERT) to a private internal bone-tumor X-ray dataset via a joint contrastive objective with three components:

$$\mathcal{L}_\text{total} = \lambda_\text{ita} \cdot \mathcal{L}_\text{ITA} + \lambda_\text{sim} \cdot \mathcal{L}_\text{sim} + \lambda_\text{reg} \cdot \mathcal{L}_\text{ortho}$$

All three losses are active from epoch 1 (no staged curriculum in v1). The weights $\lambda_*$ are either fixed or learned as unconstrained log-scale parameters optimized jointly with the rest of the network.

---

## 2. Model Architecture

### 2.1 Visual Encoder — `SharedViT`

| Component | Detail |
|-----------|--------|
| Backbone | BiomedCLIP ViT-B/16 trunk, frozen base weights |
| LoRA injection | Last `lora_layers` ∈ {2, 4, 6} transformer blocks |
| LoRA targets | Attention Q/K/V projections, attention output, MLP fc1/fc2 |
| LoRA rank / scale | $r$ ∈ {4, 8, 16}, $\alpha = 16$ (fixed) |
| Trainable fraction | ~1–2% of total parameters |
| CLS projection (`img_proj`) | Linear 768 → 512, L2-normalized → global image embedding $z_\text{img}$ |
| Patch projection (`patch_proj`) | Linear 768 → 512, L2-normalized → patch embeddings $\{p_m\}_{m=1}^{196}$ |

Two image crops are computed per sample:

| Crop | `context_fraction` | Used in |
|------|--------------------|---------|
| **Global crop** | 0.3–0.4 | $\mathcal{L}_\text{ITA}$, $\mathcal{L}_\text{ortho}$ |
| **Tight crop** (lesion-centred) | 0.1–0.15 | $\mathcal{L}_\text{sim}$ |

### 2.2 Text Encoder — `BiomedCLIPTextEncoder`

| Component | Detail |
|-----------|--------|
| Backbone | PubMedBERT transformer (768-dim hidden states), fully frozen |
| CLS projection (`cls_proj`) | 2-layer MLP 768 → 640 → 512, GELU, L2-normalized |
| Word projection (`word_proj`) | Same architecture, for token-level features |

**Text encoding modes** (controlled by `--text_mode`):

| Mode | Beurteilung | Befund | Notes |
|------|-------------|--------|-------|
| `phrase` (default) | Up to 16 individual phrases, each tokenized separately | Up to 16 individual phrases | Phrase-level granularity for soft targets |
| `full` | Full text as single sequence | Full text as single sequence | Coarser alignment |
| `concat` | All phrases concatenated into one sequence | All phrases concatenated | Middle ground |
| `mixed` | Full text as single sequence (`max_beur_text_len=256`) | Up to 16 phrases | Hybrid |

Each phrase is tokenized to `phrase_tok_len=32` tokens; sequences are padded and masked.

---

## 3. Loss Functions

### 3.1 $\mathcal{L}_\text{ITA}$ — Image–Text Alignment

Global-level contrastive alignment between CLS image embeddings and Beurteilung (assessment) phrase embeddings.

**Image-to-text (I2T) direction:**

$$\mathcal{L}_\text{ITA}^\text{i2t} = \text{KL}\!\left(\log\sigma\!\left(\frac{s_{ij}}{\tau}\right) \;\middle\|\; q^\text{phrase}_{ij}\right)$$

where the pairwise similarity is:

$$s_{ij} = z_\text{img}^{(i)} \cdot \phi_j, \quad \phi_j = \text{cls\_proj}(\text{CLS}_j)$$

and the **soft phrase-phrase target** is:

$$q^\text{phrase}_{ij} = \frac{\exp(\phi_i \cdot \phi_j / \tau_{s,\text{beur}})}{\sum_k \exp(\phi_i \cdot \phi_k / \tau_{s,\text{beur}})}$$

This anchors the distribution on text-text similarity rather than image-image similarity — justified by the high visual heterogeneity of bone tumors, where text consistency is more reliable.

**Text-to-image (T2I) direction** — controlled by `--t2i_mode`:

| Mode | Target distribution $q^\text{t2i}$ |
|------|-------------------------------------|
| `image_image` (default) | $\text{softmax}(z_i \cdot z_j / \tau_{s,\text{img}})$ — image-image similarity |
| `text_text` | $\text{softmax}(\phi_i \cdot \phi_j / \tau_{s,\text{beur}})$ — same as I2T soft targets |
| `descriptor` | Cosine similarity of 21-dim binary descriptor vectors (MedCLIP-style) |
| `infonce` | Hard one-hot: phrase → its source image (standard InfoNCE) |

**Full I2T + T2I objective:**

$$\mathcal{L}_\text{ITA} = \mathcal{L}_\text{ITA}^\text{i2t} + \mathcal{L}_\text{ITA}^\text{t2i}$$

**Optional modifiers:**

| Flag | Effect |
|------|--------|
| `--same_image_boost` $b \in [0, 20]$ | Adds $b$ to logits $s_{ij}$ when phrase $j$ originates from the same image as $i$ |
| `--reweight_by_n_phrases` | Normalizes each image's contribution equally regardless of phrase count |

**Temperature parameters:**

| Symbol | Arg | Default | Range |
|--------|-----|---------|-------|
| $\tau$ | learned | 0.07 | clamped to $[0.01, 0.5]$ |
| $\tau_{s,\text{beur}}$ | `--tau_s_beur` | 0.04 | {0.01, 0.04, 0.08} |
| $\tau_{s,\text{bef}}$ | `--tau_s_bef` | 0.07 | {0.02, 0.04, 0.07} |
| $\tau_{s,\text{img}}$ | `--tau_s_img_full` | 0.07 | {0.03, 0.05, 0.07} |

---

### 3.2 $\mathcal{L}_\text{sim}$ — Local Phrase–Patch Alignment (GLoRIA-style)

Computed only for samples where both a segmentation mask and Befund (findings) phrases are available (`has_mask=True`, `has_befund=True`). Uses the **tight crop** image.

**Attention-weighted context vector** for phrase $j$ over image $i$:

$$\alpha_{jm}^{(i)} = \frac{\exp(p_m^{(i)} \cdot \phi_j / \tau_2)}{\sum_{m'} \exp(p_{m'}^{(i)} \cdot \phi_j / \tau_2)}, \qquad c_j^{(i)} = \text{normalize}\!\left(\sum_m \alpha_{jm}^{(i)} \, p_m^{(i)}\right)$$

**Image-to-text direction:**

$$\mathcal{L}_\text{sim}^\text{i2t} = \text{KL}\!\left(\log\sigma\!\left(\frac{c_j^{(i)} \cdot \phi_j}{\tau}\right) \;\middle\|\; q^\text{phrase}_{ij}\right)$$

**Text-to-image direction** — same `--t2i_mode` options as $\mathcal{L}_\text{ITA}$, but using cross-image attention features:

$$s^\text{t2i}_{jk} = \phi_j \cdot c_j^{(k)}, \qquad \mathcal{L}_\text{sim}^\text{t2i} = f(s^\text{t2i}, q^\text{t2i})$$

where $f$ is KL-divergence (for `image_image`, `text_text`, `descriptor`) or cross-entropy (for `infonce`).

**Full objective:**

$$\mathcal{L}_\text{sim} = \mathcal{L}_\text{sim}^\text{i2t} + \lambda_\text{t2i} \cdot \mathcal{L}_\text{sim}^\text{t2i}$$

**Attention temperature** $\tau_2$:

| Arg | Default | Sweep range |
|-----|---------|-------------|
| `--tau2` | 0.07 | {0.03, 0.05, 0.07, 0.10} |

---

### 3.3 $\mathcal{L}_\text{ortho}$ — Lesion–Background Orthogonality

Pushes lesion patch embeddings away from background patch embeddings in raw ViT space (768-dim, before projection).

For each image $i$ with a valid mask, let $\mathcal{M}_i$ and $\mathcal{B}_i$ be the sets of patch indices labelled as lesion and background respectively:

$$v_\text{lesion}^{(i)} = \frac{1}{|\mathcal{M}_i|}\sum_{m \in \mathcal{M}_i} h_m^{(i)}, \qquad v_\text{bg}^{(i)} = \frac{1}{|\mathcal{B}_i|}\sum_{m \in \mathcal{B}_i} h_m^{(i)}$$

where $h_m^{(i)} \in \mathbb{R}^{768}$ are raw ViT patch tokens.

$$\mathcal{L}_\text{ortho} = \frac{1}{N}\sum_{i=1}^{N}\left(\cos\!\left(v_\text{lesion}^{(i)},\, v_\text{bg}^{(i)}\right) + 1\right)$$

The $+1$ offset shifts the loss range from $[-1, 1]$ to $[0, 2]$ without affecting gradients. Samples with fewer than 1 lesion patch or 0 background patches are skipped.

**BTXRD supplementation:** An external dataset (`BTXRDOrthoDataset`) with polygon/rectangle annotations is cycled in parallel during training to provide additional supervision for $\mathcal{L}_\text{ortho}$.

---

## 4. Training Procedure

### 4.1 Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5×10⁻⁵ (log-uniform sweep: 5×10⁻⁶ – 2×10⁻⁴) |
| Weight decay | 0.2 (log-uniform sweep: 0.001 – 0.1) |
| Betas | (0.9, 0.98) |
| Epsilon | 1×10⁻⁶ |
| Gradient clipping | max norm = 1.0 |
| Precision | Mixed (FP16 forward, FP32 backward) |

**LR schedule:**

| Arg | Behaviour |
|-----|-----------|
| `warmup_epochs=5` | Linear warmup from 0 to `lr` |
| `--scheduler cosine` | Cosine annealing: $\eta_t = 0.1 + 0.9 \cdot 0.5(1 + \cos(\pi \cdot \text{progress}))$ |
| default (no `--scheduler`) | Constant after warmup |

### 4.2 Per-Epoch Forward Pass

1. **Global crop** → ViT → CLS token $z_\text{img}$, raw patch tokens $\{h_m\}$ (768-dim), projected patches $\{p_m\}$ (512-dim)
2. **Tight crop** → ViT → projected patches for $\mathcal{L}_\text{sim}$
3. **Text encoding** → up to 16 phrases per text type → $\{\phi_j\}$ (512-dim)
4. Compute $\mathcal{L}_\text{ITA}$ (if `"ita"` in `active_losses`)
5. Compute $\mathcal{L}_\text{sim}$ for masked samples (if `"sim"` in `active_losses`)
6. Compute $\mathcal{L}_\text{ortho}$ on global patches + optional BTXRD batch (if `"ortho"` in `active_losses`)
7. Backward, clip, step

### 4.3 Loss Weight Options

| Flag | Behaviour |
|------|-----------|
| Fixed $\lambda_*$ (default) | $\lambda_\text{ita}, \lambda_\text{sim}, \lambda_\text{reg}$ set as hyperparameters |
| `--learn_loss_weights` | $\log\lambda_*$ are unconstrained learnable parameters, updated by AdamW |

Default sweep values: $\lambda_\text{ita} \in \{0.5, 0.8, 1.0\}$, $\lambda_\text{sim} \in \{0.5, 0.8, 1.0\}$, $\lambda_\text{reg} \in \{0.3, 0.5, 0.8\}$.

### 4.4 Checkpointing & Early Stopping

| Checkpoint | Trigger |
|------------|---------|
| `best_val_loss_checkpoint` | Lowest combined validation loss |
| `best_retrieval_checkpoint` | Highest mean R@1 (I2T + T2I average) — **primary metric** |
| Final checkpoint | End of training |

Early stopping: patience = 25 epochs on mean R@1.

---

## 5. Data

### 5.1 Internal Dataset — `InternalTripleDataset`

Each sample provides:
- `global_crop` $\in \mathbb{R}^{3 \times 224 \times 224}$ — context-expanded image
- `crop_image` $\in \mathbb{R}^{3 \times 224 \times 224}$ — tight lesion-centred crop
- `patch_labels` $\in \{0,1\}^{196}$ — per-patch lesion/background label
- `has_mask`, `has_befund`, `has_beurteilung` — validity flags
- `descriptor_vec` $\in \{0,1\}^{21}$ — optional binary descriptor (for `descriptor` t2i_mode)
- Tokenized phrase arrays (see text modes above)

Splits: stratified 80/10/10 train/val/test, persisted to `run_dir/split.json`.

### 5.2 BTXRD Dataset — `BTXRDOrthoDataset`

External public dataset used exclusively for $\mathcal{L}_\text{ortho}$. Provides polygon/rectangle mask annotations. Cycled indefinitely via `itertools.cycle()` during training; batch size controlled by `--btxrd_batch_size` (default 128).

---

## 6. Key Hyperparameters Summary

| Arg | Default | Sweep |
|-----|---------|-------|
| `--lora_layers` | 4 | {2, 4, 6} |
| `--lora_r` | 8 | {4, 8, 16} |
| `--embed_dim` | 512 | — |
| `--batch_size` | 128 | — |
| `--btxrd_batch_size` | 128 | — |
| `--lr` | 5e-5 | log-uniform [5e-6, 2e-4] |
| `--weight_decay` | 0.2 | log-uniform [0.001, 0.1] |
| `--epochs` | 100 | — |
| `--warmup_epochs` | 5 | — |
| `--patience` | 25 | — |
| `--tau_s_beur` | 0.04 | {0.01, 0.04, 0.08} |
| `--tau2` | 0.07 | {0.03, 0.05, 0.07, 0.10} |
| `--lambda_ita` | 1.0 | {0.5, 0.8, 1.0} |
| `--lambda_sim` | 1.0 | {0.5, 0.8, 1.0} |
| `--lambda_reg` | 0.1 | {0.3, 0.5, 0.8} |
| `--text_mode` | `phrase` | {phrase, full, concat, mixed} |
| `--t2i_mode` | `image_image` | {image_image, text_text, descriptor, infonce} |
| `--context_fraction` | 0.15 | {0.10, 0.15} |
| `--global_context_fraction` | 0.4 | {0.30, 0.40} |

Hyperparameter search: W&B Bayes optimization over 30 runs, maximising mean R@1 on the validation set.

---

## 7. File Map

| File | Contents |
|------|----------|
| `train/pretrain.py` | Training loop, loss orchestration, checkpointing |
| `models/encoders.py` | `SharedViT`, `BiomedCLIPTextEncoder`, `ProjectionHead` |
| `models/lora.py` | `LoRALinear` injection |
| `loss/objectives.py` | `L_ITA`, `L_sim`, `L_ortho` implementations |
| `data/datasets.py` | `InternalTripleDataset`, `BTXRDOrthoDataset` |
| `data/transforms.py` | Image augmentation pipeline |
| `eval/retrieval.py` | `evaluate_retrieval_lace` — R@1, R@5, median rank |
| `train/sweep_pretrain_v1.yaml` | W&B Bayes sweep configuration |
