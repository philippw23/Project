# Lesion-Grounded Vision-Language Classification of Bone Tumor Malignancy

Domain-adapts vision-language and vision-only models to a private bone-tumor X-ray dataset, then trains a downstream MLP classifier for malignancy prediction (3-class benign / intermediate / malignant, or binary benign / malignant against the external [BTXRD](data/BTXRD) test set).

Implements **LACE** (Lesion-Aligned Contrastive Embedding) — a novel curriculum-learning pretraining approach developed in this project — alongside baselines built on [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (including a GLoRIA-style local-loss variant, `biomedclip_gloria`), **CheXFound** (DINO+iBOT), **GLoRIA**, and a frozen **ImageNet** encoder.

---

## Repository layout

```
src/
├── LACE/                     # Curriculum-learning pretraining (novel contribution)
│   ├── data/                 #   Datasets, splits, transforms
│   ├── eval/                 #   kNN probe, retrieval metrics
│   ├── loss/                 #   L_ITA, L_sim, L_ortho, L_dice, L_evid_p, L_rec
│   ├── models/                #   SharedViT, text encoder, LoRA, mask decoder, prototypes
│   ├── train/                #   pretrain_v2.py, downstream.py, sweep YAMLs
│   └── ARCHITECTURE.md       #   Design writeup (objective, model, losses)
├── lace_pretrain_v2.py                        # Entry point
├── lace_downstream.py                         # Downstream head training
├── lace_downstream_eval.py                    # Score a saved head, no training
├── lace_img_text_downstream.py                # Downstream variant that also fuses text embeddings
│
├── biomedclip/                # BiomedCLIP contrastive pretraining package
│   ├── data/ distributed/ eval/ loss/ models/ train/ utils/
├── biomedclip_pretrain.py / biomedclip_downstream.py / biomedclip_downstream_eval.py
├── biomedclip_zeroshot.py / biomedclip_img_text_downstream.py
│
├── biomedclip_gloria/          # BiomedCLIP pretraining + GLoRIA-style local loss
├── biomedclip_gloria_pretrain.py   # Entry point (downstream reuses biomedclip_downstream.py)
│
├── chexfound/                 # CheXFound DINO+iBOT continued pretraining
├── chexfound_downstream.py / chexfound_downstream_eval.py
│
├── gloria/                    # GLoRIA (separate old env — see Requirements)
├── gloria_downstream.py / gloria_pretrain.py / gloria_downstream_eval.py
│
├── imagenet_img/               # Frozen ImageNet ViT-B/16 linear probe
├── imagenet_img_downstream.py / imagenet_img_downstream_eval.py
│
├── downstream_cv.py             # 10-fold CV + BTXRD eval orchestrator (shared across baselines)
│
├── qwen_llm_extractor/         # LLM phrase-extraction package (Qwen2.5-7B-Instruct)
│   ├── data/ eval/ extract/ models/ prompts/ utils/
├── llm_extractor.py            # Entry point → joint extraction
├── llm_extractor_seperated.py  # Entry point → separated (befund / beurteilung) extraction
│
├── preprocess_images.py        # Square-pad images/masks (internal + BTXRD)
├── preprocess_reports.py / translate_reports.py   # Report cleaning & translation
├── create_dataset.py           # Assemble dataset_full.json
├── create_split.py             # Patient-level stratified train/val/test split
├── create_cv_splits.py         # Derive patient-grouped 10-fold CV pool from split.json
├── build_btxrd_downstream.py   # Convert BTXRD into the internal sample-dict manifest format
├── check_dataset.py            # Reports what images/masks/metadata are missing
└── visualize_samples.py        # Sample visualisation

sbatch/                         # SLURM job scripts, mirrors src/ (biomedclip/, biomedclip_gloria/,
                                 # lace/, chexfound/, gloria/, imagenet_img/, data/, utils/)
data/
├── internal_dataset/
│   ├── images/ segmentations/ metadata.xlsx
│   ├── text/                 # full_reports.json and translation/extraction variants
│   ├── dataset_full.json     # assembled dataset
│   ├── split.json            # canonical train/val/test manifest (split_binary.json for binary)
│   └── cv/                   # 10-fold CV split pool
└── BTXRD/                     # external test set (images, annotations, downstream manifest)
results/                       # Checkpoints, CSVs, W&B artefacts (.gitignore'd)
```

> Several dated/backup files alongside the canonical ones above (e.g. `dataset_full_14B_*.json`, `split_final.json`, `full_reports_new.json`) are scratch snapshots, not defaults — check [src/biomedclip/utils/misc.py](src/biomedclip/utils/misc.py) for the paths actually used by default.

---

## Pipeline overview

### 1 — Data preprocessing

```bash
# Check for missing images / masks / metadata
sbatch sbatch/data/run_check_dataset.sh

# Square-pad images and masks to preprocessed_images/
sbatch sbatch/data/run_preprocess_images.sh

# Translate German reports to English
sbatch sbatch/data/run_translate_reports.sh
```

### 2 — Phrase extraction (optional, for evidence-phrase experiments)

LLM-based extraction using Qwen2.5-7B-Instruct:
```bash
# Joint extraction (befund + beurteilung in one prompt)
sbatch sbatch/data/run_llm_extractor.sh

# Separated extraction (separate prompts per section)
sbatch sbatch/data/run_llm_extractor_seperated.sh
```

### 3 — Dataset assembly & split creation

Two scripts turn the raw metadata, reports, images, and masks into a fixed patient-level split that is reused by pretraining and every baseline, which keeps comparisons fair.

**Step 1 — assemble `dataset_full.json`:**
```bash
python src/create_dataset.py
```
Matches each `metadata.xlsx` row to its report in `text/full_reports.json` by `report_accnr` (fallback `patid`), keeping a row only if the image exists on disk. Writes one entry per image (image, mask, befund / beurteilung / phrases, label, age, sex, patid) to `data/internal_dataset/dataset_full.json`.

**Step 2 — create `split.json`:**
```bash
sbatch sbatch/data/run_create_split.sh          # add --binary for benign-vs-malignant only
```
Filters to entries with a report, a label, and a non-empty segmentation mask; normalises sex (m/f → 1/0, unknown → 0.5) and imputes unknown age with the training-set mean. Splits at the **patient level** (no train/val/test leakage), **stratified by each patient's majority label**, into train / val / test (default 0.8 / 0.1 / 0.1, `seed=42`). Writes `data/internal_dataset/split.json` (or `split_binary.json`).

**Step 3 — (optional) create a 10-fold CV pool:**
```bash
sbatch sbatch/data/run_create_cv_splits.sh
```
Divides the existing `train` split into 8 stratified, patient-grouped parts and rotates the given `val`/`test` in as parts 9/10, writing 10 fold files to `data/internal_dataset/cv/` for cross-validated downstream evaluation.

> **Note:** `split.json` embeds the phrases copied from `dataset_full.json`. After changing phrase extraction, re-run **both** `create_dataset.py` and the split step for the change to take effect — editing prompts alone does not update existing splits.

### 4 — Pretraining

Every approach fine-tunes an image encoder via a self-/weakly-supervised objective on (image, report) pairs or images alone, then freezes it for the downstream classifier.

```bash
# LACE (novel curriculum approach)
sbatch sbatch/lace/run_lace_pretrain_v2.sh

# BiomedCLIP contrastive (LoRA) pretraining
sbatch sbatch/biomedclip/run_biomedclip_pretrain.sh

# CheXFound continued DINO+iBOT pretraining
sbatch sbatch/chexfound/run_chexfound_pretrain.sh

# GLoRIA pretraining (own environment — see Requirements)
sbatch sbatch/gloria/run_gloria_pretrain.sh

# BiomedCLIP + GLoRIA-style local loss pretraining variant
sbatch sbatch/biomedclip_gloria/run_biomedclip_gloria_pretrain.sh
```

BiomedCLIP key arguments:

| Argument | Default | Description |
|---|---|---|
| `--lora_layers` | 4 | Number of ViT blocks to inject LoRA into (from last) |
| `--lora_r` | 8 | LoRA rank |
| `--lora_alpha` | 16 | LoRA scaling factor |
| `--use_mask` | off | Crop input images to segmentation mask |
| `--epochs` | 50 | Training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--batch_size` | 32 | Per-GPU batch size |

LACE v2 additionally exposes `--loss_stages` to assign each loss (`ita`, `sim`, `ortho`, `dice`, `rec`, `evid`) to specific curriculum stages or disable it — see [src/LACE/ARCHITECTURE.md](src/LACE/ARCHITECTURE.md) for the full objective.

Outputs are written to `results/<approach>_pretrain/<run>/`, typically including the best checkpoint by retrieval R@1 and the `splits.json` used.

**Hyperparameter sweeps** (W&B Bayes):
```bash
wandb sweep src/LACE/train/sweep_pretrain_v2.yaml     # or biomedclip/train/sweep_pretrain_lora.yaml, etc.
# Set SWEEP_ID in the matching sbatch/run_sweep_pretrain.sh, then:
sbatch sbatch/run_sweep_pretrain.sh
```

### 5 — Downstream malignancy classification

Every approach follows the same pattern: `<name>_downstream.py` loads the frozen pretrained encoder, appends clinical metadata (age, sex), and trains a small MLP for malignancy prediction; `<name>_downstream_eval.py` loads a saved head checkpoint and re-scores it (against a split and/or the BTXRD manifest) with no training.

```bash
sbatch sbatch/lace/run_lace_downstream_v2.sh
sbatch sbatch/biomedclip/run_biomedclip_downstream.sh     # also used for the biomedclip_gloria checkpoint
sbatch sbatch/chexfound/run_chexfound_downstream.sh
sbatch sbatch/gloria/run_gloria_downstream.sh
sbatch sbatch/imagenet_img/run_imagenet_img.sh
```

Key arguments (vary slightly per approach):

| Argument | Description |
|---|---|
| `--checkpoint` | Path to pretrain checkpoint |
| `--splits` | Path to the split manifest to train/evaluate on |
| `--use_mask` | Use mask-cropped images (should match pretraining) |
| `--binary` | Benign-vs-malignant only (required to also score against BTXRD) |
| `--epochs` | Fine-tuning epochs |

**10-fold cross-validation** (fixed hyperparameters, reports mean ± std across folds and the external BTXRD test set): every baseline shares the same orchestrator, [src/downstream_cv.py](src/downstream_cv.py) (`--baseline lace|lace_img_text|biomedclip|biomedclip_img_text|chexfound|gloria|imagenet`), wrapped per approach as `sbatch/<approach>/run_<approach>_downstream_cv*.sh`:
```bash
sbatch sbatch/lace/run_lace_downstream_cv.sh
sbatch sbatch/biomedclip/run_biomedclip_downstream_cv_frozen.sh
sbatch sbatch/chexfound/run_chexfound_downstream_cv_frozen.sh
sbatch sbatch/gloria/run_gloria_downstream_cv_3class_frozen.sh
```

**Hyperparameter sweeps**:
```bash
wandb sweep src/LACE/train/sweep_downstream_v2.yaml
sbatch sbatch/run_sweep_downstream.sh
```

---

## Model

**LACE** (novel): shared ViT (BiomedCLIP trunk + LoRA) and PubMedBERT text encoder, jointly optimizing global image-text alignment, local lesion-phrase alignment, and (v2) lesion segmentation + prototype/evidential losses in a staged curriculum.

**BiomedCLIP baseline**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, ViT-B/16 image encoder with LoRA injected into the last N transformer blocks, PubMedBERT text encoder frozen.

**Downstream head** (shared across approaches): MLP on `[image_embedding | age | sex]` → malignancy classes.

---

## Requirements

There is no single top-level `requirements.txt`; install the key packages manually into your environment:

```bash
pip install torch transformers open_clip_torch accelerate bitsandbytes "xformers==0.0.28.post3" scikit-learn wandb
```

GLoRIA has its own older, self-contained environment (pytorch-lightning 1.1.4, torch 1.7.1):
```bash
conda env create -f src/gloria/environment.yml
pip install -e src/gloria/
```

---

## Data

Expected layout under `data/internal_dataset/`:

```
data/internal_dataset/
├── metadata.xlsx             # Patient metadata (patid, age, sex, malignancy label)
├── images/                   # One image per study
├── segmentations/            # Corresponding segmentation masks
├── dataset_full.json         # Assembled dataset (see create_dataset.py)
├── split.json                # Canonical train/val/test manifest
├── cv/                       # 10-fold CV split pool
└── text/
    ├── sanitized_reports.json   # Cleaned German reports
    └── full_reports.json       # English translations + extracted phrases
```

`data/BTXRD/` holds the external test set (images, annotations, `btxrd_downstream_binary.json`), used only for binary benign/malignant evaluation.
