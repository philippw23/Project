# Lesion-Grounded Vision-Language Classification of Bone Tumor Malignancy

Domain-adapts [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) on a bone-tumor X-ray dataset via contrastive pretraining with LoRA, then trains a downstream MLP classifier for 3-class malignancy prediction (benign / intermediate / malignant).

---

## Repository layout

```
src/
├── biomedclip/              # Core training package
│   ├── data/                #   Dataset, splits, image transforms
│   ├── distributed/         #   Multi-GPU (DDP) helpers
│   ├── eval/                #   Retrieval metrics (R@1, R@5, …)
│   ├── loss/                #   Contrastive + classification losses
│   ├── models/              #   LoRA injection, MLP classifier head
│   ├── train/               #   pretrain.py, downstream.py, sweep YAMLs
│   └── utils/               #   Checkpointing, architecture printing
├── biomedclip_pretrain.py   # Entry point → biomedclip.train.pretrain
├── biomedclip_downstream.py # Entry point → biomedclip.train.downstream
│
├── qwen_llm_extractor/      # LLM phrase-extraction package
│   ├── data/                #   Report loading (joint / separated)
│   ├── eval/                #   Phrase statistics, medbert comparison
│   ├── extract/             #   joint.py, separated.py (pipelines)
│   ├── models/              #   HuggingFace model loader
│   ├── prompts/             #   Prompt templates (joint / separated)
│   └── utils/               #   JSON repair utilities
├── llm_extractor.py         # Entry point → qwen_llm_extractor.extract.joint
├── llm_extractor_seperated.py # Entry point → qwen_llm_extractor.extract.separated
│
├── medbert_extractor.py     # KeyBERT / MedBERT baseline phrase extractor
├── preprocess_crop_images.py  # Mask-guided image cropping
├── preprocess_reports.py    # Report cleaning and translation pipeline
├── translate_reports.py     # German → English report translation
├── build_dataset_json.py    # Assemble dataset JSON from raw sources
└── visualize_samples.py     # Sample visualisation

sbatch/                      # SLURM job scripts
data/
├── images/                  # X-ray images
├── segmentations/           # Bone-tumor segmentation masks
└── text/                    # translated_reports.json, sanitized_reports.json
results/                     # Checkpoints, CSVs, W&B artefacts
```

---

## Pipeline overview

### 1 — Data preprocessing

```bash
# Crop images around segmentation masks
sbatch sbatch/run_check_dataset.sh
python src/preprocess_crop_images.py

# Translate German reports to English
sbatch sbatch/run_translate_reports.sh
```

### 2 — Phrase extraction (optional, for evidence-phrase experiments)

**LLM-based extraction** using Qwen2.5-7B-Instruct:
```bash
# Joint extraction (befund + beurteilung in one prompt)
sbatch sbatch/run_llm_extractor.sh

# Separated extraction (separate prompts per section)
python src/llm_extractor_seperated.py \
    --reports data/text/sanitized_reports.json \
    --out_dir results/
```

**KeyBERT / MedBERT baseline**:
```bash
sbatch sbatch/run_medbert_extractor.sh
```

### 3 — BiomedCLIP pretraining

Fine-tunes the ViT image encoder of BiomedCLIP with LoRA using contrastive (CLIP-style) loss on (image, report) pairs. The PubMedBERT text encoder is kept frozen.

```bash
# Single GPU
sbatch sbatch/run_biomedclip_pretrain.sh

# Multi-GPU — edit run_biomedclip_pretrain.sh to enable torchrun block
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--lora_layers` | 4 | Number of ViT blocks to inject LoRA into (from last) |
| `--lora_r` | 8 | LoRA rank |
| `--lora_alpha` | 64 | LoRA scaling factor |
| `--use_mask` | off | Crop input images to segmentation mask |
| `--epochs` | 100 | Training epochs |
| `--lr` | 5e-5 | Learning rate |
| `--distributed` | off | Enable multi-GPU DDP via torchrun |

Outputs written to `results/biomedclip_pretrain/<run>/`:
- `best_r1_checkpoint.pt` — best checkpoint by image→text R@1
- `splits.json` — train/val/test patient split

**Hyperparameter sweep** (W&B Bayes, optimises `retrieval/mean_r1`):
```bash
wandb sweep src/biomedclip/train/sweep_pretrain.yaml
# Set SWEEP_ID in run_sweep_pretrain.sh, then:
sbatch sbatch/run_sweep_pretrain.sh
```

### 4 — Downstream malignancy classification

Loads the frozen pretrained image encoder, appends clinical metadata (age, sex), and trains a small MLP for 3-class malignancy prediction.

```bash
sbatch sbatch/run_biomedclip_downstream.sh
```

Key arguments:

| Argument | Description |
|---|---|
| `--checkpoint` | Path to pretrain checkpoint |
| `--splits` | Path to splits.json from pretraining |
| `--use_mask` | Use mask-cropped images (should match pretraining) |
| `--epochs` | Fine-tuning epochs |

**Hyperparameter sweep**:
```bash
wandb sweep src/biomedclip/eval/sweep_downstream.yaml
sbatch sbatch/run_sweep_downstream.sh
```

---

## Model

**Base model**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`  
**Image encoder**: ViT-B/16 — LoRA injected into last N transformer blocks  
**Text encoder**: PubMedBERT — frozen  
**Classifier head**: MLP on `[image_embedding(512) | age(1) | sex(1)]` → 3 classes

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `open_clip_torch`, `transformers`, `accelerate`, `bitsandbytes`, `keybert`, `scikit-learn`, `wandb`

---

## Data

Expected layout under `data/`:

```
data/
├── metadata.xlsx            # Patient metadata (patid, age, sex, malignancy label)
├── images/                  # One image per study
├── segmentations/           # Corresponding segmentation masks
└── text/
    ├── sanitized_reports.json     # Cleaned German reports
    └── translated_reports.json   # English translations
```
