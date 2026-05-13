# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bone-tumor malignancy classification research project using domain-adapted vision-language models. The pipeline adapts BiomedCLIP (ViT-B/16 + PubMedBERT) to a private internal bone-tumor X-ray dataset via contrastive pretraining, then trains a downstream malignancy classifier (3-class: benign/intermediate/malignant).

Three baseline pretraining approaches are implemented (**BiomedCLIP**, **CheXFound** DINO+iBOT, **GLoRIA**) alongside **LACE**, a novel curriculum learning approach developed in this project.

## Running Experiments

All training is executed via SLURM on an HPC cluster. Scripts in `sbatch/` wrap the Python entry points in `src/`:

```bash
# BiomedCLIP contrastive pretraining
sbatch sbatch/run_biomedclip_pretrain.sh

# LACE curriculum pretraining
sbatch sbatch/run_lace_pretrain.sh

# CheXFound iBOT continued pretraining
sbatch sbatch/run_chexfound_pretrain.sh

# GLoRIA pretraining
sbatch sbatch/run_gloria_pretrain.sh

# Downstream malignancy classification
sbatch sbatch/run_downstream.sh

# LLM phrase extraction (Qwen2.5-7B)
sbatch sbatch/run_llm_extractor.sh

# W&B hyperparameter sweeps
sbatch sbatch/run_sweep_pretrain.sh
sbatch sbatch/run_sweep_downstream.sh
```

Python entry points can also be run directly:

```bash
python src/biomedclip_pretrain.py [args]
python src/lace_pretrain.py [args]
python src/downstream.py [args]
python src/gloria/run.py [config_path]
```

**There is no test suite.** Validation is done via W&B logging during training and manual inspection scripts (`src/check_dataset.py`, `src/visualize_samples.py`).

## Data Pipeline

Four-stage pipeline before training:

1. **Image preprocessing** — mask-guided cropping: `src/preprocess_images.py`
2. **Report translation** — German → English: `src/translate_reports.py`
3. **Phrase extraction** — LLM (Qwen2.5-7B): `src/llm_extractor.py`
4. **Dataset assembly** — `src/build_dataset_json.py`

Default data paths (hardcoded in [src/biomedclip/utils/misc.py](src/biomedclip/utils/misc.py)):

```
data/internal_dataset/images/          # bone tumor X-rays
data/internal_dataset/segmentations/   # lesion masks
data/internal_dataset/metadata.xlsx    # age, sex, labels
data/internal_dataset/text/translated_reports.json
results/                               # checkpoints, outputs
```

## Architecture

### BiomedCLIP (main approach)

- **Base model**: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **LoRA injection**: [src/biomedclip/models/lora.py](src/biomedclip/models/lora.py) — `LoRALinear` wraps selected ViT Q/V projection layers; text encoder is frozen
- **Contrastive loss**: symmetric InfoNCE in [src/biomedclip/loss/contrastive.py](src/biomedclip/loss/contrastive.py)
- **Downstream**: `MalignancyMLP` in [src/biomedclip/models/classifier.py](src/biomedclip/models/classifier.py) — late fusion of image embedding + age/sex metadata
- **Data**: `BoneTumorPairDataset` (image-text pairs) and `DownstreamDataset` in [src/biomedclip/data/datasets.py](src/biomedclip/data/datasets.py)
- **Splits**: stratified train/val/test via [src/biomedclip/data/splits.py](src/biomedclip/data/splits.py)

### LACE (curriculum learning)

Three-stage curriculum in [src/LACE/train/pretrain.py](src/LACE/train/pretrain.py):
1. **L_ITA** — image-text alignment with soft labels
2. **L_sim** — local lesion-phrase alignment via attention pooling
3. **L_ortho** — orthogonality regularization on BTXRD dataset

Encoders in [src/LACE/models/encoders.py](src/LACE/models/encoders.py): `SharedViT`, `BiomedCLIPTextEncoder`, `ProjectionHead`.

### CheXFound

Continued pretraining with DINO + iBOT joint objective ([src/chexfound/train/ssl_meta_arch.py](src/chexfound/train/ssl_meta_arch.py)). Uses block-based rectangular masking and KoLeo regularization. Config-driven via YAML files in [src/chexfound/configs/](src/chexfound/configs/).

### Downstream Classifier

[src/downstream.py](src/downstream.py) is a unified entry point that supports both `biomedclip` and `chexfound` encoders. The encoder backbone is frozen; only the MLP head trains. Image embeddings are pre-computed once and reused across epochs.

### Qwen LLM Extractor

[src/qwen_llm_extractor/](src/qwen_llm_extractor/) uses Qwen2.5-7B-Instruct (4-bit quantized via bitsandbytes) to extract anatomical phrases from German radiology reports. Two modes: joint (single prompt) and separated (befund + beurteilung separately). Prompts in [src/qwen_llm_extractor/prompts/](src/qwen_llm_extractor/prompts/).

## Key Hyperparameters

Relevant args for `biomedclip_pretrain.py`:

| Arg | Default | Notes |
|-----|---------|-------|
| `--lora_layers` | last 4 | Which ViT transformer blocks get LoRA |
| `--lora_r` | 8 | LoRA rank |
| `--lora_alpha` | 16 | LoRA scaling |
| `--batch_size` | 32 | Per-GPU |
| `--epochs` | 50 | |
| `--lr` | 1e-4 | |

W&B Bayes sweep configs are in `sbatch/sweep_*.yaml`.

## Dependencies

Install from project root:

```bash
pip install -r requirements.txt
```

GLoRIA has its own older environment (pytorch-lightning 1.1.4, torch 1.7.1):

```bash
conda env create -f src/gloria/environment.yml
pip install -e src/gloria/
```

Key packages: `torch`, `transformers`, `open_clip_torch`, `accelerate`, `bitsandbytes`, `xformers==0.0.28.post3`.

## Experiment Tracking

W&B is used for all runs. Results (checkpoints, embeddings) are saved to `results/`. Logs go to `logs/` (SLURM stdout/stderr). Both directories are `.gitignore`d.
