# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bone-tumor malignancy classification research project using domain-adapted vision-language models. The pipeline adapts BiomedCLIP (ViT-B/16 + PubMedBERT) to a private internal bone-tumor X-ray dataset via contrastive pretraining, then trains a downstream malignancy classifier (3-class: benign/intermediate/malignant, with a `--binary` benign/malignant mode used against the external BTXRD test set).

Baselines implemented: **BiomedCLIP** contrastive pretraining (optionally with a GLoRIA-style local loss added, `biomedclip_gloria`), **CheXFound** (DINO+iBOT), **GLoRIA**, and an **ImageNet-pretrained** frozen linear probe — alongside **LACE** (Lesion-Aligned Contrastive Embedding), the novel curriculum-learning approach developed in this project. LACE has two iterations: v1 (all losses active from epoch 1) and v2 (configurable 2-stage curriculum with a mask decoder and prototype/evidential losses); see [src/LACE/ARCHITECTURE.md](src/LACE/ARCHITECTURE.md).

## Running Experiments

All training is executed via SLURM on an HPC cluster. `sbatch/` mirrors `src/`, with one subdirectory per approach:

```bash
# BiomedCLIP contrastive pretraining / downstream / eval
sbatch sbatch/biomedclip/run_biomedclip_pretrain.sh
sbatch sbatch/biomedclip/run_biomedclip_downstream.sh
sbatch sbatch/biomedclip/run_biomedclip_downstream_eval.sh   # score a saved head, no training

# LACE curriculum pretraining (v1 or v2) / downstream / CV / eval
sbatch sbatch/lace/run_lace_pretrain_v2.sh
sbatch sbatch/lace/run_lace_downstream_v2.sh
sbatch sbatch/lace/run_lace_downstream_cv.sh                 # 10-fold CV incl. BTXRD test
sbatch sbatch/lace/run_lace_downstream_eval.sh

# CheXFound iBOT continued pretraining / downstream / eval
sbatch sbatch/chexfound/run_chexfound_pretrain.sh
sbatch sbatch/chexfound/run_chexfound_downstream.sh
sbatch sbatch/chexfound/run_chexfound_downstream_eval.sh

# GLoRIA pretraining / downstream / eval
sbatch sbatch/gloria/run_gloria_pretrain.sh
sbatch sbatch/gloria/run_gloria_downstream.sh
sbatch sbatch/gloria/run_gloria_downstream_eval.sh

# Additional baseline (image-only frozen ImageNet probe)
sbatch sbatch/imagenet_img/run_imagenet_img.sh

# BiomedCLIP + GLoRIA-style local loss pretraining variant
sbatch sbatch/biomedclip_gloria/run_biomedclip_gloria_pretrain.sh

# LLM phrase extraction (Qwen2.5-7B)
sbatch sbatch/data/run_llm_extractor.sh

# W&B hyperparameter sweeps
sbatch sbatch/run_sweep_pretrain.sh
sbatch sbatch/run_sweep_downstream.sh
```

Every baseline follows the same `<name>_downstream.py` (trains a head, saves a checkpoint) / `<name>_downstream_eval.py` (loads a saved head checkpoint, re-runs evaluation only — no training, no hyperparameter flags) split. Python entry points can also be run directly, e.g.:

```bash
python src/biomedclip_pretrain.py [args]
python src/LACE/train/pretrain_v2.py [args]
python src/downstream_cv.py --baseline lace --version v2 --checkpoint <ckpt> --cv_dir data/internal_dataset/cv [args]
python src/gloria/run.py [config_path]
```

**There is no test suite.** Validation is done via W&B logging during training and manual inspection scripts (`src/check_dataset.py`, `src/visualize_samples.py`, `src/analyze_descriptor_vectors.py`).

## Data Pipeline

Four-stage pipeline before training:

1. **Image preprocessing** — mask-guided cropping: `src/preprocess_images.py`
2. **Report translation** — German → English: `src/translate_reports.py`
3. **Phrase extraction** — LLM (Qwen2.5-7B): `src/llm_extractor.py` (or `src/llm_extractor_seperated.py` for befund/beurteilung split separately)
4. **Dataset assembly** — `src/create_dataset.py`

Downstream of that: `src/create_split.py` builds the stratified train/val/test manifest, and `src/create_cv_splits.py` derives a patient-grouped 10-fold CV pool from it (kept in `data/internal_dataset/cv/`) for `downstream_cv.py`-style fold evaluation. `src/build_btxrd_downstream.py` converts the external **BTXRD** dataset into the same sample-dict manifest format, used as a frozen external test set (binary benign/malignant only).

Default data paths (hardcoded in [src/biomedclip/utils/misc.py](src/biomedclip/utils/misc.py)):

```
data/internal_dataset/images/              # bone tumor X-rays
data/internal_dataset/segmentations/       # lesion masks
data/internal_dataset/metadata.xlsx        # age, sex, labels
data/internal_dataset/text/full_reports.json
data/internal_dataset/dataset_full.json    # assembled dataset JSON
data/internal_dataset/split.json           # default split manifest (split_binary.json for binary runs)
data/internal_dataset/cv/                  # 10-fold CV split pool
data/BTXRD/                                # external test set (images, annotations, downstream manifest)
results/                                   # checkpoints, outputs
```

Note: several dated/backup variants of the dataset JSON and split/report files live alongside the canonical ones above (e.g. `dataset_full_14B_*.json`, `split_final.json`, `full_reports_new.json`) — treat those as scratch snapshots, not defaults.

## Architecture

### BiomedCLIP (main baseline)

- **Base model**: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **LoRA injection**: [src/biomedclip/models/lora.py](src/biomedclip/models/lora.py) — `LoRALinear` wraps selected ViT Q/V projection layers; text encoder is frozen
- **Contrastive loss**: symmetric InfoNCE in [src/biomedclip/loss/contrastive.py](src/biomedclip/loss/contrastive.py)
- **Downstream**: `MalignancyMLP` in [src/biomedclip/models/classifier.py](src/biomedclip/models/classifier.py) — late fusion of image embedding + age/sex metadata
- **Data**: `BoneTumorPairDataset` (image-text pairs) and `DownstreamDataset` in [src/biomedclip/data/datasets.py](src/biomedclip/data/datasets.py)
- **Splits**: stratified train/val/test via [src/biomedclip/data/splits.py](src/biomedclip/data/splits.py)

### LACE (curriculum learning)

- **v1** — [src/LACE/train/pretrain.py](src/LACE/train/pretrain.py): all three losses active from epoch 1, fixed or learned log-scale weights.
  1. **L_ITA** — global image-text alignment with soft phrase-phrase targets
  2. **L_sim** — local lesion-phrase alignment via attention pooling over a tight, lesion-centred crop
  3. **L_ortho** — orthogonality regularization on the BTXRD dataset
- **v2** — [src/LACE/train/pretrain_v2.py](src/LACE/train/pretrain_v2.py): adds a `MaskTokenDecoder` (lesion segmentation, `L_dice`), with a fully configurable 2-stage curriculum (`--loss_stages`, e.g. `ita:2 sim:none`) — by default stage 1 warms up the mask decoder (`dice + ortho`), stage 2 adds everything.

Encoders in [src/LACE/models/encoders.py](src/LACE/models/encoders.py): `SharedViT`, `BiomedCLIPTextEncoder`, `ProjectionHead`. Downstream heads/training in [src/LACE/train/downstream.py](src/LACE/train/downstream.py) and [src/LACE/models/downstream.py](src/LACE/models/downstream.py); k-fold CV orchestration (internal folds + frozen BTXRD test) in `src/downstream_cv.py`. Design docs: [src/LACE/ARCHITECTURE.md](src/LACE/ARCHITECTURE.md), [src/LACE/CV_EVAL_PLAN.md](src/LACE/CV_EVAL_PLAN.md).

### CheXFound

Continued pretraining with DINO + iBOT joint objective ([src/chexfound/train/ssl_meta_arch.py](src/chexfound/train/ssl_meta_arch.py)). Uses block-based rectangular masking and KoLeo regularization. Config-driven via YAML files in [src/chexfound/configs/](src/chexfound/configs/).

### Additional baselines

- **ImageNet** ([src/imagenet_img/](src/imagenet_img/)): frozen ImageNet-1k ViT-B/16, linear probe only (`src/imagenet_img_downstream.py`).
- **BiomedCLIP + GLoRIA** ([src/biomedclip_gloria/](src/biomedclip_gloria/)): BiomedCLIP contrastive pretraining with an added GLoRIA-style local (word/region) loss (`src/biomedclip_gloria_pretrain.py`); downstream classification reuses `biomedclip_downstream.py`/`biomedclip_downstream_eval.py` pointed at the resulting checkpoint.

### Downstream classifiers & eval scripts

There is no single unified downstream entry point; each approach has its own `<name>_downstream.py` (trains the head, freezes the backbone, pre-computes image embeddings once and reuses them across epochs) and matching `<name>_downstream_eval.py` (loads a saved head checkpoint — including stored config/args, normalization stats, and any LoRA deltas — and re-scores it against a split or the BTXRD manifest, no training). Shared eval helpers live in [src/biomedclip/utils/downstream_eval.py](src/biomedclip/utils/downstream_eval.py).

### Qwen LLM Extractor

[src/qwen_llm_extractor/](src/qwen_llm_extractor/) uses Qwen2.5-7B-Instruct (4-bit quantized via bitsandbytes) to extract anatomical phrases from German radiology reports. Two modes: joint (single prompt, `src/llm_extractor.py`) and separated (befund + beurteilung separately, `src/llm_extractor_seperated.py`). Prompts in [src/qwen_llm_extractor/prompts/](src/qwen_llm_extractor/prompts/).

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

W&B Bayes sweep configs live next to each approach's training code, e.g. `src/LACE/train/sweep_pretrain_v2.yaml`, `src/imagenet_img/train/sweep.yaml`, as well as the top-level `sbatch/run_sweep_*.sh` wrappers.

## Dependencies

There is no top-level `requirements.txt` in the repo currently — install the key packages manually into your environment: `torch`, `transformers`, `open_clip_torch`, `accelerate`, `bitsandbytes`, `xformers==0.0.28.post3`.

GLoRIA has its own older, self-contained environment (pytorch-lightning 1.1.4, torch 1.7.1):

```bash
conda env create -f src/gloria/environment.yml
pip install -e src/gloria/
```

## Experiment Tracking

W&B is used for all runs. Results (checkpoints, embeddings) are saved to `results/`. Logs go to `logs/` (SLURM stdout/stderr). Both directories are `.gitignore`d.
