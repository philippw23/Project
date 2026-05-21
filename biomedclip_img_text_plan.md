# Plan: BiomedCLIP Image+Text Concatenation Downstream

## Goal

Add a new probing baseline that concatenates BiomedCLIP's post-projection image and text embeddings (512-dim each → 1024-dim) and feeds them into a linear or MLP probing head for 3-class malignancy classification.

Works with both vanilla BiomedCLIP weights (`--freezed_biomedclip`) and a further-pretrained checkpoint with LoRA (`--checkpoint`).

## Design Decisions (from interview)

| Decision | Choice |
|---|---|
| Text usage | At inference time (all splits have reports) |
| Feature space | Post-projection 512-dim (via `encode_image` / `encode_text`) |
| Normalization | L2-normalize each embedding before concatenation |
| Encoder | Always frozen (probing, not fine-tuning) |
| Projection heads | Included in the frozen encoders (no separate unfreezing) |
| Head variants | `linear`, `mlp`, `mlp_no_meta` — same as existing downstream |
| Entry point location | Top-level `src/biomedclip_img_text_downstream.py` (all logic inline) |

## Files to Change

### 1. `src/biomedclip/data/datasets.py`

Add `DownstreamDatasetWithText` class after `DownstreamDataset`. Returns image tensor, tokenized report (`context_length=256`), age, sex, and label for each sample. Text field: `befund_en + beurteilung_en` → fallback to `report` → fallback to `""`.

Also export from `src/biomedclip/data/__init__.py`.

### 2. `src/biomedclip_img_text_downstream.py` (new file)

Structure mirrors `biomedclip_downstream.py`. Key differences:

**Model loading**
- Load full `model` (not just `model.visual.trunk`)
- Support `--freezed_biomedclip` (vanilla weights) and `--checkpoint` (pretrained with LoRA)
- After loading: freeze all parameters (`requires_grad_(False)`)

**Embedding pre-computation** (new function `extract_img_text_embeddings`)
- Runs over a `DownstreamDatasetWithText` loader
- `img_emb = F.normalize(model.encode_image(images), dim=-1)` → (B, 512)
- `txt_emb = F.normalize(model.encode_text(texts), dim=-1)` → (B, 512)
- Concatenate → (B, 1024)
- Returns `(emb, age, sex, labels)` on CPU, stored in `EmbeddingDataset`

**Head** — `build_head(args, embed_dim=1024, device)` → reuses existing `LinearHead` / `MalignancyMLP`

**Training loop** — identical to the non-finetune path in `biomedclip_downstream.py`; all training is on pre-computed embeddings via `EmbeddingDataset`

**Key args**

| Arg | Default | Notes |
|---|---|---|
| `--checkpoint` | `results/.../best_r1_checkpoint.pt` | Pretrained BiomedCLIP |
| `--freezed_biomedclip` | off | Use vanilla BiomedCLIP, no checkpoint |
| `--splits` | `data/.../split.json` | Must include `report` field for all splits |
| `--head` | `mlp` | `linear`, `mlp`, `mlp_no_meta` |
| `--use_mask` | off | Mask-guided crop |
| `--epochs` | 50 | |
| `--patience` | 10 | |
| `--early_stopping_metric` | `val_loss` | `val_loss` or `val_bal_acc` |
| `--batch_size` | 64 | |
| `--lr` | 1e-3 | Head only |
| `--dropout` | 0.3 | MLP only |
| `--hidden_dims` | `[256, 128]` | MLP only |
| `--meta_embed_dim` | 16 | MLP+meta only |
| `--weight_decay` | 0.01 | |
| `--loss` | `ce` | Same choices as existing downstream |
| `--wandb_project` | `biomedclip-img-text-downstream` | |

**Output dir**: `results/biomedclip_img_text_downstream/<run_name>/`

## What is NOT changing

- `biomedclip_pretrain.py` — no changes needed; `inject_lora` in `lora.py` already unfreezes `model.visual.head`, `model.text.proj`, and `model.logit_scale` (lines 92–100 of `lora.py`)
- `biomedclip_downstream.py` — no changes
- `scratch_img_text/` — not touched (but not used for this baseline)
