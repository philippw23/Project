# Plan: `imagenet_img` Baseline

## Goal
Add an `imagenet_img` baseline alongside the existing `scratch_img`. Uses a frozen ViT-B/16 with ImageNet pretrained weights (linear probing), matching BiomedCLIP's encoder architecture exactly.

## Design Decisions
| Decision | Choice | Rationale |
|---|---|---|
| Architecture | ViT-B/16 (`vit_base_patch16_224`, pretrained=True) | Matches BiomedCLIP backbone exactly; isolates pretraining domain as the only variable |
| Encoder | Frozen throughout | Linear probing; consistent with all other baselines; `scratch_img` kept for ablation |
| Output dim | 768 | Same as BiomedCLIP; MalignancyMLP handles dynamically |
| Val/test embeddings | Pre-computed once | Encoder is frozen + deterministic transforms; reuses `extract_embeddings` + `EmbeddingDataset` from biomedclip |
| Train embeddings | Computed each batch | Random augmentations require fresh encoder pass per sample |
| `scratch_img` | Kept unchanged | Gap between scratch↔imagenet quantifies ImageNet pretraining value |

## Files to Create

### 1. `src/imagenet_img/__init__.py` — empty package marker

### 2. `src/imagenet_img/models/__init__.py` — empty

### 3. `src/imagenet_img/models/encoders.py`
- Single function `build_encoder() -> tuple[nn.Module, int]`
- `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)` → 768-dim
- No `--encoder` arg needed (only one architecture)

### 4. `src/imagenet_img/data/__init__.py` — empty

### 5. `src/imagenet_img/data/transforms.py`
- Identical to `scratch_img/data/transforms.py` (ImageNet normalization stats, same augmentation)

### 6. `src/imagenet_img/train/__init__.py` — empty

### 7. `src/imagenet_img/train/downstream.py`
Key changes from `scratch_img/train/downstream.py`:

**Removed:**
- `--encoder` arg (fixed to ViT-B/16)
- `--lr_encoder` arg
- `--warmup_epochs` arg
- Encoder param group in optimizer
- LambdaLR scheduler (encoder group only; MLP uses flat LR anyway)

**Changed:**
- `build_encoder()` imported from `imagenet_img.models.encoders`
- `encoder` frozen immediately after construction: `encoder.eval(); encoder.requires_grad_(False)`
- `train_one_epoch`: encoder stays in eval + `torch.no_grad()` for encoder forward
- `evaluate()`: replaced by `evaluate_cached()` that operates on pre-computed `EmbeddingDataset`
- After building loaders, pre-compute val and test embeddings:
  ```python
  val_emb, val_age, val_sex, val_lbl   = extract_embeddings(encoder, val_loader, device)
  test_emb, test_age, test_sex, test_lbl = extract_embeddings(encoder, test_loader, device)
  val_emb_loader  = DataLoader(EmbeddingDataset(...), ...)
  test_emb_loader = DataLoader(EmbeddingDataset(...), ...)
  ```
- Optimizer: `torch.optim.AdamW(mlp.parameters(), lr=args.lr_mlp, ...)`
- Scheduler: single `LambdaLR` with flat `lambda epoch: 1.0`
- W&B project: `imagenet-img-downstream`
- out_dir default: `results/imagenet_img`
- run_tag: `{ts}_{args.head}_{args.loss}` (no encoder name)

### 8. `src/imagenet_img/train/sweep.yaml`
- Copied from `scratch_img/train/sweep.yaml`
- Removed: `encoder`, `lr_encoder`, `warmup_epochs`
- Updated: `program`, `project`, `--out_dir`, `--wandb_project`
- `run_cap`: 40

### 9. `src/imagenet_img_downstream.py`
```python
from imagenet_img.train.downstream import main, parse_args
if __name__ == "__main__":
    main(parse_args())
```

### 10. `sbatch/imagenet_img/run_imagenet_img.sh`
- Copied from `sbatch/scratch/run_scratch_img.sh`
- Removed: `ENCODER`, `LR_ENCODER`, `WARMUP_EPOCHS` variables and corresponding args
- Updated: job-name, log file path, out_dir, wandb_project, entry point script
