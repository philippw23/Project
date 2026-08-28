# Downstream Hyperparameter Sweeps

Reference notes for the thesis write-up on which downstream (malignancy classifier head) hyperparameters were tuned via W&B Bayesian sweeps, per baseline. Configs live at `src/<approach>/train/sweep_downstream*.yaml`; sweep search space is the `parameters:` block, everything else (checkpoint, split file, `--use_mask`, `--eval_test`, epochs, seed) is pinned in the `command:` block.

---

## BiomedCLIP

One sweep, [sweep_downstream.yaml](src/biomedclip/train/sweep_downstream.yaml), Bayes, minimizing `val/loss`.

| Tuned | Range |
|---|---|
| `lr` | 3e-5–5e-4 (log) |
| `dropout` | {0.2, 0.3, 0.4, 0.5} |
| `weight_decay` | {0.01, 0.05, 0.1} |
| `hidden_dims` | {[32], [64], [128], [64,32]} |
| `batch_size` | {16, 32, 64} |
| `loss` | {focal, cb_focal} |
| `class_weighting` | {sqrt, inverse} (ignored by `cb_focal`, which uses `cb_beta` instead) |
| `focal_gamma` | {2.0, 3.0, 4.0} (active for focal and cb_focal) |
| `cb_beta` | {0.99, 0.999} (active for cb_focal only) |

`head` fixed="mlp_no_meta". Run against the binary split (`split_binary_final.json`, `--binary true`) from a fixed pretrain checkpoint (`run_bs128_unfreeze4_20260727_234731`); `--use_mask` and `--eval_test` on; 50 epochs.

---

## GLoRIA

Two sweep variants (frozen ResNet50 backbone / full-unfreeze pretrained backbone), both binary, both Bayes, minimizing `val/loss`. Identical search space.

| Variant | File | Checkpoint |
|---|---|---|
| Frozen | [sweep_downstream_frozen_binary.yaml](src/gloria/train/sweep_downstream_frozen_binary.yaml) | `src/gloria/pretrained/chexpert_resnet50.ckpt` (original CheXpert weights, not domain-adapted) |
| Full-pretrained | [sweep_downstream_full_pretrained.yaml](src/gloria/train/sweep_downstream_full_pretrained.yaml) | `results/gloria_pretrain/gloria_pretrain_unfreeze4_20260812_215155/best_retrieval_checkpoint.pt` |

| Tuned | Range |
|---|---|
| `lr` | 1e-6–1e-4 (frozen) / 1e-5–1e-3 (full-pretrained), log |
| `dropout` | {0.2, 0.3, 0.4, 0.5} |
| `weight_decay` | {0.001, 0.01, 0.05, 0.1} |
| `hidden_dims` | {[64],[32],[128],[256],[128,64],[256,128]} |
| `batch_size` | {16, 32, 64} |
| `loss` | {focal, cb_focal} |
| `class_weighting` | {sqrt, inverse, effective} (ignored by `cb_focal`) |
| `focal_gamma` | 2.5–3.5 (uniform) |
| `cb_beta` | {0.99, 0.999, 0.9999} (cb_focal only) |

`head` fixed="mlp_no_meta"; both run on `split_binary_final.json` with `--binary true`, `--use_mask`, `--eval_test`, 100 epochs, 60 run cap.

---

## CheXFound

Three sweep variants, all Bayes, all from the same frozen `checkpoint_best.pth` sweep-selected iBOT teacher (`results/chexfound_pretrain_sweep/2gx5kqho/checkpoint_best.pth`, except the "frozen" variant below which uses the raw teacher checkpoint).

### 3-class — [sweep_downstream.yaml](src/chexfound/train/sweep_downstream.yaml)

Narrow, hand-centered search space; minimizes `val/loss`.

| Tuned | Range |
|---|---|
| `lr` | 2.5e-5–8e-5 (log) |
| `dropout` | {0.2, 0.3} |
| `hidden_dims` | {[64],[64,32],[128,64]} |
| `batch_size` | {32, 64} |
| `focal_gamma` | 3.3–3.8 (uniform) |

`loss` fixed="focal", `class_weighting` fixed="inverse", `weight_decay` fixed=0.1. Full 3-class split (`split_final.json`, `--binary false`).

### Binary — [sweep_downstream_binary.yaml](src/chexfound/train/sweep_downstream_binary.yaml)

Wider space than the 3-class variant; maximizes `val/balanced_acc` instead of minimizing loss.

| Tuned | Range |
|---|---|
| `class_weighting` | {inverse, sqrt} |
| `weight_decay` | {0.05, 0.1} |
| `hidden_dims` | {[64],[64,32],[128],[128,64]} |
| `dropout` | {0.2, 0.3, 0.4} |
| `lr` | 2.5e-5–3e-4 (log) |
| `focal_gamma` | 2.5–3.5 (uniform) |

`loss` fixed="focal", `batch_size` fixed=64. Binary split (`split_binary_final.json`) plus `--btxrd_manifest` for external BTXRD scoring.

### Frozen backbone — [sweep_downstream_frozen.yaml](src/chexfound/train/sweep_downstream_frozen.yaml)

Same search space as BiomedCLIP's sweep, minimizing `val/loss`.

| Tuned | Range |
|---|---|
| `lr` | 3e-5–5e-4 (log) |
| `dropout` | {0.2, 0.3, 0.4, 0.5} |
| `weight_decay` | {0.01, 0.05, 0.1} |
| `hidden_dims` | {[32],[64],[128],[64,32]} |
| `batch_size` | {16, 32, 64} |
| `loss` | {focal, cb_focal} |
| `class_weighting` | {sqrt, inverse} (ignored by cb_focal) |
| `focal_gamma` | {2.0, 3.0, 4.0} |
| `cb_beta` | {0.99, 0.999} (cb_focal only) |

Uses the raw ("frozen"/never fine-tuned) `teacher_checkpoint.pth` directly rather than a domain-pretrained checkpoint (`--checkpoint none`); `binary` fixed=false, full split. 30 run cap (vs. 60 for BiomedCLIP's equivalent sweep).

All three: `head` fixed="mlp_no_meta", `--use_mask` and `--eval_test` on, 50 epochs.

---

## LACE

Two sweep variants, one per architecture iteration, both from fixed pretrain checkpoints, both `head` fixed="mlp_no_meta", `--use_mask` on.

### v1 — [sweep_downstream_v1.yaml](src/LACE/train/sweep_downstream_v1.yaml) (Bayes, run_cap 30)

Minimizes `val/loss`. Checkpoint `results/lace_pretrain/run_20260613_202920/best_retrieval_checkpoint.pt`, binary split (`split_binary.json`), 100 epochs.

| Tuned | Range |
|---|---|
| `lr` | 1e-5–1e-3 (log) |
| `dropout` | {0.0, 0.1, 0.2, 0.3} |
| `weight_decay` | {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5} |
| `hidden_dims` | {[128,64],[128],[256,128],[512,256],[256]} |
| `batch_size` | {16, 32, 64} |
| `focal_gamma` | 2.5–3.25 (uniform) |

`loss` fixed="focal", `class_weighting` fixed="effective".

### v2 — [sweep_downstream_v2.yaml](src/LACE/train/sweep_downstream_v2.yaml) (Bayes, run_cap 50)

Maximizes `val/best_f1_macro` (best checkpoint's F1, not final-epoch loss). Checkpoint `results/lace_v2_pretrain/run_20260814_175443/best_retrieval_checkpoint.pt`, full 3-class split (`split_final.json`) with `--eval_test`, 200 epochs, patience 10.

| Tuned | Range |
|---|---|
| `downstream_visual_mode` | {cls, fg, cls_fg} |
| `batch_size` | {32, 64} |
| `hidden_dims` | {[128,64],[256,128],[512,256]} |
| `dropout` | 0.15–0.275 (uniform) |
| `lr` | 3e-6–5e-5 (log) |
| `weight_decay` | 0.01–0.30 (log) |
| `focal_gamma` | 2.7–3.2 (uniform) |
| `class_weighting` | {none, sqrt} |

`loss` fixed=["focal"], `image_size` fixed=224. Notably tunes `downstream_visual_mode` (choice of visual pooling for the head), which none of the other approaches' downstream sweeps expose.

#### `downstream_visual_mode` logic (`LACEv2Classifier._get_visual`, [downstream.py:165-182](src/LACE/models/downstream.py#L165-L182))

The frozen backbone (ViT + mask decoder + mask head, all `requires_grad_(False)` — only `age_emb`/`sex_emb`/the MLP head are trained) exposes three ways to turn a crop into the visual feature vector fed to the head:

| Mode | Feature | Dim | Construction |
|---|---|---|---|
| `cls` | Global image embedding | 512 | `vit.img_proj(cls_raw)` — the CLS token, projected through the same head used for $\mathcal{L}_\text{ITA}$'s image embedding during pretraining. |
| `fg` | Lesion-localized embedding | 512 | Heatmap-weighted average of projected patch tokens (see below). |
| `cls_fg` (default) | Both concatenated | 1024 | `cat([cls_repr, fg_repr])`. |

**How `fg` is built:** at inference time the frozen mask decoder + `MaskPredictionHead` produce `mask_logits` for every mask token; since there's no ground truth here, the foreground token is picked via `select_fg_token` — the token with the highest peak-to-mean activation ratio (most spatially concentrated, i.e. most lesion-like rather than uniformly firing over background). That token's 196 patch logits are softmax'd with temperature `sim_attn_tau` into attention weights `w`, which then weight-average the projected patch embeddings (`vit.patch_proj`) into a single 512-d vector — the same mechanism $\mathcal{L}_\text{sim}$'s I2T context vector uses during pretraining ([ARCHITECTURE.md §3.2](src/LACE/ARCHITECTURE.md)), just without the GT-mask restriction (there's no GT mask downstream).

So `cls` gives the classifier CLIP-style global context, `fg` gives it a representation pooled specifically over the (predicted) lesion region, and `cls_fg` lets the MLP head see both and learn how to weigh them — which is why it's the default and why the sweep still searches over the other two rather than assuming `cls_fg` always wins.
