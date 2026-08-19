# Pretraining Hyperparameter Sweeps

Reference notes for the thesis write-up on which pretraining hyperparameters were tuned via W&B Bayesian (or random) sweeps, per baseline. Configs live at `src/<approach>/train/sweep_pretrain*.yaml`; sweep search space is the `parameters:` block, everything else is pinned in the `command:` block.

---

## BiomedCLIP

Four sweep variants — LoRA vs. full-block unfreeze, each with/without phrase-level text — all Bayes-optimized, maximizing `retrieval/mean_r1`.

| Variant | File | Tuned parameters |
|---|---|---|
| LoRA | [sweep_pretrain_lora.yaml](src/biomedclip/train/sweep_pretrain_lora.yaml) | `lora_layers` {4,6,8}, `lora_r` {4,8,16}, `lr_lora` (1e-6–1e-4, log), `lr_proj` (5e-5–5e-4, log), `weight_decay` (0.01–0.3, log) |
| LoRA + phrases | [sweep_pretrain_lora_phrases.yaml](src/biomedclip/train/sweep_pretrain_lora_phrases.yaml) | same as above, `lora_layers` extended to {2,4,6,8} |
| Unfreeze | [sweep_pretrain_unfreeze.yaml](src/biomedclip/train/sweep_pretrain_unfreeze.yaml) | `lr_proj` (5e-5–5e-4, log), `lr_blocks` (1e-6–1e-4, log), `weight_decay` (0.01–0.3, log); `unfreeze_blocks` fixed at 4 |
| Unfreeze + phrases | [sweep_pretrain_unfreeze_phrases.yaml](src/biomedclip/train/sweep_pretrain_unfreeze_phrases.yaml) | same as above, `unfreeze_blocks` also tuned {2,4,6,8} |

`batch_size=128` fixed in all four; 55 epochs fixed.

---

## GLoRIA

Three sweep variants (LoRA / full-unfreeze / full-unfreeze-binary), all Bayes, maximizing `mean_r1`.

| Variant | File | Tuned parameters |
|---|---|---|
| LoRA | [sweep_pretrain_lora.yaml](src/gloria/train/sweep_pretrain_lora.yaml) | `lr` (5e-6–5e-4, log), `weight_decay` (0.001–0.1, log), `n_layers` {0,1,2,3}, `lora_r` {4,8,16}, `temp1` {4,5,6}, `temp2` {4,5,6}, `temp3` {5,10,15}, `local_loss_weight` {0.5,1.0,1.5}, `global_loss_weight` {0.5,1.0,1.5} |
| Unfreeze (full/3-class split) | [sweep_pretrain_unfreeze_full.yaml](src/gloria/train/sweep_pretrain_unfreeze_full.yaml) | same as LoRA minus `n_layers`/`lora_r`; `--adapter_mode unfreeze --n_layers 4` fixed instead |
| Unfreeze (binary split) | [sweep_pretrain_unfreeze_binary.yaml](src/gloria/train/sweep_pretrain_unfreeze_binary.yaml) | identical search space to the unfreeze-full variant; only the split file differs (`split_binary_final.json`) |

`batch_size=32`, 50 epochs, 5 warmup epochs, patience=15 fixed in all three.

---

## CheXFound

One sweep, [sweep_pretrain.yaml](src/chexfound/train/sweep_pretrain.yaml), Bayes, minimizing `train/loss_total`.

| Tuned | Range |
|---|---|
| `base_lr` | 1e-6–2e-4 (log) |
| `momentum_teacher` | 0.990–0.9995 (uniform) |

`head_mode` is pinned to `"train"` (not swept). `lora_r`/`lora_layers` are present in the file but **commented out** — not currently tuned.

---

## LACE

Three sweep variants, all maximizing `retrieval/mean_r1`.

### v1 — [sweep_pretrain_v1.yaml](src/LACE/train/sweep_pretrain_v1.yaml) (Bayes)

| Tuned | Range |
|---|---|
| `lr` | 5e-6–2e-4 (log) |
| `lora_r` | {4, 8, 16} |
| `lora_layers` | {2, 4, 6} |
| `weight_decay` | 0.001–0.1 (log) |
| `tau_s_beur` | {0.02, 0.04, 0.07} |
| `tau_s_bef` | {0.02, 0.04, 0.07} |
| `tau2` | {0.03, 0.04, 0.07, 0.1} |
| `tau_s_img_full` | {0.03, 0.04, 0.07} |
| `t2i_mode` | {text_text, infonce} |
| `same_image_boost` | {0.0, 10.0, 20.0} |

`lambda_ita`/`lambda_sim`/`lambda_ortho` fixed at 1.0; `context_fraction`/`global_context_fraction` fixed at 0.15.

### v2 — [sweep_pretrain_v2.yaml](src/LACE/train/sweep_pretrain_v2.yaml) (Bayes)

| Tuned | Range |
|---|---|
| `lr` | 1e-5–5e-4 (log) |
| `weight_decay` | 7e-4–2e-3 (log) |
| `tau_s_beur` | 0.030–0.040 (uniform) |
| `tau_s_bef` | 0.035–0.045 (uniform) |
| `tau_s_img_full` | 0.025–0.035 (uniform) |
| `same_image_boost` | {5.0, 10.0, 15.0, 20.0} |
| `n_mask_heads` | {16, 24, 32} |
| `stage1_epochs` | {0, 16, 18, 20} |

`t2i_mode` fixed="text_text", `lambda_t2i` fixed=0.5. Architecture (`lora_layers=6, lora_r=8, lora_alpha=32, unfreeze_layers=4`), loss-stage curriculum (`dice:1,2 ortho:1,2 ita:2 sim:2`), and remaining optimization settings (`scheduler=cosine`, 120 epochs, patience=20) are pinned in the `command` block, not swept.

### v2 full — [sweep_pretrain_v2_full.yaml](src/LACE/train/sweep_pretrain_v2_full.yaml) (**random** search, not Bayes)

Narrower search space, centered around the best full-split run found so far:

| Tuned | Range |
|---|---|
| `lr` | 4e-5–1.5e-4 (log) |
| `weight_decay` | 3e-4–3e-3 (log) |
| `stage1_epochs` | {16, 18, 20} |
| `tau_s_beur` | {0.030, 0.0325, 0.035} |
| `tau_s_bef` | {0.036, 0.038, 0.040} |
| `tau_s_img_full` | {0.024, 0.028, 0.032} |
| `t2i_mode` | {image_image, text_text} |
| `lambda_t2i` | {0.25, 0.5, 0.75, 1.0} |
| `same_image_boost` | {5.0, 10.0, 15.0} |
| `n_mask_heads` | {8, 16, 24} |

Same fixed architecture/curriculum as v2 above; `patience` fixed=20, 160 epochs, run on the full 3-class split (`split_final.json`) rather than the binary split used by the base v2 sweep.
