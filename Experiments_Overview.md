# Experiment-Übersicht

Zusammenfassung des experimentellen Vorgehens über alle Baselines und LACE
hinweg: Pretraining-Sweeps → Downstream-Sweeps → CV-Testing mit den jeweils
besten Hyperparametern. Alle Sweeps laufen über W&B Bayes-Sweeps
(`sbatch/run_sweep_pretrain.sh` / `sbatch/run_sweep_downstream.sh` als
generische SLURM-Wrapper, die per `SWEEP_ID` auf die jeweilige
`sweep_*.yaml`-Config zeigen).

## Genereller Ablauf pro Ansatz

1. **Pretraining-Sweep** (contrastive image-text pretraining, sofern
   zutreffend) — Ziel-Metrik i.d.R. `retrieval/mean_r1`.
2. **Downstream-Sweep** — Kopf/Klassifikator auf eingefrorenem Encoder,
   Ziel-Metrik `val/loss` bzw. bei LACE v2 `val/best_f1_macro`.
3. **CV-Testing** mit den besten Sweep-Hyperparametern fest verdrahtet in
   `sbatch/*/run_*_downstream_cv*.sh`, orchestriert über
   `src/downstream_cv.py` (10-fold, patientengruppiert, inkl. BTXRD als
   externem Test).
4. Jeweils **binary** (benign/malignant, gegen BTXRD) und **3-Klassen**
   (benign/intermediate/malignant) Varianten parallel durchgeführt.

## LoRA → Unfreeze

Anfänglich wurde die Bildencoder-Adaption per **LoRA** (Low-Rank Adaptation)
auf ausgewählten ViT-Layern umgesetzt (`sweep_pretrain_lora.yaml` bei
BiomedCLIP und GLoRIA). Später umgestellt auf **volles Unfreezing** der
letzten N Encoder-Blöcke (`--no_lora` / `--unfreeze_blocks` bzw.
`--adapter_mode unfreeze`), da dieser Ansatz in den Sweeps bessere Retrieval-
Ergebnisse lieferte. Bei LACE v2 ist inzwischen eine Kombination aus LoRA auf
frühen Layern (`--lora_layers 6 --lora_r 8`) und vollem Unfreeze der letzten
Blöcke (`--unfreeze_layers 4`) Standard.

## BiomedCLIP

- **Pretrain-Sweeps**: `src/biomedclip/train/sweep_pretrain_lora.yaml`
  (LoRA, `lora_layers ∈ {4,6,8}`, `lora_r ∈ {4,8,16}`) und
  `sweep_pretrain_unfreeze.yaml` (`unfreeze_blocks=4`, getrennte LR für
  Blocks/Projection). Zusätzlich Phrase-Varianten
  (`sweep_pretrain_lora_phrases.yaml`, `sweep_pretrain_unfreeze_phrases.yaml`).
- **Downstream-Sweeps**: `sweep_downstream.yaml` (binary, Kopf `mlp_no_meta`,
  Fokus auf `focal`/`cb_focal` Loss, Klassen-Gewichtung) sowie
  Img-Text-Downstream-Varianten (`sweep_img_text_downstream.yaml` für
  3-Klassen, `..._binary.yaml` für binary — hier bleibt BiomedCLIP eingefroren,
  `--freezed_biomedclip`).
- **CV**: `run_biomedclip_downstream_cv.sh`,
  `run_biomedclip_img_text_downstream_cv.sh`.

## GLoRIA

- **Pretrain-Sweeps**: erst `sweep_pretrain_lora.yaml` (LoRA auf CheXpert-
  ResNet50-Encoder), dann `sweep_pretrain_unfreeze_binary.yaml` /
  `sweep_pretrain_unfreeze_full.yaml` (`--adapter_mode unfreeze`,
  `n_layers=4`) — getrennt für binary/3-Klassen-Splits.
- **Downstream-Sweeps**: `sweep_downstream_frozen_binary.yaml` (frozen
  CheXpert-Checkpoint, binary) und `sweep_downstream_full_pretrained.yaml`
  (eigener pretrained Checkpoint, 3-Klassen).
- **CV**: `run_gloria_downstream_cv.sh` (3-Klassen) und
  `run_gloria_downstream_cv_binary.sh`. GLoRIA hat kein CV-Pretraining (kein
  Fold-Checkpoint) — derselbe fixe pretrainierte Checkpoint wird eingefroren
  über alle Folds verwendet.

## LACE (v1 → v2)

- **v1-Pretrain-Sweep** (`sweep_pretrain_v1.yaml`): LoRA (`lora_r`,
  `lora_layers`), Soft-Target-Temperaturen (`tau_s_*`), `t2i_mode`,
  `same_image_boost` — alle drei Losses (`L_ITA`, `L_sim`, `L_ortho`) von
  Anfang an aktiv.
- **v2-Pretrain-Sweeps** (`sweep_pretrain_v2.yaml`,
  `sweep_pretrain_v2_full.yaml`): Curriculum-basiert
  (`--loss_stages "dice:1,2" "ortho:1,2" "ita:2" "sim:2"`), Architektur fixiert
  auf LoRA(6 Layer, r=8) + Unfreeze(4 Layer), gesweept werden `lr`,
  `weight_decay`, Soft-Target-Temperaturen, `t2i_mode`/`lambda_t2i`,
  `same_image_boost`, `n_mask_heads`, `stage1_epochs`. Separater
  `..._full.yaml`-Sweep für den vollen (nicht CV-)Split mit erweitertem
  Suchraum um den bisher besten Lauf herum.
- **Downstream-Sweeps**: v1 (`sweep_downstream_v1.yaml`, binary,
  `val/loss`-Ziel) und v2 (`sweep_downstream_v2.yaml`, binary,
  `val/best_f1_macro`-Ziel, zusätzlich `downstream_visual_mode ∈
  {cls_fg, fg}` gesweept).
- **CV**: `run_lace_downstream_cv.sh` (binary, gegen BTXRD) und
  `run_lace_downstream_cv_full.sh` (3-Klassen, „full"). Im Gegensatz zu
  GLoRIA/ImageNet hat LACE ein **CV-Pretraining** — pro Fold ein eigener
  Checkpoint (`fold*/best_retrieval_checkpoint.pt`), sodass Encoder-CV und
  Downstream-CV zusammen orchestriert werden.

## Weitere Baselines (ImageNet, Scratch, Scratch+Text)

- **ImageNet**: kein Pretraining nötig (immer vanilla ViT-B/16, frozen),
  daher nur Downstream-Sweep (`src/imagenet_img/train/sweep.yaml`) und CV
  (`run_imagenet_img_downstream_cv.sh` / `..._cv_binary.sh`).
- **Scratch (img)** und **Scratch (img+text)**: End-to-End-Sweeps
  (`src/scratch_img/train/sweep.yaml`, `src/scratch_img_text/train/sweep.yaml`)
  über Encoder-Wahl (`resnet18`/`vit_tiny`), Encoder- und Kopf-LR getrennt,
  sowie bei img+text zusätzlich Text-Encoder-Architektur
  (`text_n_layers`, `text_hidden_dim`, `text_n_heads`) und Gewicht der
  kontrastiven Hilfs-Loss (`lambda_contrastive`).

## Splits: binary vs. 3-Klassen

Durchgängig zwei parallele Split-Familien:
- `split_binary_final.json` / `data/internal_dataset/cv_binary/` für binary
  Runs (benign/malignant, evaluiert zusätzlich gegen den externen
  BTXRD-Testsatz via `btxrd_downstream_binary.json`).
- `split_final.json` / `data/internal_dataset/cv/` für die 3-Klassen-Runs
  (benign/intermediate/malignant, kein BTXRD-Vergleich, da BTXRD nur binary
  gelabelt ist).
