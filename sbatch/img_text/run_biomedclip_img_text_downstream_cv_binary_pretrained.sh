#!/bin/bash
#SBATCH --job-name=biomedclip_img_text_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/img_text/slurm-%j_biomedclip_img_text_downstream_cv_binary_pretrained.out"

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export CONDA_EXE=$home_dir/miniconda3/bin/conda
source $home_dir/miniconda3/etc/profile.d/conda.sh
conda activate $MY_CONDA_ENV
echo Environment activated

export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export WANDB_DIR=$home_dir/Project/logs
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$PATH

# ── Fixed hyperparameters (fill in the winning sweep config) ──────────────────
# No --image_size here: biomedclip_img_text_downstream.py always uses open_clip's
# fixed 224×224 preprocessing (unlike biomedclip_downstream.py).
USE_MASK=true

# CV-mode continued-pretraining checkpoints (from run_biomedclip_pretrain.sh
# with --cv_dir), one fold<N>/{split.json,best_r1_checkpoint.pt} per fold.
# FROZEN=false tells the orchestrator to pick up each fold's own checkpoint
# next to its split file, instead of vanilla weights.
FROZEN=false
CV_DIR=$home_dir/Project/results/biomedclip_pretrain/run_bs128_unfreeze4_20260902_214812
PATTERN="fold*/split.json"
CHECKPOINT_FILENAME=best_r1_checkpoint.pt

# No BTXRD evaluation for this baseline — BTXRD samples carry no report text,
# so there is nothing for the text-encoder pathway to embed on that dataset.
BINARY=true

EPOCHS=200
PATIENCE=15
BATCH_SIZE=32
LR=7.752606669394482e-05
WEIGHT_DECAY=0.01373297164444574
DROPOUT=0.1633276669853014
HEAD=mlp_no_meta
HIDDEN_DIMS="[512, 256]"
META_EMBED_DIM=16

LOSS=focal
CLASS_WEIGHTING=effective
FOCAL_GAMMA=2.848604292771915
CB_BETA=0.99567647698647
SEED=42
EARLY_STOPPING_METRIC="val_bal_acc"
# ─────────────────────────────────────────────────────────────────────────────

# Note: --splits, --eval_test, --run_name and --checkpoint are managed per
# fold by the orchestrator (--checkpoint injected from CHECKPOINT_FILENAME
# when NOT --frozen).
python $home_dir/Project/src/downstream_cv.py \
    --baseline               biomedclip_img_text \
    $( [ "$FROZEN" = "true" ] && echo "--frozen" ) \
    $( [ "$FROZEN" != "true" ] && echo "--checkpoint_filename $CHECKPOINT_FILENAME" ) \
    --cv_dir                  $CV_DIR \
    --pattern                 "$PATTERN" \
    --out_dir                 $home_dir/Project/results \
    --epochs                  $EPOCHS \
    --patience                $PATIENCE \
    --batch_size               $BATCH_SIZE \
    --lr                      $LR \
    --weight_decay            $WEIGHT_DECAY \
    --dropout                 $DROPOUT \
    --head                    $HEAD \
    --hidden_dims              $HIDDEN_DIMS \
    --meta_embed_dim          $META_EMBED_DIM \
    --loss                    $LOSS \
    --focal_gamma             $FOCAL_GAMMA \
    --cb_beta                 $CB_BETA \
    --class_weighting         $CLASS_WEIGHTING \
    $( [ "$USE_MASK" = "true" ] && echo "--use_mask" ) \
    $( [ "$FROZEN" = "true" ] && echo "--freezed_biomedclip" ) \
    --binary                  $BINARY \
    --seed                    $SEED \
    --early_stopping_metric   $EARLY_STOPPING_METRIC \
    --wandb \
    --wandb_project biomedclip-img-text-downstream \
    --wandb_entity  philipp-wiese \
