#!/bin/bash
#SBATCH --job-name=gloria_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_gloria_downstream_cv.out"

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

# ── Fixed hyperparameters (winning sweep config) ───────────────────────────────
# CV-mode continued-pretraining checkpoints (from run_gloria_pretrain_cv.sh),
# one fold<N>/{split.json,best_retrieval_checkpoint.pt} per fold. FROZEN=false
# tells the orchestrator to pick up each fold's own checkpoint next to its
# split file, instead of a single fixed checkpoint used across every fold.
CV_DIR=$home_dir/Project/results/gloria_pretrain/gloria_pretrain_unfreeze4_20260830_145459
PATTERN="fold*/split.json"
CHECKPOINT_FILENAME=best_retrieval_checkpoint.pt
FROZEN=false

USE_PROJECTION=""   # leave unset for 2048-dim pre-projection features; set "1" for 768-dim post-projection
USE_MASK=true

BTXRD_MANIFEST=$home_dir/Project/data/BTXRD/btxrd_downstream_binary.json
BINARY=true

EPOCHS=100
PATIENCE=100
BATCH_SIZE=32
LR=7.993845727565584e-05
WEIGHT_DECAY=0.05
DROPOUT=0.2
HEAD=mlp_no_meta
HIDDEN_DIMS="[64]"
META_EMBED_DIM=16

LOSS=cb_focal
CLASS_WEIGHTING=sqrt
FOCAL_GAMMA=3.387585331220415
CB_BETA=0.9999
SEED=42
EARLY_STOPPING_METRIC="val_bal_acc"
# ─────────────────────────────────────────────────────────────────────────────

PROJ_FLAG=""
[ -n "$USE_PROJECTION" ] && PROJ_FLAG="--use_projection"

# Note: --splits, --eval_test and --run_name are managed per fold by the
# orchestrator; --checkpoint is reserved (must not be passed here) when NOT
# --frozen — the orchestrator injects it per fold from CHECKPOINT_FILENAME.
python $home_dir/Project/src/downstream_cv.py \
    --baseline               gloria \
    $( [ "$FROZEN" = "true" ] && echo "--frozen" ) \
    --checkpoint_filename     $CHECKPOINT_FILENAME \
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
    $PROJ_FLAG \
    $( [ "$USE_MASK" = "true" ] && echo "--use_mask" ) \
    --binary                  $BINARY \
    --seed                    $SEED \
    --early_stopping_metric   $EARLY_STOPPING_METRIC \
    --wandb \
    --wandb_project gloria-downstream \
    --wandb_entity  philipp-wiese \
    --btxrd_manifest         $BTXRD_MANIFEST \
