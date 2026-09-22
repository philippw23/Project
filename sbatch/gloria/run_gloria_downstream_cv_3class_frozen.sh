#!/bin/bash
#SBATCH --job-name=gloria_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/gloria/slurm-%j_gloria_downstream_cv_3class_frozen.out"

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
# GLoRIA has no CV-mode pretraining (no per-fold checkpoints) — the same fixed
# pretrained checkpoint is used, frozen, across every fold. --cv_dir therefore
# points straight at the raw fold split pool, not a pretrain run's fold<N>/ dirs.
CHECKPOINT="$home_dir/Project/src/gloria/pretrained/chexpert_resnet50.ckpt"
USE_PROJECTION=""   # leave unset for 2048-dim pre-projection features; set "1" for 768-dim post-projection
USE_MASK=true

CV_DIR=$home_dir/Project/data/internal_dataset/cv
PATTERN="split_fold*.json"

BINARY=false

EPOCHS=50
PATIENCE=100
BATCH_SIZE=64
LR=0.00021850570810341867
WEIGHT_DECAY=0.05
DROPOUT=0.2
HEAD=mlp_no_meta
HIDDEN_DIMS="[64]"
META_EMBED_DIM=16

LOSS=focal
CLASS_WEIGHTING=effective
FOCAL_GAMMA=3.3840067487188277
CB_BETA=0.9999
SEED=42
EARLY_STOPPING_METRIC="val_bal_acc"
# ─────────────────────────────────────────────────────────────────────────────

PROJ_FLAG=""
[ -n "$USE_PROJECTION" ] && PROJ_FLAG="--use_projection"

# Note: --splits, --eval_test and --run_name are managed per fold by the
# orchestrator; --checkpoint stays reserved only when NOT --frozen, so it is
# passed through here fixed for every fold.
python $home_dir/Project/src/downstream_cv.py \
    --baseline               gloria \
    --frozen \
    --checkpoint              $CHECKPOINT \
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
