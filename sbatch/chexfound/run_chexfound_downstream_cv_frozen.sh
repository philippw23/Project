#!/bin/bash
#SBATCH --job-name=chexfound_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/chexfound/slurm-%j_chexfound_downstream_cv_3class_frozen.out"

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
IMAGE_SIZE=512   # 512 = native CheXFound | 224 = BiomedCLIP-equivalent resolution
USE_MASK=true

# Frozen original CheXFound weights (no continued pretraining) — no per-fold
# checkpoint to look up, so --cv_dir points straight at the raw fold split pool.
CHEXFOUND_WEIGHTS=$home_dir/Project/src/chexfound/data/teacher_checkpoint.pth
CV_DIR=$home_dir/Project/data/internal_dataset/cv
PATTERN="split_fold*.json"
FROZEN=true
# 3-class ("full") CV — no BTXRD comparison (BTXRD is binary-labeled only).
BINARY=false

EPOCHS=50
PATIENCE=10
BATCH_SIZE=32
LR=0.0002799678754176103
WEIGHT_DECAY=0.01
DROPOUT=0.5
HEAD=mlp_no_meta
HIDDEN_DIMS="[128]"
META_EMBED_DIM=16

LOSS=cb_focal
CLASS_WEIGHTING=sqrt
FOCAL_GAMMA=2.5860866438596446
CB_BETA=0.99
SEED=42
EARLY_STOPPING_METRIC="val_bal_acc"
# ─────────────────────────────────────────────────────────────────────────────

# Note: --splits, --eval_test and --run_name are managed per fold by the
# orchestrator; --checkpoint is passed explicitly below since --frozen leaves
# it unreserved (no per-fold checkpoint for the orchestrator to inject).
python $home_dir/Project/src/downstream_cv.py \
    --baseline               chexfound \
    $( [ "$FROZEN" = "true" ] && echo "--frozen" ) \
    --checkpoint              none \
    --chexfound_weights       $CHEXFOUND_WEIGHTS \
    --image_size              $IMAGE_SIZE \
    --cv_dir                  $CV_DIR \
    --pattern                 "$PATTERN" \
    --out_dir                 $home_dir/Project/results \
    --epochs                  $EPOCHS \
    --patience                $PATIENCE \
    --batch_size              $BATCH_SIZE \
    --lr                      $LR \
    --weight_decay            $WEIGHT_DECAY \
    --dropout                 $DROPOUT \
    --head                    $HEAD \
    --hidden_dims             $HIDDEN_DIMS \
    --meta_embed_dim          $META_EMBED_DIM \
    --loss                    $LOSS \
    --focal_gamma             $FOCAL_GAMMA \
    --cb_beta                 $CB_BETA \
    --class_weighting         $CLASS_WEIGHTING \
    $( [ "$USE_MASK" = "true" ] && echo "--use_mask" ) \
    --binary                  $BINARY \
    --seed                    $SEED \
    --early_stopping_metric   $EARLY_STOPPING_METRIC \
    --wandb \
    --wandb_project chexfound-downstream \
    --wandb_entity  philipp-wiese \
    --sweep \
