#!/bin/bash
#SBATCH --job-name=chexfound_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/chexfound/slurm-%j_chexfound_downstream_cv_3class_pretrained.out"

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

# CV-mode continued-pretraining checkpoints (from run_chexfound_pretrain_cv.sh),
# one fold<N>/{split.json,checkpoint_best.pth} per fold. FROZEN=false tells the
# orchestrator to pick up each fold's own checkpoint next to its split file.
CV_DIR=$home_dir/Project/results/chexfound_pretrain_sweep/run_20260828_184339
PATTERN="fold*/split.json"
CHECKPOINT_FILENAME=checkpoint_best.pth
FROZEN=false

# 3-class ("full") CV — no BTXRD comparison (BTXRD is binary-labeled only).
BINARY=false

EPOCHS=50
PATIENCE=10
BATCH_SIZE=32
LR=0.0004718800200308377
WEIGHT_DECAY=0.1
DROPOUT=0.4
HEAD=mlp_no_meta
HIDDEN_DIMS="[64, 32]"
META_EMBED_DIM=16

LOSS=focal
CLASS_WEIGHTING=inverse
FOCAL_GAMMA=3.2187805897655695
CB_BETA=0.9
SEED=42
EARLY_STOPPING_METRIC="val_bal_acc"
# ─────────────────────────────────────────────────────────────────────────────

# Note: --splits, --eval_test and --run_name are managed per fold by the
# orchestrator; --checkpoint is reserved (must not be passed here) when NOT
# --frozen — the orchestrator injects it per fold from CHECKPOINT_FILENAME.
python $home_dir/Project/src/downstream_cv.py \
    --baseline               chexfound \
    $( [ "$FROZEN" = "true" ] && echo "--frozen" ) \
    --checkpoint_filename     $CHECKPOINT_FILENAME \
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
