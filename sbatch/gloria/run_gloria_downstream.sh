#!/bin/bash
#SBATCH --job-name=gloria_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
CHECKPOINT="/mnt/nfs/homedirs/philippw/Project/results/gloria_pretrain/gloria_pretrain_lora2_r8_20260610_133210/best_retrieval_checkpoint.pt"
# /mnt/nfs/homedirs/philippw/Project/src/gloria/pretrained/chexpert_resnet50.ckpt
SPLITS="/mnt/nfs/homedirs/philippw/Project/data/internal_dataset/split.json"

BATCH_SIZE=16
LR=1e-3
DROPOUT=0.3
WEIGHT_DECAY=0.01
EPOCHS=50

# Head mode: mlp (age+sex fusion), mlp_no_meta (no metadata), linear (linear probe)
HEAD="mlp_no_meta"
HIDDEN_DIMS="256 128"
META_EMBED_DIM=16

# Loss: ce, wce, ce_smooth, focal, cb_focal, ldam, balanced_softmax
LOSS="focal"
FOCAL_GAMMA=2.5
# Class weighting: none, inverse, sqrt, effective
CLASS_WEIGHTING="sqrt"

# Embedding: leave unset to use 2048-dim pre-projection features (default),
# or set USE_PROJECTION=1 to use 768-dim post-projection features.
USE_PROJECTION=""
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/slurm-${SLURM_JOBID}_gloria_downstream_${HEAD}_${LOSS}.out"
exec > "$LOG_FILE" 2>&1

echo "Starting job ${SLURM_JOBID}"
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export WANDB_DIR=$home_dir/Project/logs
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
export PYTHONPATH=$home_dir/Project/src
echo "Environment: $MY_CONDA_ENV"

PROJ_FLAG=""
[ -n "$USE_PROJECTION" ] && PROJ_FLAG="--use_projection"

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/gloria_downstream.py \
    --checkpoint    $CHECKPOINT \
    --splits        $SPLITS \
    --out_dir       $home_dir/Project/results \
    --epochs        $EPOCHS \
    --batch_size    $BATCH_SIZE \
    --lr            $LR \
    --dropout       $DROPOUT \
    --hidden_dims   $HIDDEN_DIMS \
    --meta_embed_dim $META_EMBED_DIM \
    --weight_decay  $WEIGHT_DECAY \
    --head          $HEAD \
    --loss          $LOSS \
    --focal_gamma   $FOCAL_GAMMA \
    --class_weighting $CLASS_WEIGHTING \
    $PROJ_FLAG \
    --seed 42 \
    --wandb \
    --wandb_project gloria-downstream \
    --wandb_entity  philipp-wiese
