#!/bin/bash
#SBATCH --job-name=visualize_tsne
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
# Encoder type: biomedclip | chexfound | lace | imagenet
ENCODER_TYPE="chexfound"

# Path to encoder checkpoint (leave empty or set to "none" for base pretrained weights)
CHECKPOINT=""
#/mnt/nfs/homedirs/philippw/Project/results/lace_pretrain/run_20260528_174330/best_retrieval_checkpoint.pt
# Path to splits.json
SPLITS="/mnt/nfs/homedirs/philippw/Project/data/internal_dataset/split.json"

# Which split(s) to visualize: train | val | test | all
SPLIT="test"

# Crop images to lesion bounding box: --use_mask | --no-use_mask
USE_MASK="--use_mask"

# t-SNE hyperparameters
PERPLEXITY=30
N_ITER=1000
SEED=42

BATCH_SIZE=32

# LACE-specific: version v1 or v2 (ignored for other encoders)
LACE_VERSION="v1"

# CheXFound-specific (only required for chexfound encoder)
CHEXFOUND_CONFIG="/mnt/nfs/homedirs/philippw/Project/src/chexfound/configs/chexfound_vitl16_bonetumor.yaml"
CHEXFOUND_WEIGHTS="/mnt/nfs/homedirs/philippw/Project/src/chexfound/data/teacher_checkpoint.pth"  # Required when CHECKPOINT is empty/none

# Output PNG path (leave empty to auto-generate: results/tsne_{encoder}_{checkpoint}.png)
OUTPUT=""
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/slurm-${SLURM_JOBID}_visualize_tsne_${ENCODER_TYPE}.out"
exec > "$LOG_FILE" 2>&1

echo "Starting job ${SLURM_JOBID}"
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
export PYTHONPATH=$home_dir/Project/src
echo "Environment: $MY_CONDA_ENV"

# Build optional args
EXTRA_ARGS=""
[ -n "$CHECKPOINT" ] && EXTRA_ARGS="$EXTRA_ARGS --checkpoint $CHECKPOINT"
[ -n "$OUTPUT" ]     && EXTRA_ARGS="$EXTRA_ARGS --output $OUTPUT"

if [ "$ENCODER_TYPE" = "chexfound" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --chexfound_config $CHEXFOUND_CONFIG"
    [ -n "$CHEXFOUND_WEIGHTS" ] && EXTRA_ARGS="$EXTRA_ARGS --chexfound_weights $CHEXFOUND_WEIGHTS"
fi

if [ "$ENCODER_TYPE" = "lace" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --version $LACE_VERSION"
fi

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/data/visualize_tsne.py \
    --encoder_type  $ENCODER_TYPE \
    --splits        $SPLITS \
    --split         $SPLIT \
    $USE_MASK \
    --perplexity    $PERPLEXITY \
    --n_iter        $N_ITER \
    --seed          $SEED \
    --batch_size    $BATCH_SIZE \
    $EXTRA_ARGS
