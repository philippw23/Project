#!/bin/bash
#SBATCH --job-name=biomedclip_zeroshot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
CHECKPOINT="/mnt/nfs/homedirs/philippw/Project/results/biomedclip_pretrain/run_bs128_unfreeze4_20260516_153641/best_r1_checkpoint.pt"
SPLITS="/mnt/nfs/homedirs/philippw/Project/data/internal_dataset/split.json"
BATCH_SIZE=64
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/slurm-${SLURM_JOBID}_biomedclip_zeroshot.out"
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

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/biomedclip_zeroshot.py \
    --checkpoint    $CHECKPOINT \
    --splits        $SPLITS \
    --batch_size    $BATCH_SIZE \
    --seed 42 \
    --wandb \
    --wandb_project biomedclip-zeroshot \
    --wandb_entity  philipp-wiese
