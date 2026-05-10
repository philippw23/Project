#!/bin/bash
#SBATCH --job-name=downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --exclude=aioserver2
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"
#SBATCH --error="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.err"

# Usage:
#   sbatch run_downstream.sh biomedclip
#   sbatch run_downstream.sh chexfound

BASELINE="${1:?Usage: sbatch run_downstream.sh (biomedclip|chexfound)}"

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo "Starting job ${SLURM_JOBID} — baseline: $BASELINE"
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export WANDB_DIR=$home_dir/Project/logs
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
export PYTHONPATH=$home_dir/Project/src
unset PYTORCH_NVML_BASED_CUDA_CHECK
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
echo "Environment: $MY_CONDA_ENV"

if [ "$BASELINE" = "biomedclip" ]; then
    $home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/downstream.py \
        --baseline biomedclip \
        --biomedclip_checkpoint $home_dir/Project/results/biomedclip_pretrain/run_20260424_154249/best_r1_checkpoint.pt \
        --splits     $home_dir/Project/results/biomedclip_pretrain/run_20260424_154249/splits.json \
        --excel      $home_dir/Project/data/internal_dataset/metadata.xlsx \
        --out_dir    $home_dir/Project/results \
        --use_mask \
        --epochs 50 \
        --batch_size 32 \
        --lr 0.0004537818939304447 \
        --dropout 0.3 \
        --hidden_dims 128 \
        --meta_embed_dim 16 \
        --seed 42 \
        --wandb \
        --wandb_project biomedclip-downstream \
        --wandb_entity philipp-wiese

elif [ "$BASELINE" = "chexfound" ]; then
    $home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/downstream.py \
        --baseline chexfound \
        --chexfound_checkpoint $home_dir/Project/results/chexfound_pretrain/job_6306/checkpoint_last.pth \
        --splits    $home_dir/Project/results/chexfound_pretrain/job_6306/biomedclip_pretrain/splits.json \
        --excel     $home_dir/Project/data/internal_dataset/metadata.xlsx \
        --out_dir   $home_dir/Project/results \
        --use_mask \
        --epochs    50 \
        --batch_size 64 \
        --lr        1e-3 \
        --dropout   0.3 \
        --hidden_dims 256 128 \
        --meta_embed_dim 16 \
        --weight_decay 0.01 \
        --loss      ce \
        --class_weighting sqrt \
        --seed      42 \
        --wandb \
        --wandb_project chexfound-downstream \
        --wandb_entity  philipp-wiese

else
    echo "ERROR: Unknown baseline '$BASELINE'. Use: biomedclip | chexfound"
    exit 1
fi
