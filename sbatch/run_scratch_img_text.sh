#!/bin/bash
#SBATCH --job-name=scratch_img_text
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --exclude=aioserver2
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_scratch_img_text.out"
#SBATCH --error="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_scratch_img_text.err"

# Usage:
#   sbatch run_scratch_img_text.sh resnet18
#   sbatch run_scratch_img_text.sh vit_tiny

ENCODER="${1:?Usage: sbatch run_scratch_img_text.sh (resnet18|vit_tiny)}"

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo "Starting job ${SLURM_JOBID} — encoder: $ENCODER"
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
export PYTHONPATH=$home_dir/Project/src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
unset PYTORCH_NVML_BASED_CUDA_CHECK

python $home_dir/Project/src/scratch_img_text_downstream.py \
    --encoder           $ENCODER \
    --splits            $home_dir/Project/results/biomedclip_pretrain/run_YYYYMMDD_HHMMSS/splits.json \
    --excel             $home_dir/Project/data/internal_dataset/metadata.xlsx \
    --out_dir           $home_dir/Project/results \
    --epochs            100 \
    --patience          15 \
    --batch_size        32 \
    --lr_encoder        1e-3 \
    --lr_mlp            1e-3 \
    --warmup_epochs     10 \
    --weight_decay      0.05 \
    --dropout           0.3 \
    --hidden_dims       128 \
    --meta_embed_dim    16 \
    --loss              ce \
    --class_weighting   sqrt \
    --text_n_layers     4 \
    --text_hidden_dim   256 \
    --text_n_heads      4 \
    --proj_dim          256 \
    --lambda_contrastive 0.1 \
    --use_mask \
    --seed              42 \
    --wandb \
    --wandb_project     scratch-img-text-downstream \
    --wandb_entity      philipp-wiese
