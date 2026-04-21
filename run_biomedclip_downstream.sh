#!/bin/bash
#SBATCH --job-name=biomedclip_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

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

python_path=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python
$python_path $home_dir/Project/src/biomedclip_downstream.py \
    --checkpoint $home_dir/Project/results/biomedclip_pretrain/best_checkpoint.pt \
    --splits     $home_dir/Project/results/biomedclip_pretrain/splits.json \
    --excel      $home_dir/Project/data/metadata.xlsx \
    --use_mask \
    --lora_layers 4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --dropout 0.3 \
    --hidden_dims 128 128 \
    --meta_embed_dim 16 \
    --seed 42 \
    --wandb \
    --wandb_project biomedclip-downstream \
    --wandb_entity philipp-wiese
