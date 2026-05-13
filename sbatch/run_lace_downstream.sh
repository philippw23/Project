#!/bin/bash
#SBATCH --job-name=lace_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
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
export WANDB_DIR=$home_dir/Project/logs
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$PATH

python $home_dir/Project/src/lace_downstream.py \
    --version    v2 \
    --checkpoint $home_dir/Project/results/lace_v2_pretrain/run_YYYYMMDD_HHMMSS/best_checkpoint.pt \
    --splits     $home_dir/Project/results/lace_v2_pretrain/run_YYYYMMDD_HHMMSS/splits.json \
    --out_dir    $home_dir/Project/results \
    --epochs     50 \
    --patience   10 \
    --batch_size 64 \
    --lr         1e-3 \
    --meta_embed_dim 32 \
    --loss       wce \
    --class_weighting sqrt \
    --seed       42 \
    --wandb \
    --wandb_project lace-downstream \
    --wandb_entity  philipp-wiese
