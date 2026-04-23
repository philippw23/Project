#!/bin/bash
#SBATCH --job-name=biomedclip_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
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

python $home_dir/Project/src/biomedclip_pretrain.py \
    --excel    $home_dir/Project/data/metadata.xlsx \
    --reports  $home_dir/Project/data/text/translated_reports.json \
    --english \
    --images   $home_dir/Project/data/images \
    --masks    $home_dir/Project/data/segmentations \
    --out_dir  $home_dir/Project/results \
    --use_mask \
    --lora_layers 4 \
    --lora_r 8 \
    --lora_alpha 64 \
    --batch_size 32 \
    --epochs 50 \
    --lr 5e-5 \
    --downstream_train_frac 0.8 \
    --downstream_val_frac 0.1 \
    --test_frac 0.1 \
    --seed 42 \
    --wandb \
    --wandb_project biomedclip-pretrain \
    --wandb_entity philipp-wiese


# --lora_layers 2 \
# --lora_r 4 \
# --lora_alpha 16 \
# --batch_size 32 \
# --epochs 50 \
# --lr 1e-5 \
