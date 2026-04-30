#!/bin/bash
#SBATCH --job-name=chexfound_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --exclude=aioserver2
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"
#SBATCH --error="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.err"

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
export PYTHONPATH=$home_dir/Project/src

# No external CheXFound repo needed.
#
# Mode A (frozen CheXFound baseline): add --checkpoint none
#         and --chexfound_weights /path/to/chexfound_vitl16.pth
# Mode B (bundled continued-pretrain checkpoint, DEFAULT): no extra flags needed.
# Mode B (custom checkpoint): add --checkpoint /path/to/custom.pth

python $home_dir/Project/src/chexfound_downstream.py \
    --chexfound_weights $home_dir/Project/src/chexfound/data/teacher_checkpoint.pth \
    --checkpoint none \
    --splits    $home_dir/Project/results/biomedclip_pretrain/run_20260428_121303/splits.json \
    --excel     $home_dir/Project/data/metadata.xlsx \
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
