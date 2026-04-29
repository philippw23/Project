#!/bin/bash
#SBATCH --job-name=chexfound_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
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

# CheXFound repo must be cloned at this path before submitting.
export CHEXFOUND_ROOT=$home_dir/CheXFound
export PYTHONPATH=$CHEXFOUND_ROOT

# Copy the BoneTumorDataset into the CheXFound repo (idempotent).
cp $home_dir/Project/src/chexfound/bone_tumor_patch/bone_tumor.py \
   $CHEXFOUND_ROOT/chexfound/data/datasets/bone_tumor.py

# Register BoneTumorDataset if not already present.
grep -q "BoneTumorDataset" $CHEXFOUND_ROOT/chexfound/data/datasets/__init__.py || \
    echo "from .bone_tumor import BoneTumorDataset" \
    >> $CHEXFOUND_ROOT/chexfound/data/datasets/__init__.py

torchrun --nproc_per_node=1 \
    $CHEXFOUND_ROOT/chexfound/train/train.py \
    --config-file $home_dir/Project/configs/chexfound_vitl16_bonetumor.yaml \
    --output-dir  $home_dir/Project/results/chexfound_pretrain
