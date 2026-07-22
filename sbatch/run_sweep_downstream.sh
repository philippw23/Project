#!/bin/bash
#SBATCH --job-name=sweep_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

# Usage:
#   1. Create sweep and get ID:
#        wandb sweep src/biomedclip/eval/sweep_downstream.yaml
#   2. Set SWEEP_ID below and submit:
#        sbatch run_sweep_downstream.sh
SWEEP_ID="philipp-wiese/lace-downstream/4bo44xv1"   # e.g. "philipp-wiese/philipp-wiese/abc12345"

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Set SWEEP_ID in this script before submitting."
    exit 1
fi

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo Starting sweep agent job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export CONDA_EXE=$home_dir/miniconda3/bin/conda
source $home_dir/miniconda3/etc/profile.d/conda.sh
conda activate $MY_CONDA_ENV
echo Environment activated

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export WANDB_DIR=$home_dir/Project/logs
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
export PYTHONPATH=$home_dir/Project/src

python -m wandb agent $SWEEP_ID
