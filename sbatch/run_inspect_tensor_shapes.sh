#!/bin/bash
#SBATCH --job-name=inspect_shapes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

home_dir="/mnt/nfs/homedirs/$USER"
cd "${SLURM_SUBMIT_DIR:-$home_dir/Project}"

echo "Starting job ${SLURM_JOBID:-local}"
echo "SLURM assigned me these nodes:"
squeue -j "${SLURM_JOBID}" -O nodelist 2>/dev/null | tail -n +2

MY_CONDA_ENV="master"
export CONDA_EXE="$home_dir/miniconda3/bin/conda"
source "$home_dir/miniconda3/etc/profile.d/conda.sh"
conda activate "$MY_CONDA_ENV"
echo "Environment activated"

export HF_HOME="$home_dir/.cache/huggingface"
export TRANSFORMERS_CACHE="$home_dir/.cache/huggingface/transformers"

python_path="$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python"
"$python_path" "$home_dir/Project/src/inspect_tensor_shapes.py"
