#!/bin/bash
#SBATCH --job-name=chexfound_arch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --exclude=aioserver2
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
source $home_dir/miniconda3/etc/profile.d/conda.sh
# conda.sh hardcodes /home/philippw paths; override with NFS paths for compute nodes
export CONDA_EXE=$home_dir/miniconda3/bin/conda
export _CONDA_EXE=$home_dir/miniconda3/bin/conda
export CONDA_PYTHON_EXE=$home_dir/miniconda3/bin/python
export _CONDA_ROOT=$home_dir/miniconda3
conda activate $MY_CONDA_ENV
echo Environment activated

export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export PYTHONUNBUFFERED=1

python_path=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python
$python_path $home_dir/Project/src/inspect_tensor_shapes.py \
    --model   chexfound \
    --config  $home_dir/Project/configs/chexfound_vitl16_bonetumor.yaml \
    --weights $home_dir/Project/src/chexfound/data/teacher_checkpoint.pth
