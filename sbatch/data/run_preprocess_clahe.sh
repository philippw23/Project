#!/bin/bash
#SBATCH --job-name=preprocess_clahe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
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
export PYTHONPATH=$home_dir/Project/src
echo Environment activated

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/preprocess_clahe.py \
    --input_dir $home_dir/Project/data/internal_dataset/images \
    --output_dir $home_dir/Project/data/internal_dataset/images_clahe
