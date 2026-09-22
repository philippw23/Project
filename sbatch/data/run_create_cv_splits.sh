#!/bin/bash
#SBATCH --job-name=create_cv_splits
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
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

export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$PATH

python $home_dir/Project/src/data/create_cv_splits.py \
    --input   $home_dir/Project/data/internal_dataset/split_binary.json \
    --out_dir $home_dir/Project/data/internal_dataset/cv \
    --folds 10 \
    --val_frac 0.1 \
    --seed 42
