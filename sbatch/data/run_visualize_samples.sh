#!/bin/bash
#SBATCH --job-name=visualize_samples
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
export PYTHONPATH=$home_dir/Project/src
export MPLCONFIGDIR=$home_dir/.config/matplotlib
echo Environment activated

GLOBAL_CONTEXT_FRACTION=0.3
CONTEXT_FRACTION=0.05

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/visualize_samples.py \
    --samples 10 \
    --global_context_fraction $GLOBAL_CONTEXT_FRACTION \
    --context_fraction $CONTEXT_FRACTION \
    #--btxrd \
    #--seed 42
