#!/bin/bash
#SBATCH --job-name=build_btxrd_downstream
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
export SSL_CERT_FILE=$home_dir/miniconda3/envs/$MY_CONDA_ENV/ssl/cert.pem
export CONDA_EXE=$home_dir/miniconda3/bin/conda
source $home_dir/miniconda3/etc/profile.d/conda.sh
conda activate $MY_CONDA_ENV
echo Environment activated

export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$PATH

# Rasterizes BTXRD polygon annotations to PNG masks and writes the downstream
# manifest data/BTXRD/btxrd_downstream_binary.json used as the external test set.
python $home_dir/Project/src/build_btxrd_downstream.py \
    --btxrd_dir $home_dir/Project/data/BTXRD
