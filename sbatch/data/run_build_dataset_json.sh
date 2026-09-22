#!/bin/bash
#SBATCH --job-name=build_dataset_json
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}

MY_CONDA_ENV="master"
export CONDA_EXE=$home_dir/miniconda3/bin/conda
source $home_dir/miniconda3/etc/profile.d/conda.sh
conda activate $MY_CONDA_ENV

python_path=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python
$python_path $home_dir/Project/src/data/create_dataset.py \
    --reports_path $home_dir/Project/data/internal_dataset/test/full_reports_14B_260714_1053.json \
    --output_path $home_dir/Project/data/internal_dataset/dataset_full_14B_260714_1053.json
