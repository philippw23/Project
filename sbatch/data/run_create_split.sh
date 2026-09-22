#!/bin/bash
#SBATCH --job-name=create_split
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

export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$PATH

python $home_dir/Project/src/data/create_split.py \
    --dataset  $home_dir/Project/data/internal_dataset/dataset_full_14B_260714_1053.json \
    --out_dir  $home_dir/Project/data/internal_dataset/test \
    --downstream_train_frac 0.8 \
    --downstream_val_frac 0.1 \
    --test_frac 0.1 \
    --seed 42 \
    --binary
    # Add --binary to exclude intermediate cases and save to split_binary.json
