#!/bin/bash
#SBATCH --job-name=soft_target_viability
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_soft_target_viability.out"

TAU_S_IMG=0.028
TAU_S_BEUR=0.0325
TAU_S_BEF=0.038
TAU_S_JOINT=0.07

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
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$PATH

python $home_dir/Project/src/test_soft_target_viability.py \
    --splits      $home_dir/Project/data/internal_dataset/split_final.json \
    --out_dir     $home_dir/Project/results/soft_target_viability \
    --same_image_boost 20 \
    --tau_s_img   $TAU_S_IMG \
    --tau_s_beur  $TAU_S_BEUR \
    --tau_s_bef   $TAU_S_BEF \
    --tau_s_joint $TAU_S_JOINT \
    --batch_size  128
