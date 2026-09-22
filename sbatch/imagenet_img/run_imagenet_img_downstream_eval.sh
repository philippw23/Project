#!/bin/bash
#SBATCH --job-name=imagenet_img_downstream_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_imagenet_img_downstream_eval.out"

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
export PYTHONPATH=$home_dir/Project/src

# All architecture/loss hyperparameters + train age-stats are read from the head
# checkpoint itself. Optional: --btxrd_manifest to also score BTXRD (requires the
# head to have been trained with --binary).
HEAD_CHECKPOINT=$home_dir/Project/results/imagenet_img/<run>/best_checkpoint.pt   # <-- set this
SPLITS=$home_dir/Project/data/internal_dataset/split_binary.json
BTXRD_MANIFEST=$home_dir/Project/data/BTXRD/btxrd_downstream_binary.json
# ─────────────────────────────────────────────────────────────────────────────

python $home_dir/Project/src/imagenet_img/train/downstream_eval.py \
    --head_checkpoint $HEAD_CHECKPOINT \
    --splits          $SPLITS \
    --btxrd_manifest  $BTXRD_MANIFEST
