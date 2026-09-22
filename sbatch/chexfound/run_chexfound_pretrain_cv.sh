#!/bin/bash
#SBATCH --job-name=chexfound_pretrain_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/chexfound/slurm-%j_chexfound_pretrain_cv_binary.out"
# Note: bump --time above roughly N x a normal single-run time budget — CV runs
# N folds back to back in a single job (see CV_DIR below).

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export SSL_CERT_FILE=$home_dir/miniconda3/envs/$MY_CONDA_ENV/ssl/cert.pem
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export WANDB_DIR=$home_dir/Project/logs
export PYTHONPATH=$home_dir/Project/src
# Avoid PyTorch NVML allocator crashes on nodes where libnvidia-ml is visible
# but nvmlInit fails. This must be set before Python imports torch.
unset PYTORCH_NVML_BASED_CUDA_CHECK
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
echo Environment activated

echo "CUDA/NVML preflight:"
nvidia-smi || echo "WARNING: nvidia-smi failed on $(hostname) (NVML mismatch) — CUDA may still work."

# ── Cross-validation ─────────────────────────────────────────────────────────
# One full pretraining pass per fold file, results land under
# out_dir/fold0/, fold1/, ... Mutually exclusive with SPLITS/--sweep.
# cv/ (3-class, split_fold*.json) matches split_final.json used below for the
# single-split equivalent — switch to cv_binary/ + split_binary_fold*.json for
# a binary CV pretrain run.
CV_DIR=$home_dir/Project/data/internal_dataset/cv_binary
CV_PATTERN="split_binary_fold*.json"

EPOCHS=30
BATCH_SIZE=4
OUT_DIR=$home_dir/Project/results/chexfound_pretrain_sweep

# Run the Python script
$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python \
    $home_dir/Project/src/chexfound/train/pretrain.py \
    --config     $home_dir/Project/src/chexfound/configs/chexfound_vitl16_bonetumor_binary.yaml \
    --base_cfg   $home_dir/Project/src/chexfound/data/config.yaml \
    --out_dir    $OUT_DIR \
    --epochs     $EPOCHS \
    --batch_size $BATCH_SIZE \
    --cv_dir     $CV_DIR \
    --cv_pattern "$CV_PATTERN" \
    --wandb \
    --wandb_project chexfound-pretrain \
    --wandb_entity  philipp-wiese
