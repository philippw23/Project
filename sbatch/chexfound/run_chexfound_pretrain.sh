#!/bin/bash
#SBATCH --job-name=chexfound_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/chexfound/slurm-%j.out"
#SBATCH --error="/mnt/nfs/homedirs/%u/Project/logs/chexfound/slurm-%j.err"


home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
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
# Set CV_DIR to run one full pretraining pass per fold file instead of a single
# run (results land under out_dir/fold0/, fold1/, ...). Leave empty for a
# normal single-split run using the config's baked-in dataset_path (override
# with SPLITS below). Mutually exclusive with SPLITS/--sweep.
CV_DIR=""   # e.g. $home_dir/Project/data/internal_dataset/cv
CV_PATTERN="split_binary_fold*.json"
SPLITS="$home_dir/Project/data/internal_dataset/split_final.json"   # e.g. $home_dir/Project/data/internal_dataset/split_final.json — ignored when CV_DIR is set

EPOCHS=30
BATCH_SIZE=4
OUT_DIR=$home_dir/Project/results/chexfound_pretrain_sweep

# Run the Python script
$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python \
    $home_dir/Project/src/chexfound/train/pretrain.py \
    --config     $home_dir/Project/src/chexfound/configs/chexfound_vitl16_bonetumor.yaml \
    --base_cfg   $home_dir/Project/src/chexfound/data/config.yaml \
    --out_dir    $OUT_DIR \
    --epochs     $EPOCHS \
    --batch_size $BATCH_SIZE \
    $( [ -n "$CV_DIR" ] && echo "--cv_dir $CV_DIR" ) \
    $( [ -n "$CV_DIR" ] && [ -n "$CV_PATTERN" ] && echo "--cv_pattern $CV_PATTERN" ) \
    $( [ -z "$CV_DIR" ] && [ -n "$SPLITS" ] && echo "--splits $SPLITS" ) \
    --wandb \
    --wandb_project chexfound-pretrain \
    --wandb_entity  philipp-wiese
