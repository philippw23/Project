#!/bin/bash
#SBATCH --job-name=test_scratch_overfit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --exclude=aioserver2
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_test_scratch_overfit.out"
#SBATCH --error="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_test_scratch_overfit.err"

# Overfit sanity check: trains scratch_img and scratch_img_text on 8 real
# samples for 400 epochs with no regularization and checks training loss drops.

SPLITS="${1:-$HOME/Project/results/biomedclip_pretrain/run_20260428_041058/splits.json}"
ENCODER="${2:-resnet18}"

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo "Starting overfit test job ${SLURM_JOBID} — encoder: $ENCODER"

MY_CONDA_ENV="master"
export CONDA_EXE=$home_dir/miniconda3/bin/conda
source $home_dir/miniconda3/etc/profile.d/conda.sh
conda activate $MY_CONDA_ENV
echo "Environment activated"

export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$PATH
export PYTHONPATH=$home_dir/Project/src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
unset PYTORCH_NVML_BASED_CUDA_CHECK

echo ""
echo "============================================================"
echo "BASELINE 1: scratch_img (image-only)"
echo "============================================================"
python $home_dir/Project/src/scratch_img_downstream.py \
    --encoder        $ENCODER \
    --splits         $SPLITS \
    --excel          $home_dir/Project/data/internal_dataset/metadata.xlsx \
    --out_dir        $home_dir/Project/results \
    --overfit_n      8 \
    --epochs         400 \
    --patience       400 \
    --batch_size     8 \
    --lr_encoder     1e-3 \
    --lr_mlp         1e-3 \
    --warmup_epochs  0 \
    --weight_decay   0.0 \
    --dropout        0.0 \
    --hidden_dims    128 \
    --loss           ce \
    --class_weighting none \
    --seed           42

echo ""
echo "============================================================"
echo "BASELINE 2: scratch_img_text (image + text)"
echo "============================================================"
python $home_dir/Project/src/scratch_img_text_downstream.py \
    --encoder        $ENCODER \
    --splits         $SPLITS \
    --excel          $home_dir/Project/data/internal_dataset/metadata.xlsx \
    --out_dir        $home_dir/Project/results \
    --overfit_n      8 \
    --epochs         400 \
    --patience       400 \
    --batch_size     8 \
    --lr_encoder     1e-3 \
    --lr_mlp         1e-3 \
    --warmup_epochs  0 \
    --weight_decay   0.0 \
    --dropout        0.0 \
    --hidden_dims    128 \
    --loss           ce \
    --class_weighting none \
    --seed           42
