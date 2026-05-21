#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
DATA="/mnt/nfs/homedirs/philippw/Project/data/yolo/dataset.yaml"
WEIGHTS="yolov5l6u.pt"
IMGSZ=1024
EPOCHS=50
BATCH=8
LR0=0.01
PATIENCE=20
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/slurm-${SLURM_JOBID}_yolo_train.out"
exec > "$LOG_FILE" 2>&1

echo "Starting job ${SLURM_JOBID}"
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME=$home_dir/.cache/huggingface
export WANDB_DIR=$home_dir/Project/logs
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
export PYTHONPATH=$home_dir/Project/src
echo "Environment: $MY_CONDA_ENV"

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python \
    $home_dir/Project/src/yolo/train/train.py \
    --data      $DATA \
    --weights   $WEIGHTS \
    --imgsz     $IMGSZ \
    --epochs    $EPOCHS \
    --batch     $BATCH \
    --lr0       $LR0 \
    --patience  $PATIENCE \
    --workers   4 \
    --device    0 \
    --project   $home_dir/Project/results/yolo \
    --name      run \
    --seed      42 \
    --wandb \
    --wandb_project yolo-malignancy \
    --wandb_entity  philipp-wiese
