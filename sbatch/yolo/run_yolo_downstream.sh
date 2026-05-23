#!/bin/bash
#SBATCH --job-name=yolo_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
WEIGHTS="yolov5l6u.pt"   # or yolo26n.pt
IMGSZ=640
EPOCHS=100
BATCH=16
LR_MLP=3e-4
PATIENCE=15
LAMBDA_BBOX=1.0
LOSS="ce"
CLASS_WEIGHTING="sqrt"
# FINETUNE="--finetune --lr_backbone 1e-5"   # uncomment to fine-tune backbone
FINETUNE=""
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/slurm-${SLURM_JOBID}_yolo_downstream.out"
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
    $home_dir/Project/src/yolo/train/downstream.py \
    --weights        $WEIGHTS \
    --imgsz          $IMGSZ \
    --epochs         $EPOCHS \
    --batch_size     $BATCH \
    --lr_mlp         $LR_MLP \
    --patience       $PATIENCE \
    --lambda_bbox    $LAMBDA_BBOX \
    --loss           $LOSS \
    --class_weighting $CLASS_WEIGHTING \
    $FINETUNE \
    --wandb \
    --wandb_project  yolo-downstream \
    --wandb_entity   philipp-wiese
