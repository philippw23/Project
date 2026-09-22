#!/bin/bash
#SBATCH --job-name=gloria_pretrain_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Note: CV runs N folds back to back in a single job, so --time needs roughly
# N x a normal single-run time budget.

# ── Cross-validation ────────────────────────────────────────────────────────────
# One full pretraining pass per fold file, results land under
# out_dir/gloria_pretrain/gloria_pretrain_<tag>_<timestamp>/foldN/.
home_dir="/mnt/nfs/homedirs/$USER"
CV_DIR=$home_dir/Project/data/internal_dataset/cv_binary
CV_PATTERN="split_binary_fold*.json"

# ── Parameters (edit here) ────────────────────────────────────────────────────
CHECKPOINT="/mnt/nfs/homedirs/philippw/Project/src/gloria/pretrained/chexpert_resnet50.ckpt"

# Adapter: lora or unfreeze
ADAPTER_MODE="unfreeze"
N_LAYERS=4          # number of last ResNet layer3 blocks to adapt (layer4 always included)
LORA_R=8
LORA_ALPHA=16

BATCH_SIZE=32
LR=0.0004347797382475661
WEIGHT_DECAY=0.005296255392794296
EPOCHS=50
WARMUP_EPOCHS=5
PATIENCE=15
MAX_TEXT_LEN=97

TEMP1=6
TEMP2=6
TEMP3=10
LOCAL_LOSS_WEIGHT=1
GLOBAL_LOSS_WEIGHT=0.5
# ─────────────────────────────────────────────────────────────────────────────

export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/gloria/slurm-${SLURM_JOBID}_gloria_pretrain_cv_${ADAPTER_MODE}${N_LAYERS}.out"
mkdir -p "$(dirname "$LOG_FILE")"
exec > "$LOG_FILE" 2>&1

echo "Starting job ${SLURM_JOBID}"
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export SSL_CERT_FILE=$home_dir/miniconda3/envs/$MY_CONDA_ENV/ssl/cert.pem
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export WANDB_DIR=$home_dir/Project/logs
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
export PYTHONPATH=$home_dir/Project/src
echo "Environment: $MY_CONDA_ENV"

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/gloria_pretrain.py \
    --checkpoint        $CHECKPOINT \
    --cv_dir            $CV_DIR \
    --cv_pattern        "$CV_PATTERN" \
    --out_dir           $home_dir/Project/results \
    --adapter_mode      $ADAPTER_MODE \
    --n_layers          $N_LAYERS \
    --lora_r            $LORA_R \
    --lora_alpha        $LORA_ALPHA \
    --batch_size        $BATCH_SIZE \
    --lr                $LR \
    --weight_decay      $WEIGHT_DECAY \
    --epochs            $EPOCHS \
    --warmup_epochs     $WARMUP_EPOCHS \
    --patience          $PATIENCE \
    --max_text_len      $MAX_TEXT_LEN \
    --temp1             $TEMP1 \
    --temp2             $TEMP2 \
    --temp3             $TEMP3 \
    --local_loss_weight  $LOCAL_LOSS_WEIGHT \
    --global_loss_weight $GLOBAL_LOSS_WEIGHT \
    --seed 42 \
    --wandb \
    --wandb_project gloria-pretrain \
    --wandb_entity  philipp-wiese
