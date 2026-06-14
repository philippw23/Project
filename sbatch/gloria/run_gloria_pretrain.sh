#!/bin/bash
#SBATCH --job-name=gloria_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
CHECKPOINT="/mnt/nfs/homedirs/philippw/Project/src/gloria/pretrained/chexpert_resnet50.ckpt"
SPLITS="/mnt/nfs/homedirs/philippw/Project/data/internal_dataset/split.json"

# Adapter: lora or unfreeze
ADAPTER_MODE="lora"
N_LAYERS=4          # number of last ResNet layer3 blocks to adapt (layer4 always included)
LORA_R=8
LORA_ALPHA=16

BATCH_SIZE=128
LR=1e-4
WEIGHT_DECAY=0.01
EPOCHS=50
WARMUP_EPOCHS=5
PATIENCE=15
MAX_TEXT_LEN=256

# GLoRIA loss temperatures (keep GLoRIA paper defaults)
# TEMP1=4.0
# TEMP2=5.0
# TEMP3=10.0
TEMP1=0.07 # set all temps to the same value for simplicity; tune this single temp1 for best performance (0.07 worked well in preliminary experiments)
TEMP2=0.07 
TEMP3=0.07
LOCAL_LOSS_WEIGHT=1.0
GLOBAL_LOSS_WEIGHT=1.0
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/slurm-${SLURM_JOBID}_gloria_pretrain_${ADAPTER_MODE}${N_LAYERS}.out"
exec > "$LOG_FILE" 2>&1

echo "Starting job ${SLURM_JOBID}"
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
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
    --splits            $SPLITS \
    --out_dir           $home_dir/Project/results \
    --adapter_mode      $ADAPTER_MODE \
    --use_mask \
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
