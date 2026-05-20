#!/bin/bash
#SBATCH --job-name=biomedclip_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
#run_bs128_unfreeze11_20260516_125128
#run_bs128_lora11_20260516_125104
#run_bs128_lora4_20260516_125103
CHECKPOINT="/mnt/nfs/homedirs/philippw/Project/results/biomedclip_pretrain/run_bs128_lora4_20260519_191014/best_r1_checkpoint.pt"
SPLITS="/mnt/nfs/homedirs/philippw/Project/data/internal_dataset/split.json"
BATCH_SIZE=32
LR=0.0003181387905557346
DROPOUT=0.3
HIDDEN_DIMS="64 32"
META_EMBED_DIM=16
WEIGHT_DECAY=0.01
EPOCHS=50

# Head mode: mlp (age+sex fusion), mlp_no_meta (no metadata), linear (linear probe)
HEAD="mlp_no_meta"
# Loss: ce, wce, ce_smooth, focal, cb_focal, ldam, balanced_softmax
LOSS="focal"
# Class weighting: none, inverse, sqrt, effective
CLASS_WEIGHTING="sqrt"
# Encoder fine-tuning: 0 = frozen (linear probing), N = LoRA last N blocks
FINETUNE_LORA_LAYERS=4
LR_ENCODER=1e-5
FINETUNE_LORA_R=8
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/slurm-${SLURM_JOBID}_biomedclip_downstream_${HEAD}_${LOSS}.out"
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

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/biomedclip_downstream.py \
    --checkpoint    $CHECKPOINT \
    --splits        $SPLITS \
    --out_dir       $home_dir/Project/results \
    --use_mask \
    --epochs        $EPOCHS \
    --batch_size    $BATCH_SIZE \
    --lr            $LR \
    --dropout       $DROPOUT \
    --hidden_dims   $HIDDEN_DIMS \
    --meta_embed_dim $META_EMBED_DIM \
    --weight_decay  $WEIGHT_DECAY \
    --head          $HEAD \
    --loss          $LOSS \
    --class_weighting $CLASS_WEIGHTING \
    --finetune_lora_layers $FINETUNE_LORA_LAYERS \
    --lr_encoder    $LR_ENCODER \
    --finetune_lora_r $FINETUNE_LORA_R \
    --seed 42 \
    --wandb \
    --wandb_project biomedclip-downstream \
    --wandb_entity  philipp-wiese
