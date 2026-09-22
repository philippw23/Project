#!/bin/bash
#SBATCH --job-name=lace_img_text_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
home_dir="/mnt/nfs/homedirs/$USER"
CHECKPOINT=$home_dir/Project/results/lace_v2_pretrain/run_20260728_234136/best_retrieval_checkpoint.pt
SPLITS="$home_dir/Project/data/internal_dataset/split_binary_final_img_text.json"
BINARY=true             # true = benign vs malignant only (requires a split_binary*.json with intermediate excluded)

# Visual mode: cls [512-dim] | fg [512-dim] | cls_fg [1024-dim] — concatenated with 512-dim text
DOWNSTREAM_VISUAL_MODE="cls_fg"
MAX_TEXT_LEN=384        # tokenisation length for combined befund_en + beurteilung_en text

BATCH_SIZE=64
LR=1e-3
DROPOUT=0.3
WEIGHT_DECAY=0.01
EPOCHS=100
PATIENCE=15

# Head mode: mlp (age+sex fusion), mlp_no_meta (no metadata), linear (linear probe)
HEAD="mlp_no_meta"
HIDDEN_DIMS="256 128"
META_EMBED_DIM=0
USE_MASK=true
# Loss: ce, wce, ce_smooth, focal, cb_focal, ldam, balanced_softmax
LOSS="focal"
FOCAL_GAMMA=2
CB_BETA=0.99
LABEL_SMOOTHING=0
# Class weighting: none, inverse, sqrt, effective
CLASS_WEIGHTING="sqrt"
# Early stopping metric: val_loss, val_bal_acc
EARLY_STOPPING_METRIC="val_bal_acc"
# ─────────────────────────────────────────────────────────────────────────────

export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/lace/slurm-${SLURM_JOBID}_lace_img_text_downstream_${HEAD}_${LOSS}.out"
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

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python \
    $home_dir/Project/src/LACE/train/img_text_downstream.py \
    --checkpoint     $CHECKPOINT \
    --splits         $SPLITS \
    --out_dir        $home_dir/Project/results \
    --downstream_visual_mode $DOWNSTREAM_VISUAL_MODE \
    --max_text_len   $MAX_TEXT_LEN \
    --head           $HEAD \
    --epochs         $EPOCHS \
    --patience       $PATIENCE \
    --early_stopping_metric $EARLY_STOPPING_METRIC \
    --batch_size     $BATCH_SIZE \
    --lr             $LR \
    --dropout        $DROPOUT \
    --hidden_dims    $HIDDEN_DIMS \
    --meta_embed_dim $META_EMBED_DIM \
    --weight_decay   $WEIGHT_DECAY \
    --loss           $LOSS \
    --focal_gamma    $FOCAL_GAMMA \
    --cb_beta        $CB_BETA \
    --label_smoothing $LABEL_SMOOTHING \
    --class_weighting $CLASS_WEIGHTING \
    $( [ "$USE_MASK" = "true" ] && echo "--use_mask" ) \
    --binary         $BINARY \
    --seed 42 \
    --eval_test
