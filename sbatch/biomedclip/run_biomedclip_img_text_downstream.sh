#!/bin/bash
#SBATCH --job-name=biomedclip_img_text_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
CHECKPOINT="/mnt/nfs/homedirs/philippw/Project/results/biomedclip_pretrain/run_bs128_unfreeze2_20260521_200712/best_r1_checkpoint.pt"
SPLITS="/mnt/nfs/homedirs/philippw/Project/data/internal_dataset/split.json"

BATCH_SIZE=64
LR=0.0003
DROPOUT=0.3
WEIGHT_DECAY=0.01
EPOCHS=100
PATIENCE=100

# Head mode: mlp (age+sex fusion), mlp_no_meta (no metadata), linear (linear probe)
HEAD="mlp_no_meta"
HIDDEN_DIMS="64"
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
EARLY_STOPPING_METRIC="val_loss"
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/biomedclip/slurm-${SLURM_JOBID}_biomedclip_img_text_downstream_${HEAD}_${LOSS}.out"
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

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python \
    $home_dir/Project/src/biomedclip_img_text_downstream.py \
    --checkpoint     $CHECKPOINT \
    --splits         $SPLITS \
    --out_dir        $home_dir/Project/results \
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
    --seed 42 \
    --wandb \
    --wandb_project  biomedclip-img-text-downstream \
    --wandb_entity   philipp-wiese
