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

#$home_dir/Project/results/lace_v2_pretrain/run_20260808_232942
# ── Parameters (edit here) ────────────────────────────────────────────────────
home_dir="/mnt/nfs/homedirs/$USER"
CHECKPOINT=$home_dir/Project/results/lace_v2_pretrain/run_20260728_234136/best_retrieval_checkpoint.pt #"$home_dir/Project/results/biomedclip_pretrain/run_bs128_unfreeze2_20260521_200712/best_r1_checkpoint.pt"
SPLITS="$home_dir/Project/data/internal_dataset/split_binary_final.json"
BINARY=true             # true = benign vs malignant only (requires a split_binary*.json with intermediate excluded)

BATCH_SIZE=64
LR=1.2566974219113507e-05
DROPOUT=0.2124982416168622
WEIGHT_DECAY=0.033234584146834466
EPOCHS=200
PATIENCE=15

# Head mode: mlp (age+sex fusion), mlp_no_meta (no metadata), linear (linear probe)
HEAD="mlp_no_meta"
HIDDEN_DIMS="[128, 64]"
META_EMBED_DIM=0
USE_MASK=true
# Loss: ce, wce, ce_smooth, focal, cb_focal, ldam, balanced_softmax
LOSS="cb_focal"
FOCAL_GAMMA=3.0630964700908816
CB_BETA=0.9910913732770436
LABEL_SMOOTHING=0
# Class weighting: none, inverse, sqrt, effective
CLASS_WEIGHTING="inverse"
# Early stopping metric: val_loss, val_bal_acc
EARLY_STOPPING_METRIC="val_bal_acc"
# ─────────────────────────────────────────────────────────────────────────────

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
    --binary         $BINARY \
    --seed 42 \
    --eval_test
    
    # --wandb \
    # --wandb_project  biomedclip-img-text-downstream \
    # --wandb_entity   philipp-wiese

    
