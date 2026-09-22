#!/bin/bash
#SBATCH --job-name=imagenet_img
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --exclude=aioserver2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
SPLITS="/mnt/nfs/homedirs/philippw/Project/data/internal_dataset/split.json"
EPOCHS=100
BATCH_SIZE=64
LR_MLP=3e-4
WEIGHT_DECAY=0.05
DROPOUT=0.3
HIDDEN_DIMS="64"
META_EMBED_DIM=0 # only used if head is mlp, e.g 16

# Head mode: mlp (age+sex fusion), mlp_no_meta (image only)
HEAD="mlp_no_meta"
# Loss: ce, wce, focal, cb_focal, balanced_softmax, ldam
LOSS="focal"
# Class weighting: none, inverse, sqrt, effective
CLASS_WEIGHTING="sqrt"

PATIENCE=100
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/slurm-${SLURM_JOBID}_imagenet_img_${HEAD}_${LOSS}.out"
exec > "$LOG_FILE" 2>&1

echo "Starting job ${SLURM_JOBID} — encoder: ViT-B/16 (ImageNet pretrained) | head: $HEAD"
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
unset PYTORCH_NVML_BASED_CUDA_CHECK
echo "Environment: $MY_CONDA_ENV"

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/imagenet_img/train/downstream.py \
    --head           $HEAD \
    --splits         $SPLITS \
    --out_dir        $home_dir/Project/results/imagenet_img \
    --epochs         $EPOCHS \
    --patience       $PATIENCE \
    --batch_size     $BATCH_SIZE \
    --lr_mlp         $LR_MLP \
    --weight_decay   $WEIGHT_DECAY \
    --dropout        $DROPOUT \
    --hidden_dims    $HIDDEN_DIMS \
    --meta_embed_dim $META_EMBED_DIM \
    --loss           $LOSS \
    --class_weighting $CLASS_WEIGHTING \
    --use_mask \
    --seed           42 \
    --wandb \
    --wandb_project  imagenet-img-downstream \
    --wandb_entity   philipp-wiese
