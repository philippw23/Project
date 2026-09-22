#!/bin/bash
#SBATCH --job-name=chexfound_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Parameters (edit here) ────────────────────────────────────────────────────
# Set CHECKPOINT to a continued-pretrain .pth path, or "none" to use frozen original weights
CHECKPOINT="/mnt/nfs/homedirs/philippw/Project/results/chexfound_pretrain_sweep/o98dx8kf/checkpoint_last.pth"
CHEXFOUND_WEIGHTS="/mnt/nfs/homedirs/philippw/Project/src/chexfound/data/teacher_checkpoint.pth"
IMAGE_SIZE=512   # 512 = native CheXFound | 224 = BiomedCLIP-equivalent resolution
SPLITS="/mnt/nfs/homedirs/philippw/Project/data/internal_dataset/split_final.json"
BINARY="false"  # true | false
HEAD="mlp_no_meta"   # linear | mlp | mlp_no_meta
BATCH_SIZE=16
LR=0.0001851273023136428
DROPOUT=0.2
HIDDEN_DIMS="128"
META_EMBED_DIM=16
WEIGHT_DECAY=0.01
EPOCHS=1
LOSS="cb_focal"  # ce | wce | ce_smooth | focal | cb_focal | ldam | balanced_softmax
FOCAL_GAMMA=3.0
CLASS_WEIGHTING="effective"  # none | inverse | sqrt | effective
CB_BETA=0.99  # only active for cb_focal
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/chexfound/slurm-${SLURM_JOBID}_chexfound_downstream.out"
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

$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python $home_dir/Project/src/chexfound/train/downstream.py \
    --checkpoint    $CHECKPOINT \
    --chexfound_weights $CHEXFOUND_WEIGHTS \
    --image_size    $IMAGE_SIZE \
    --splits        $SPLITS \
    --out_dir       $home_dir/Project/results \
    --use_mask \
    --head          $HEAD \
    --epochs        $EPOCHS \
    --batch_size    $BATCH_SIZE \
    --lr            $LR \
    --dropout       $DROPOUT \
    --hidden_dims   $HIDDEN_DIMS \
    --meta_embed_dim $META_EMBED_DIM \
    --weight_decay  $WEIGHT_DECAY \
    --loss          $LOSS \
    --class_weighting $CLASS_WEIGHTING \
    --focal_gamma $FOCAL_GAMMA \
    --cb_beta $CB_BETA \
    --seed 42 \
    --wandb \
    --wandb_project chexfound-downstream \
    --wandb_entity  philipp-wiese \
    --binary        $BINARY \
    --sweep \
    --eval_test
