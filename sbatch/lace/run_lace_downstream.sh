#!/bin/bash
#SBATCH --job-name=lace_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export CONDA_EXE=$home_dir/miniconda3/bin/conda
source $home_dir/miniconda3/etc/profile.d/conda.sh
conda activate $MY_CONDA_ENV
echo Environment activated

export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export WANDB_DIR=$home_dir/Project/logs
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$PATH

# ── Hyperparameter ────────────────────────────────────────────────────────────
VERSION=v1       # v1: ViT CLS token → MalignancyMLP | v2: ViT + MaskTokenModule → LACEv2Classifier (requires v2 pretrain checkpoint)
HEAD=mlp_no_meta   # Head mode 
CHECKPOINT=$home_dir/Project/results/lace_pretrain/run_20260613_202920/best_retrieval_checkpoint.pt  # path to pretrained checkpoint (set to "" to train from scratch)
#results/lace_pretrain/run_20260606_145451/best_retrieval_checkpoint.pt  # path to pretrained checkpoint (set to "" to train from scratch)
# run_20260530_104844
EPOCHS=50
PATIENCE=10
BATCH_SIZE=16
LR=0.00009447014464909024
META_EMBED_DIM=32
HIDDEN_DIMS="128 64"
LOSS=focal
FOCAL_GAMMA=3.05351237670772
CLASS_WEIGHTING=inverse
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

python $home_dir/Project/src/lace_downstream.py \
    --version    $VERSION \
    --head       $HEAD \
    --checkpoint $CHECKPOINT \
    --splits     $home_dir/Project/data/internal_dataset/split.json \
    --out_dir    $home_dir/Project/results \
    --epochs     $EPOCHS \
    --patience   $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr         $LR \
    --meta_embed_dim $META_EMBED_DIM \
    --hidden_dims $HIDDEN_DIMS \
    --loss       $LOSS \
    --focal_gamma $FOCAL_GAMMA \
    --class_weighting $CLASS_WEIGHTING \
    --seed       $SEED \
    --wandb \
    --wandb_project lace-downstream \
    --wandb_entity  philipp-wiese
