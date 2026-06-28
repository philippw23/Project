#!/bin/bash
#SBATCH --job-name=lacev2_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_lacev2_downstream.out"

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

# ── Model ─────────────────────────────────────────────────────────────────────
VERSION=v1              # v1: CLS token | v2: MaskTokenDecoder (requires v2 pretrain ckpt)
VISUAL_MODE=cls      # cls [B,512] | fg [B,512] | cls_fg [B,1024]
CHECKPOINT=$home_dir/Project/results/lace_pretrain/run_20260614_025925/best_retrieval_checkpoint.pt
# # results/lace_v2_pretrain/run_20260624_121345/best_checkpoint.pt
SPLITS=$home_dir/Project/data/internal_dataset/split.json
BINARY=false            # true = benign vs malignant only (intermediate skipped)
# # ── Training ──────────────────────────────────────────────────────────────────
EPOCHS=50
PATIENCE=10
BATCH_SIZE=16
LR=0.00009447014464909024
WEIGHT_DECAY=0.01
DROPOUT=0.3
HEAD=mlp_no_meta                # linear | mlp (with age/sex meta) | mlp_no_meta
META_EMBED_DIM=32
HIDDEN_DIMS="128 64"                 # only used for mlp heads

# # ── Loss ──────────────────────────────────────────────────────────────────────
LOSS=focal
CLASS_WEIGHTING=inverse    # none | inverse | sqrt | effective
FOCAL_GAMMA=3.05351237670772
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

python $home_dir/Project/src/lace_downstream.py \
    --version                $VERSION \
    --downstream_visual_mode $VISUAL_MODE \
    --checkpoint             $CHECKPOINT \
    --splits                 $SPLITS \
    --out_dir                $home_dir/Project/results \
    --epochs                 $EPOCHS \
    --patience               $PATIENCE \
    --batch_size             $BATCH_SIZE \
    --lr                     $LR \
    --weight_decay           $WEIGHT_DECAY \
    --dropout                $DROPOUT \
    --head                   $HEAD \
    --meta_embed_dim         $META_EMBED_DIM \
    --hidden_dims            $HIDDEN_DIMS \
    --loss                   $LOSS \
    --focal_gamma            $FOCAL_GAMMA \
    --class_weighting        $CLASS_WEIGHTING \
    $( [ "$BINARY" = "true" ] && echo "--binary" ) \
    --seed                   $SEED \
    --wandb \
    --wandb_project lace-downstream \
    --wandb_entity  philipp-wiese
