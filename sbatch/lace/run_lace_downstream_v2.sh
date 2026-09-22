#!/bin/bash
#SBATCH --job-name=lacev2_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/lace/slurm-%j_lacev2_downstream.out"

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export SSL_CERT_FILE=$home_dir/miniconda3/envs/$MY_CONDA_ENV/ssl/cert.pem
export CONDA_EXE=$home_dir/miniconda3/bin/conda
source $home_dir/miniconda3/etc/profile.d/conda.sh
conda activate $MY_CONDA_ENV
echo Environment activated

export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export WANDB_DIR=$home_dir/Project/logs
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$PATH
export PYTHONPATH=$home_dir/Project/src

# ── Model ─────────────────────────────────────────────────────────────────────
# Reproduction of sweep run: cls_fg / run_20260707_164235 / split_binary.json
IMAGE_SIZE=224          # 224 = default | 512 = CheXFound-equivalent resolution
USE_MASK=true           # apply lesion-mask cropping to input images (else the full image is just resized)
VERSION=v2              # v1: CLS token | v2: MaskTokenDecoder (requires v2 pretrain ckpt)
VISUAL_MODE=cls_fg      # cls [B,512] | fg [B,512] | cls_fg [B,1024]  run_20260721_080616
CHECKPOINT=$home_dir/Project/results/lace_v2_pretrain/run_20260831_174119/best_retrieval_checkpoint.pt
SPLITS=$home_dir/Project/data/internal_dataset/split_final.json # run_20260715_101527
BINARY=false            # true = benign vs malignant only (intermediate skipped)
BTXRD_MANIFEST=$home_dir/Project/data/BTXRD/btxrd_downstream_binary.json
# # ── Training ──────────────────────────────────────────────────────────────────
EPOCHS=200
PATIENCE=10
BATCH_SIZE=64
LR=4.555143374320362e-06
WEIGHT_DECAY=0.03809353499198539
DROPOUT=0.17628065817422953
HEAD=mlp_no_meta                # linear | mlp (with age/sex meta) | mlp_no_meta
META_EMBED_DIM=0
HIDDEN_DIMS="[256, 128]"            # only used for mlp heads

# # ── Loss ──────────────────────────────────────────────────────────────────────
LOSS=focal
CLASS_WEIGHTING=sqrt  # none | inverse | sqrt | effective
FOCAL_GAMMA=2.7378671997645583
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

python $home_dir/Project/src/LACE/train/downstream.py \
    --version                $VERSION \
    --image_size             $IMAGE_SIZE \
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
    --use_mask               $USE_MASK \
    $( [ "$BINARY" = "true" ] && echo "--binary" ) \
    --seed                   $SEED \
    --wandb \
    --wandb_project lace-downstream \
    --wandb_entity  philipp-wiese \
    --eval_test 
    # --btxrd_manifest         $BTXRD_MANIFEST \
