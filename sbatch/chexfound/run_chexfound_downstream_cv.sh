#!/bin/bash
#SBATCH --job-name=chexfound_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/chexfound/slurm-%j_chexfound_downstream_cv.out"

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

# ── Fixed hyperparameters (fill in the winning sweep config) ──────────────────
IMAGE_SIZE=512   # 512 = native CheXFound | 224 = BiomedCLIP-equivalent resolution
USE_MASK=true

# Set CHECKPOINT to a continued-pretrain .pth path, or "none" to use frozen
# original weights (--chexfound_weights). There is currently no CV-mode
# continued-pretraining for CheXFound (no per-fold checkpoints), so FROZEN
# should stay true — it tells the orchestrator to skip its per-fold checkpoint
# lookup and lets --cv_dir point straight at the raw fold split pool below.
CHECKPOINT="none"
CHEXFOUND_WEIGHTS=$home_dir/Project/src/chexfound/data/teacher_checkpoint.pth
FROZEN=true

# Raw fold split pool (not a pretrain run's fold<N>/ dirs — there's nothing
# pretrained to pick up per fold here). Pick the pool matching BINARY below:
# cv/ (3-class) or cv_binary/ (binary) — both are named split_binary_fold*.json
# regardless of which classes they actually contain.
CV_DIR=$home_dir/Project/data/internal_dataset/cv_binary
PATTERN="split_binary_fold*.json"

BTXRD_MANIFEST=$home_dir/Project/data/BTXRD/btxrd_downstream_binary.json
BINARY=true

EPOCHS=50
PATIENCE=10
BATCH_SIZE=16
LR=0.0003946323640213164
WEIGHT_DECAY=0.1
DROPOUT=0.4
HEAD=mlp_no_meta
HIDDEN_DIMS="64"
META_EMBED_DIM=16

LOSS=cb_focal
CLASS_WEIGHTING=inverse
FOCAL_GAMMA=4.0
CB_BETA=0.99
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

# Note: --splits, --eval_test and --run_name are managed per fold by the
# orchestrator; --checkpoint is only reserved when not --frozen.
python $home_dir/Project/src/downstream_cv.py \
    --baseline               chexfound \
    $( [ "$FROZEN" = "true" ] && echo "--frozen" ) \
    --checkpoint              $CHECKPOINT \
    --chexfound_weights       $CHEXFOUND_WEIGHTS \
    --image_size              $IMAGE_SIZE \
    --cv_dir                  $CV_DIR \
    --pattern                 "$PATTERN" \
    --out_dir                 $home_dir/Project/results \
    --epochs                  $EPOCHS \
    --patience                $PATIENCE \
    --batch_size              $BATCH_SIZE \
    --lr                      $LR \
    --weight_decay            $WEIGHT_DECAY \
    --dropout                 $DROPOUT \
    --head                    $HEAD \
    --hidden_dims             $HIDDEN_DIMS \
    --meta_embed_dim          $META_EMBED_DIM \
    --loss                    $LOSS \
    --focal_gamma             $FOCAL_GAMMA \
    --cb_beta                 $CB_BETA \
    --class_weighting         $CLASS_WEIGHTING \
    $( [ "$USE_MASK" = "true" ] && echo "--use_mask" ) \
    --binary                  $BINARY \
    --seed                    $SEED \
    --wandb \
    --wandb_project chexfound-downstream \
    --wandb_entity  philipp-wiese \
    --sweep \
    --btxrd_manifest         $BTXRD_MANIFEST \
