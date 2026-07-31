#!/bin/bash
#SBATCH --job-name=imagenet_img_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_imagenet_img_downstream_cv.out"

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

# ── Fixed hyperparameters (winning sweep config) ───────────────────────────────
USE_MASK=true

# ImageNet has no checkpoint concept at all (always vanilla ViT-B/16, frozen) —
# --frozen just tells the orchestrator to skip its per-fold checkpoint lookup,
# so --cv_dir points straight at the raw fold split pool.
CV_DIR=$home_dir/Project/data/internal_dataset/cv
PATTERN="split_fold*.json"

BTXRD_MANIFEST=$home_dir/Project/data/BTXRD/btxrd_downstream_binary.json
BINARY=false

EPOCHS=100
PATIENCE=100
BATCH_SIZE=64
LR_MLP=3e-4
WEIGHT_DECAY=0.05
DROPOUT=0.3
HEAD=mlp_no_meta
HIDDEN_DIMS="64"
META_EMBED_DIM=0

LOSS=focal
CLASS_WEIGHTING=sqrt
FOCAL_GAMMA=2.0
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

# Note: --splits, --eval_test and --run_name are managed per fold by the
# orchestrator; imagenet has no --checkpoint flag at all.
python $home_dir/Project/src/downstream_cv.py \
    --baseline               imagenet \
    --frozen \
    --cv_dir                  $CV_DIR \
    --pattern                 "$PATTERN" \
    --out_dir                 $home_dir/Project/results/imagenet_img \
    --epochs                  $EPOCHS \
    --patience                $PATIENCE \
    --batch_size               $BATCH_SIZE \
    --lr_mlp                  $LR_MLP \
    --weight_decay            $WEIGHT_DECAY \
    --dropout                 $DROPOUT \
    --head                    $HEAD \
    --hidden_dims              $HIDDEN_DIMS \
    --meta_embed_dim          $META_EMBED_DIM \
    --loss                    $LOSS \
    --focal_gamma             $FOCAL_GAMMA \
    --class_weighting         $CLASS_WEIGHTING \
    $( [ "$USE_MASK" = "true" ] && echo "--use_mask" ) \
    --binary                  $BINARY \
    --seed                    $SEED \
    --wandb \
    --wandb_project imagenet-img-downstream \
    --wandb_entity  philipp-wiese \
    #--btxrd_manifest         $BTXRD_MANIFEST \
