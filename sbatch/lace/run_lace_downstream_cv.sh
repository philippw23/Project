#!/bin/bash
#SBATCH --job-name=lace_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/lace/slurm-%j_lace_downstream_cv.out"

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
IMAGE_SIZE=224
USE_MASK=true
VERSION=v2
VISUAL_MODE=cls          # cls | fg | cls_fg

# CV-mode pretrain run dir: one fold<N>/{split.json,best_retrieval_checkpoint.pt}
# per fold. Each fold's checkpoint + split are picked up together from there —
# update this to the CV pretrain run you want to evaluate.  run_20260724_221040
CV_DIR=$home_dir/Project/results/lace_v2_pretrain/run_20260728_191929
PATTERN="binary_fold*/split.json"
CHECKPOINT_FILENAME=best_retrieval_checkpoint.pt

BTXRD_MANIFEST=$home_dir/Project/data/BTXRD/btxrd_downstream_binary.json
BINARY=true

EPOCHS=100
PATIENCE=10
BATCH_SIZE=16
LR=5.373534110012835e-06
WEIGHT_DECAY=0.05
DROPOUT=0.3
HEAD=mlp_no_meta
HIDDEN_DIMS="[128, 64]"

LOSS=focal
CLASS_WEIGHTING=effective
FOCAL_GAMMA=3.224219517589033
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

# Note: --splits, --eval_test and --checkpoint are managed per fold by the orchestrator.
python $home_dir/Project/src/downstream_cv.py \
    --baseline               lace \
    --version                $VERSION \
    --image_size             $IMAGE_SIZE \
    --downstream_visual_mode $VISUAL_MODE \
    --cv_dir                 $CV_DIR \
    --pattern                "$PATTERN" \
    --checkpoint_filename    $CHECKPOINT_FILENAME \
    --out_dir                $home_dir/Project/results \
    --epochs                 $EPOCHS \
    --patience               $PATIENCE \
    --batch_size             $BATCH_SIZE \
    --lr                     $LR \
    --weight_decay           $WEIGHT_DECAY \
    --dropout                $DROPOUT \
    --head                   $HEAD \
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
    --btxrd_manifest         $BTXRD_MANIFEST \