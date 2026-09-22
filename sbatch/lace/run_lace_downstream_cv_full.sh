#!/bin/bash
#SBATCH --job-name=lace_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/lace/slurm-%j_lace_downstream_cv_A6_3class_auroc_dry-sweep-5.out"

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
VISUAL_MODE=cls

# CV-mode pretrain run dir: one fold<N>/{split.json,best_retrieval_checkpoint.pt}
# per fold. Each fold's checkpoint + split are picked up together from there —
# update this to the CV pretrain run you want to evaluate.  run_20260819_125955
CV_DIR=$home_dir/Project/results/lace_v2_pretrain/run_20260901_093211
PATTERN="fold*/split.json"
CHECKPOINT_FILENAME=best_checkpoint.pt #best_retrieval_checkpoint.pt

BINARY=false

EPOCHS=200
PATIENCE=20
BATCH_SIZE=64
LR=2.186767697250412e-05
WEIGHT_DECAY=0.07222815989209905
DROPOUT=0.18959282563735488
HEAD=mlp_no_meta
HIDDEN_DIMS="[256, 128]"

LOSS=focal
CLASS_WEIGHTING=inverse
FOCAL_GAMMA=3.136278999868267
SEED=42
EARLY_STOPPING_METRIC="val_bal_acc"
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
    --early_stopping_metric  $EARLY_STOPPING_METRIC \
    --wandb \
    --wandb_project lace-downstream \
    --wandb_entity  philipp-wiese \