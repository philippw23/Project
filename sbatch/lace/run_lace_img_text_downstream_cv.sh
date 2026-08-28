#!/bin/bash
#SBATCH --job-name=lace_img_text_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/lace/slurm-%j_lace_img_text_downstream_cv.out"

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
USE_MASK=true
VISUAL_MODE=cls_fg   # cls | fg | cls_fg — determines embed_dim (concatenated with text)

# CV-mode pretrain run dir: one fold<N>/{split.json,best_retrieval_checkpoint.pt}
# per fold — both the frozen visual backbone AND the frozen text encoder are
# loaded from this same LACE v2 checkpoint. Update to the CV pretrain run to evaluate.
CV_DIR=$home_dir/Project/results/lace_v2_pretrain/run_20260808_232942
PATTERN="fold*/split.json"
CHECKPOINT_FILENAME=best_retrieval_checkpoint.pt

# No BTXRD evaluation for this baseline — BTXRD samples carry no report text,
# so there is nothing for the text-encoder pathway to embed on that dataset
# (mirrors biomedclip_img_text_downstream_cv*.sh).
BINARY=true

EPOCHS=200
PATIENCE=10
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=0.1
DROPOUT=0.2
HEAD=mlp_no_meta
HIDDEN_DIMS="[256, 128]"

LOSS=focal
CLASS_WEIGHTING=sqrt
FOCAL_GAMMA=2.5
CB_BETA=0.99
SEED=42
EARLY_STOPPING_METRIC="val_bal_acc"
# ─────────────────────────────────────────────────────────────────────────────

# Note: --splits, --eval_test, --run_name and --checkpoint are managed per
# fold by the orchestrator.
python $home_dir/Project/src/downstream_cv.py \
    --baseline               lace_img_text \
    --downstream_visual_mode $VISUAL_MODE \
    --cv_dir                 $CV_DIR \
    --pattern                "$PATTERN" \
    --checkpoint_filename    $CHECKPOINT_FILENAME \
    --out_dir                $home_dir/Project/results \
    --epochs                 $EPOCHS \
    --patience               $PATIENCE \
    --batch_size              $BATCH_SIZE \
    --lr                     $LR \
    --weight_decay           $WEIGHT_DECAY \
    --dropout                $DROPOUT \
    --head                   $HEAD \
    --hidden_dims             $HIDDEN_DIMS \
    --loss                   $LOSS \
    --focal_gamma            $FOCAL_GAMMA \
    --cb_beta                $CB_BETA \
    --class_weighting        $CLASS_WEIGHTING \
    --use_mask                $USE_MASK \
    --binary                  $BINARY \
    --seed                   $SEED \
    --early_stopping_metric  $EARLY_STOPPING_METRIC \
    --wandb \
    --wandb_project lace-img-text-downstream \
    --wandb_entity  philipp-wiese \
    --sweep \
