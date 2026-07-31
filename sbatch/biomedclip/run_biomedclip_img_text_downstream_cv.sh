#!/bin/bash
#SBATCH --job-name=biomedclip_img_text_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/biomedclip/slurm-%j_biomedclip_img_text_downstream_cv.out"

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
# No --image_size here: biomedclip_img_text_downstream.py always uses open_clip's
# fixed 224×224 preprocessing (unlike biomedclip_downstream.py).
USE_MASK=true

# Non-frozen: reuses the per-fold LoRA checkpoints from a biomedclip CV
# pretraining run (sbatch/biomedclip/run_biomedclip_pretrain.sh with CV_DIR
# set), which writes fold<N>/{split.json,best_r1_checkpoint.pt} under the run
# dir. Point CV_DIR at that pretrain run; --checkpoint is injected per fold by
# the orchestrator next to each fold's split file.
CV_DIR=$home_dir/Project/results/biomedclip_pretrain/run_.../
PATTERN="fold*/split.json"
CHECKPOINT_FILENAME=best_r1_checkpoint.pt

# No BTXRD evaluation for this baseline — BTXRD samples carry no report text,
# so there is nothing for the text-encoder pathway to embed on that dataset.
BINARY=true

EPOCHS=50
PATIENCE=10
BATCH_SIZE=64
LR=1e-3
WEIGHT_DECAY=0.01
DROPOUT=0.3
HEAD=mlp_no_meta
HIDDEN_DIMS="256 128"
META_EMBED_DIM=16

LOSS=focal
CLASS_WEIGHTING=sqrt
FOCAL_GAMMA=2.0
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

# Note: --splits, --eval_test, --run_name and --checkpoint are managed per
# fold by the orchestrator.
python $home_dir/Project/src/downstream_cv.py \
    --baseline               biomedclip_img_text \
    --cv_dir                  $CV_DIR \
    --pattern                 "$PATTERN" \
    --checkpoint_filename     $CHECKPOINT_FILENAME \
    --out_dir                 $home_dir/Project/results \
    --epochs                  $EPOCHS \
    --patience                $PATIENCE \
    --batch_size               $BATCH_SIZE \
    --lr                      $LR \
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
    --wandb_project biomedclip-img-text-downstream \
    --wandb_entity  philipp-wiese \
    --sweep \
