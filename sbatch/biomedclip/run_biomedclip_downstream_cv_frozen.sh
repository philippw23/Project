#!/bin/bash
#SBATCH --job-name=biomedclip_downstream_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/biomedclip/slurm-%j_biomedclip_downstream_cv_3class_frozen.out"

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
FREEZED_BIOMEDCLIP=true   # true = vanilla BiomedCLIP weights, no checkpoint / LoRA

# Frozen encoder — no pretrain checkpoint to pick up, so --cv_dir points straight
# at the raw fold split pool instead of a pretrain run's fold<N>/ dirs. 3-class
# ("full") CV pool — no BTXRD comparison (BTXRD is binary-labeled only).
CV_DIR=$home_dir/Project/data/internal_dataset/cv
PATTERN="split_fold*.json"
CHECKPOINT_FILENAME=best_r1_checkpoint.pt   # unused in --frozen mode

BINARY=false

EPOCHS=50
#EPOCHS=100
PATIENCE=20
BATCH_SIZE=16
#BATCH_SIZE=64
LR=4.973766440357406e-05
#LR=3.1805704415443494e-05
WEIGHT_DECAY=0.01
#WEIGHT_DECAY=0.1957370680853576
DROPOUT=0.5
#DROPOUT=0.17191021624718292
HEAD=mlp_no_meta
HIDDEN_DIMS="[32]"
#HIDDEN_DIMS="[512, 256]"
META_EMBED_DIM=16

LOSS=focal
CLASS_WEIGHTING=inverse
FOCAL_GAMMA=3.441216155058626
#FOCAL_GAMMA=2.949553925848543
CB_BETA=0.99
SEED=42
EARLY_STOPPING_METRIC="val_bal_acc"

FINETUNE_LORA_LAYERS=0
LR_ENCODER=1e-5
FINETUNE_LORA_R=8
# ─────────────────────────────────────────────────────────────────────────────

# Note: --splits, --eval_test and --checkpoint are managed per fold by the orchestrator.
python $home_dir/Project/src/downstream_cv.py \
    --baseline               biomedclip \
    $( [ "$FREEZED_BIOMEDCLIP" = "true" ] && echo "--frozen" ) \
    --image_size              $IMAGE_SIZE \
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
    --hidden_dims             $HIDDEN_DIMS \
    --meta_embed_dim          $META_EMBED_DIM \
    --loss                    $LOSS \
    --focal_gamma             $FOCAL_GAMMA \
    --cb_beta                 $CB_BETA \
    --class_weighting         $CLASS_WEIGHTING \
    --finetune_lora_layers    $FINETUNE_LORA_LAYERS \
    --lr_encoder              $LR_ENCODER \
    --finetune_lora_r         $FINETUNE_LORA_R \
    $( [ "$USE_MASK" = "true" ] && echo "--use_mask" ) \
    $( [ "$FREEZED_BIOMEDCLIP" = "true" ] && echo "--freezed_biomedclip" ) \
    --binary                  $BINARY \
    --seed                    $SEED \
    --early_stopping_metric   $EARLY_STOPPING_METRIC \
    --wandb \
    --wandb_project biomedclip-downstream \
    --wandb_entity  philipp-wiese \
