#!/bin/bash
#SBATCH --job-name=biomedclip_gloria_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Note: bump --time above when CV_DIR is set below — CV runs N folds back to
# back in a single job, so it needs roughly N x a normal single-run time budget.

# ── Cross-validation ─────────────────────────────────────────────────────────
# Set CV_DIR to run one full pretraining pass per fold file instead of a single
# run (results land under run_<name>/fold0/, fold1/, ...). Leave empty for a
# normal single-split run using SPLITS below. Mutually exclusive with SPLITS.
home_dir="/mnt/nfs/homedirs/$USER"
CV_DIR=""                                                          # e.g. $home_dir/Project/data/internal_dataset/cv
CV_PATTERN="split_fold*.json"                                      # glob for fold files inside CV_DIR
SPLITS=$home_dir/Project/data/internal_dataset/split_final.json    # ignored when CV_DIR is set

# ── LoRA / training parameters — same defaults as run_biomedclip_pretrain_3class.sh
# for apples-to-apples comparison against the plain BiomedCLIP baseline ────────
BATCH_SIZE=32
LORA_LAYERS=4
LORA_R=8
LR_LORA=1e-4
LR_PROJ=5e-4
WEIGHT_DECAY=0.2
EPOCHS=50

# ── GLoRIA loss hyperparameters (same values as sbatch/gloria/run_gloria_pretrain.sh) ──
TEMP1=5
TEMP2=4
TEMP3=15
LOCAL_LOSS_WEIGHT=1
GLOBAL_LOSS_WEIGHT=1.5
# ─────────────────────────────────────────────────────────────────────────────

export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/biomedclip_gloria/slurm-${SLURM_JOBID}_bs${BATCH_SIZE}_lora${LORA_LAYERS}.out"
mkdir -p "$(dirname "$LOG_FILE")"
exec > "$LOG_FILE" 2>&1

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

python $home_dir/Project/src/biomedclip_gloria_pretrain.py \
    $( [ -z "$CV_DIR" ] && echo "--splits $SPLITS" ) \
    $( [ -n "$CV_DIR" ] && echo "--cv_dir $CV_DIR" ) \
    $( [ -n "$CV_DIR" ] && [ -n "$CV_PATTERN" ] && echo "--cv_pattern $CV_PATTERN" ) \
    --out_dir  $home_dir/Project/results \
    --use_mask \
    --lora_layers $LORA_LAYERS \
    --lora_r $LORA_R \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr_lora $LR_LORA \
    --lr_proj $LR_PROJ \
    --weight_decay $WEIGHT_DECAY \
    --temp1 $TEMP1 \
    --temp2 $TEMP2 \
    --temp3 $TEMP3 \
    --local_loss_weight  $LOCAL_LOSS_WEIGHT \
    --global_loss_weight $GLOBAL_LOSS_WEIGHT \
    --seed 42 \
    --wandb \
    --wandb_project biomedclip-gloria-pretrain \
    --wandb_entity philipp-wiese
