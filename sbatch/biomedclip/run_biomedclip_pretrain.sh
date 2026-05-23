#!/bin/bash
#SBATCH --job-name=biomedclip_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=all_nodes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ── Training parameters (edit here) ──────────────────────────────────────────
BATCH_SIZE=128
NO_LORA=false       # true → unfreeze blocks, false → LoRA
LORA_LAYERS=4
LORA_R=32
UNFREEZE_BLOCKS=8
LR=1e-5 # 5e-5 
# ─────────────────────────────────────────────────────────────────────────────

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

if [ "$NO_LORA" = true ]; then
    tune_tag="unfreeze${UNFREEZE_BLOCKS}"
else
    tune_tag="lora${LORA_LAYERS}"
fi

LOG_FILE="$home_dir/Project/logs/slurm-${SLURM_JOBID}_bs${BATCH_SIZE}_${tune_tag}_lr${LR}.out"
exec > "$LOG_FILE" 2>&1

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

if [ "$NO_LORA" = true ]; then
    python $home_dir/Project/src/biomedclip_pretrain.py \
        --splits   $home_dir/Project/data/internal_dataset/split.json \
        --out_dir  $home_dir/Project/results \
        --use_mask \
        --no_lora \
        --unfreeze_blocks $UNFREEZE_BLOCKS \
        --batch_size $BATCH_SIZE \
        --epochs 100 \
        --lr $LR \
        --seed 42 \
        --wandb \
        --wandb_project biomedclip-pretrain \
        --wandb_entity philipp-wiese
else
    python $home_dir/Project/src/biomedclip_pretrain.py \
        --splits   $home_dir/Project/data/internal_dataset/split.json \
        --out_dir  $home_dir/Project/results \
        --use_mask \
        --lora_layers $LORA_LAYERS \
        --lora_r $LORA_R \
        --batch_size $BATCH_SIZE \
        --epochs 100 \
        --lr $LR \
        --seed 42 \
        --wandb \
        --wandb_project biomedclip-pretrain \
        --wandb_entity philipp-wiese
fi
