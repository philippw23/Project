#!/bin/bash
#SBATCH --job-name=chexfound_downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

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
export CHEXFOUND_ROOT=$home_dir/CheXFound

# ── Set one of the two modes below ───────────────────────────────────────────
#
# Mode A: frozen CheXFound baseline (no continued pretraining)
#   Remove --checkpoint entirely.
#
# Mode B: after continued iBOT pretraining
#   Set PRETRAIN_CKPT to the checkpoint saved by run_chexfound_pretrain.sh,
#   e.g. results/chexfound_pretrain/model_final.pth

PRETRAIN_CKPT=""   # leave empty for frozen baseline, or set path for Mode B

CHECKPOINT_ARG=""
if [ -n "$PRETRAIN_CKPT" ]; then
    CHECKPOINT_ARG="--checkpoint $PRETRAIN_CKPT"
fi

python $home_dir/Project/src/chexfound_downstream.py \
    --chexfound_config  $home_dir/Project/configs/chexfound_vitl16_bonetumor.yaml \
    --chexfound_weights $home_dir/CheXFound/weights/chexfound_vitl16.pth \
    $CHECKPOINT_ARG \
    --splits    $home_dir/Project/results/biomedclip_pretrain/splits.json \
    --excel     $home_dir/Project/data/metadata.xlsx \
    --out_dir   $home_dir/Project/results \
    --use_mask \
    --epochs    50 \
    --batch_size 64 \
    --lr        1e-3 \
    --dropout   0.3 \
    --hidden_dims 256 128 \
    --meta_embed_dim 16 \
    --weight_decay 0.01 \
    --loss      ce \
    --class_weighting sqrt \
    --seed      42 \
    --wandb \
    --wandb_project chexfound-downstream \
    --wandb_entity  philipp-wiese
