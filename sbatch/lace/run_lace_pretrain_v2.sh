#!/bin/bash
#SBATCH --job-name=lace_v2_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_lacev2.out"

# ── Hyperparameter ────────────────────────────────────────────────────────────
LORA_LAYERS=4
LORA_R=8
LORA_ALPHA=16
EMBED_DIM=512
N_MASK_TOKENS=16
TAU_SPATIAL=0.1
BATCH_SIZE=32
BTXRD_BATCH_SIZE=16
MAX_TEXT_LEN=128
TEXT_MODE=phrase
MAX_BEF_PHRASES=16
MAX_BEUR_PHRASES=16
STAGE1_EPOCHS=10
STAGE2_EPOCHS=15
STAGE3_EPOCHS=15
PATIENCE=20
LR=5e-5
WEIGHT_DECAY=0.2
LAMBDA_SEG=1.0
LAMBDA_SIM=1.0
WARM_START_PROJECTIONS=true  # init projection heads from pretrained BiomedCLIP weights
REWEIGHT_BY_N_PHRASES=true  # weight each phrase by 1/n_phrases_i so all images contribute equally
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

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

# LACE v2: L_ITA (stage 1) + L_seg (stage 2) + L_sim v2 with phrases (stage 3)
python $home_dir/Project/src/lace_pretrain_v2.py \
    --splits       $home_dir/Project/data/internal_dataset/split.json \
    --btxrd_images $home_dir/Project/data/BTXRD/images \
    --btxrd_annots $home_dir/Project/data/BTXRD/Annotations \
    --out_dir      $home_dir/Project/results \
    --lora_layers  $LORA_LAYERS \
    --lora_r       $LORA_R \
    --lora_alpha   $LORA_ALPHA \
    --embed_dim    $EMBED_DIM \
    --n_mask_tokens $N_MASK_TOKENS \
    --tau_spatial   $TAU_SPATIAL \
    --batch_size   $BATCH_SIZE \
    --btxrd_batch_size $BTXRD_BATCH_SIZE \
    --max_text_len $MAX_TEXT_LEN \
    --text_mode    $TEXT_MODE \
    --max_bef_phrases  $MAX_BEF_PHRASES \
    --max_beur_phrases $MAX_BEUR_PHRASES \
    --stage1_epochs $STAGE1_EPOCHS \
    --stage2_epochs $STAGE2_EPOCHS \
    --stage3_epochs $STAGE3_EPOCHS \
    --patience     $PATIENCE \
    --lr           $LR \
    --weight_decay $WEIGHT_DECAY \
    --lambda_seg   $LAMBDA_SEG \
    --lambda_sim   $LAMBDA_SIM \
    $( [ "$REWEIGHT_BY_N_PHRASES" = "true" ] && echo "--reweight_by_n_phrases" ) \
    $( [ "$WARM_START_PROJECTIONS" = "true" ] && echo "--warm_start_projections" ) \
    --seed         $SEED \
    --wandb \
    --wandb_project lace-v2-pretrain \
    --wandb_entity  philipp-wiese
