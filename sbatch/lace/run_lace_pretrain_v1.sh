#!/bin/bash
#SBATCH --job-name=lace_v1_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_lacev1.out"

# ── Hyperparameter ────────────────────────────────────────────────────────────
LORA_LAYERS=6
LORA_R=8
LORA_ALPHA=32
EMBED_DIM=512
BATCH_SIZE=128
BTXRD_BATCH_SIZE=128
MAX_TEXT_LEN=128
TEXT_MODE=phrase
MAX_BEF_PHRASES=16
MAX_BEUR_PHRASES=16
EPOCHS=100
PATIENCE=25
WARMUP_EPOCHS=5
LR=0.000199055811638479
SCHEDULER=cosine
WEIGHT_DECAY=0.001846597080866615
LAMBDA_ITA=1.0
LAMBDA_SIM=1.0
LAMBDA_REG=0.5

TAU_S_BEUR=0.04
TAU_S_BEF=0.07
TAU_S_IMG_FULL=0.05
TAU_S_IMG_CROP=0.05

SAME_IMAGE_BOOST=10.0   # logit boost for same-image phrase pairs in I2T soft target (0.0 = original behaviour)
REWEIGHT_BY_N_PHRASES=true  # weight each phrase by 1/n_phrases_i so all images contribute equally
GLOBAL_CONTEXT_FRACTION=0.3  # context fraction for the global crop used as full_image
CONTEXT_FRACTION=0.15        # context fraction for the tight tumor crop used as crop_image

LEARN_LOSS_WEIGHTS=true   # set to true to make λ_ita/λ_sim/λ_reg learnable
WARM_START_PROJECTIONS=true  # init projection heads from pretrained BiomedCLIP weights
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

# LACE v1: L_ITA + L_sim + L_ortho from epoch 1
python $home_dir/Project/src/lace_pretrain.py \
    --splits       $home_dir/Project/data/internal_dataset/split.json \
    --btxrd_images $home_dir/Project/data/BTXRD/images \
    --btxrd_annots $home_dir/Project/data/BTXRD/Annotations \
    --out_dir      $home_dir/Project/results \
    --lora_layers  $LORA_LAYERS \
    --lora_r       $LORA_R \
    --lora_alpha   $LORA_ALPHA \
    --embed_dim    $EMBED_DIM \
    --batch_size   $BATCH_SIZE \
    --btxrd_batch_size $BTXRD_BATCH_SIZE \
    --max_text_len $MAX_TEXT_LEN \
    --text_mode    $TEXT_MODE \
    --max_bef_phrases  $MAX_BEF_PHRASES \
    --max_beur_phrases $MAX_BEUR_PHRASES \
    --epochs        $EPOCHS \
    --patience      $PATIENCE \
    --warmup_epochs $WARMUP_EPOCHS \
    --lr           $LR \
    --scheduler    $SCHEDULER \
    --weight_decay $WEIGHT_DECAY \
    --lambda_ita   $LAMBDA_ITA \
    --lambda_sim   $LAMBDA_SIM \
    --lambda_reg   $LAMBDA_REG \
    --tau_s_beur     $TAU_S_BEUR \
    --tau_s_bef      $TAU_S_BEF \
    --tau_s_img_full $TAU_S_IMG_FULL \
    --tau_s_img_crop $TAU_S_IMG_CROP \
    --same_image_boost $SAME_IMAGE_BOOST \
    --global_context_fraction $GLOBAL_CONTEXT_FRACTION \
    --context_fraction $CONTEXT_FRACTION \
    $( [ "$REWEIGHT_BY_N_PHRASES" = "true" ] && echo "--reweight_by_n_phrases" ) \
    $( [ "$LEARN_LOSS_WEIGHTS" = "true" ] && echo "--learn_loss_weights" ) \
    $( [ "$WARM_START_PROJECTIONS" = "true" ] && echo "--warm_start_projections" ) \
    --seed         $SEED \
    --wandb \
    --wandb_project lace-pretrain \
    --wandb_entity  philipp-wiese
