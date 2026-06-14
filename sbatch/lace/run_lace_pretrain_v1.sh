#!/bin/bash
#SBATCH --job-name=lace_v1_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_lacev1.out"

# ── Hyperparameter ────────────────────────────────────────────────────────────
LORA_LAYERS=4
LORA_R=8
LORA_ALPHA=16
EMBED_DIM=512
BATCH_SIZE=128
BTXRD_BATCH_SIZE=128
MAX_TEXT_LEN=256       # max number of tokens for text encoder input (after tokenization) for full and concatenated text inputs; phrase inputs are not truncated but limited by max_bef_phrases/max_beur_phrases
TEXT_MODE=mixed        # full, concat, phrase, mixed (full + phrase)
MAX_BEF_PHRASES=16
MAX_BEUR_PHRASES=16
EPOCHS=100
PATIENCE=25
WARMUP_EPOCHS=5
LR=5e-5
SCHEDULER=cosine
WEIGHT_DECAY=0.001846597080866615
LAMBDA_ITA=1.0
LAMBDA_SIM=1.0
LAMBDA_ORTHO=1.0

TAU_S_BEUR=0.04
TAU_S_BEF=0.04
TAU_S_IMG_FULL=0.04
TAU2=0.03         # attention softmax temperature for GLoRIA-style patch attention
LAMBDA_T2I=1.0     # weight for T2I hard InfoNCE in L_sim

SAME_IMAGE_BOOST=20.0   # logit boost for same-image phrase pairs in I2T soft target (0.0 = original behaviour)
REWEIGHT_BY_N_PHRASES=true  # weight each phrase by 1/n_phrases_i so all images contribute equally
T2I_MODE=text_text   # image_image | text_text | descriptor | infonce
GLOBAL_CONTEXT_FRACTION=0.15  # context fraction for the global crop used as full_image
CONTEXT_FRACTION=0.15        # context fraction for the tight tumor crop used as crop_image

LEARN_LOSS_WEIGHTS=true   # set to true to make λ_ita/λ_sim/λ_ortho learnable
WARM_START_PROJECTIONS=true  # init projection heads from pretrained BiomedCLIP weights
USE_BTXRD=false               # set to false to disable BTXRD ortho dataset

LOSSES="ita sim ortho"       # active loss terms: any subset of ita sim ortho
SIM_LESION_ONLY=true         # restrict L_sim phrase-patch attention to lesion patches
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
export PYTORCH_NVML_BASED_CUDA_CHECK=0

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
    --lambda_ortho $LAMBDA_ORTHO \
    --tau_s_beur     $TAU_S_BEUR \
    --tau_s_bef      $TAU_S_BEF \
    --tau_s_img_full $TAU_S_IMG_FULL \
    --tau2           $TAU2 \
    --lambda_t2i     $LAMBDA_T2I \
    --same_image_boost $SAME_IMAGE_BOOST \
    --global_context_fraction $GLOBAL_CONTEXT_FRACTION \
    --context_fraction $CONTEXT_FRACTION \
    $( [ "$REWEIGHT_BY_N_PHRASES" = "true" ] && echo "--reweight_by_n_phrases" ) \
    $( [ "$SIM_LESION_ONLY" = "true" ] && echo "--sim_lesion_only" ) \
    --t2i_mode $T2I_MODE \
    --descriptor_vectors $home_dir/Project/data/internal_dataset/text/descriptor_vectors.json \
    $( [ "$LEARN_LOSS_WEIGHTS" = "true" ] && echo "--learn_loss_weights" ) \
    $( [ "$WARM_START_PROJECTIONS" = "true" ] && echo "--warm_start_projections" ) \
    $( [ "$USE_BTXRD" = "false" ] && echo "--no_btxrd" ) \
    --losses $LOSSES \
    --seed         $SEED \
    --wandb \
    --wandb_project lace-pretrain \
    --wandb_entity  philipp-wiese
