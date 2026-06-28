#!/bin/bash
#SBATCH --job-name=lace_v2_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_lacev2.out"

# ── Architecture ──────────────────────────────────────────────────────────────
LORA_LAYERS=2
LORA_R=8
LORA_ALPHA=16
EMBED_DIM=512
N_MASK_TOKENS=4
N_MASK_HEADS=16
GAUSS_SIGMA=4.0
MASK_HEAD_TAU=0.07
SIM_ATTN_TAU=0.07
WARM_START_PROJECTIONS=true

# ── Data ──────────────────────────────────────────────────────────────────────
BATCH_SIZE=128
BTXRD_BATCH_SIZE=128
NO_BTXRD=false             # set to true to disable BTXRD auxiliary dataset
OVERFIT_N=                 # set to e.g. 10 to run overfit sanity-check (empty = disabled)
MAX_TEXT_LEN=128
TEXT_MODE=mixed            # full | phrase | mixed (full beur + bef phrases)
MAX_BEUR_TEXT_LEN=256      # tokenisation length for full beurteilung text (mixed mode)
MAX_BEF_PHRASES=16
MAX_BEUR_PHRASES=16
CONTEXT_FRACTION=-1.0      # -1.0 = full image | 0.0 = tight bbox crop | >0 = crop with context margin
# Standard is 0.15
# ── Curriculum ────────────────────────────────────────────────────────────────
# overfit: STAGE1_EPOCHS=500 EPOCHS=500
STAGE1_EPOCHS=1           # stage 1: L_ITA + L_dice; warmup = stage1_epochs // 5 = 10
EPOCHS=1                 # total (stage1 + stage2); stage 2 warmup = (epochs - stage1_epochs) // 5 = 16
PATIENCE=10

# ── Optimisation ──────────────────────────────────────────────────────────────
# overfit: LR=1e-3 SCHEDULER=constant
LR=8e-5
SCHEDULER=cosine
WEIGHT_DECAY=0.001

# ── Loss weights ──────────────────────────────────────────────────────────────
LOSSES="ita sim ortho dice" #ortho
LEARN_LOSS_WEIGHTS=true
LAMBDA_SIM=1.0           # only used when LEARN_LOSS_WEIGHTS=false
LAMBDA_ORTHO=1.0
LAMBDA_DICE=1.0

# ── Soft-target / t2i ────────────────────────────────────────────────────────
TAU_S_BEUR=0.04
TAU_S_BEF=0.04
TAU_S_IMG_FULL=0.04
LAMBDA_T2I=1.0
SAME_IMAGE_BOOST=20.0
REWEIGHT_BY_N_PHRASES=true
T2I_MODE=text_text

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

python $home_dir/Project/src/lace_pretrain_v2.py \
    --splits            $home_dir/Project/data/internal_dataset/split.json \
    --btxrd_images      $home_dir/Project/data/BTXRD/images \
    --btxrd_annots      $home_dir/Project/data/BTXRD/Annotations \
    --out_dir           $home_dir/Project/results \
    --lora_layers       $LORA_LAYERS \
    --lora_r            $LORA_R \
    --lora_alpha        $LORA_ALPHA \
    --embed_dim         $EMBED_DIM \
    --n_mask_tokens     $N_MASK_TOKENS \
    --n_mask_heads      $N_MASK_HEADS \
    --gauss_sigma       $GAUSS_SIGMA \
    --mask_head_tau     $MASK_HEAD_TAU \
    --sim_attn_tau      $SIM_ATTN_TAU \
    --batch_size        $BATCH_SIZE \
    --btxrd_batch_size  $BTXRD_BATCH_SIZE \
    --max_text_len      $MAX_TEXT_LEN \
    --max_beur_text_len $MAX_BEUR_TEXT_LEN \
    --text_mode         $TEXT_MODE \
    --max_bef_phrases   $MAX_BEF_PHRASES \
    --max_beur_phrases  $MAX_BEUR_PHRASES \
    --context_fraction  $CONTEXT_FRACTION \
    --stage1_epochs     $STAGE1_EPOCHS \
    --epochs            $EPOCHS \
    --patience          $PATIENCE \
    --lr                $LR \
    --scheduler         $SCHEDULER \
    --weight_decay      $WEIGHT_DECAY \
    --losses            $LOSSES \
    --lambda_sim        $LAMBDA_SIM \
    --lambda_ortho      $LAMBDA_ORTHO \
    --lambda_dice       $LAMBDA_DICE \
    --tau_s_beur        $TAU_S_BEUR \
    --tau_s_bef         $TAU_S_BEF \
    --tau_s_img_full    $TAU_S_IMG_FULL \
    --lambda_t2i        $LAMBDA_T2I \
    --same_image_boost  $SAME_IMAGE_BOOST \
    --t2i_mode          $T2I_MODE \
    $( [ "$LEARN_LOSS_WEIGHTS"     = "true" ] && echo "--learn_loss_weights" ) \
    $( [ "$REWEIGHT_BY_N_PHRASES"  = "true" ] && echo "--reweight_by_n_phrases" ) \
    $( [ "$WARM_START_PROJECTIONS" = "true" ] && echo "--warm_start_projections" ) \
    $( [ "$NO_BTXRD"               = "true" ] && echo "--no_btxrd" ) \
    $( [ -n "$OVERFIT_N" ] && echo "--overfit_n $OVERFIT_N" ) \
    --seed              $SEED \
    --wandb \
    --wandb_project lace-v2-pretrain \
    --wandb_entity  philipp-wiese
