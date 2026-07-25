#!/bin/bash
#SBATCH --job-name=lace_v2_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_lacev2.out"
# Note: bump --time above when CV_DIR is set below — CV runs N folds back to
# back in a single job, so it needs roughly N x a normal single-run time budget.

# ── Cross-validation ────────────────────────────────────────────────────────────
# Set CV_DIR to run one full pretraining pass per fold file instead of a single
# run (results land under run_<timestamp>/fold0/, fold1/, ...). Leave empty for
# a normal single-split run using SPLITS below. Mutually exclusive with SPLITS.
home_dir="/mnt/nfs/homedirs/$USER"
CV_DIR=$home_dir/Project/data/internal_dataset/cv_binary # e.g. $home_dir/Project/data/internal_dataset/cv_binary
CV_PATTERN="split_binary_fold*.json"                # glob for fold files inside CV_DIR (empty = script default "split_binary_fold*.json")
SPLITS=$home_dir/Project/data/internal_dataset/split_binary_final.json   # ignored when CV_DIR is set

# ── Image encoder ─────────────────────────────────────────────────────────────
IMAGE_ENCODER=biomedclip   # biomedclip | chexfound

# ── Architecture ──────────────────────────────────────────────────────────────
LORA_LAYERS=6
LORA_R=8
LORA_ALPHA=32
EMBED_DIM=512
UNFREEZE_LAYERS=4          # 0 = use LoRA | >0 = full fine-tune last N ViT blocks (overrides LoRA, biomedclip only)
N_MASK_TOKENS=1
N_MASK_HEADS=16
GAUSS_SIGMA=4.5
MASK_HEAD_TAU=0.03 # shared soft-assignment temp for mask heads; sweep {0.01, 0.03, 0.05}
SIM_ATTN_TAU=0.05 # shared soft-assignment temp for similarity attention; sweep {0.01, 0.05, 0.1}
WARM_START_PROJECTIONS=true

# ── Evidence prototype space (LGDEA) — used when "evid" is in LOSSES ──────────
# See src/LACE/EVIDENCE_PROTOTYPE_PLAN.md. 'evid' is an alternative to 'sim';
# L_rec warms up prototypes in stage 1, L_evid_p activates in stage 2.
N_PROTOTYPES=32            # K; sweep {16, 32, 64}
TAU_PROTO=0.1             # shared soft-assignment temp; sweep {0.05, 0.1, 0.2}
LAMBDA_MU=0.01            # prototype-norm shrinkage weight in L_rec

# ── Data ──────────────────────────────────────────────────────────────────────
BATCH_SIZE=128
BTXRD_BATCH_SIZE=128
USE_BTXRD=false            # set to true to enable BTXRD auxiliary dataset
OVERFIT_N=                 # set to e.g. 10 to run overfit sanity-check (empty = disabled)
MAX_TEXT_LEN=128
TEXT_MODE=mixed            # full | phrase | mixed (full beur + bef phrases)
MAX_BEUR_TEXT_LEN=256      # tokenisation length for full beurteilung text (mixed mode)
MAX_BEF_PHRASES=16
MAX_BEUR_PHRASES=16
CONTEXT_FRACTION=0.15      # -1.0 = full image | 0.0 = tight bbox crop | >0 = crop with context margin
# Standard is 0.15
# ── Curriculum ────────────────────────────────────────────────────────────────
# overfit: STAGE1_EPOCHS=500 EPOCHS=500
STAGE1_EPOCHS=16           # stage 1: L_ITA + L_dice; warmup = stage1_epochs // 5 = 10
EPOCHS=120                 # total (stage1 + stage2); stage 2 warmup = (epochs - stage1_epochs) // 5 = 16
PATIENCE=20

# ── Optimisation ──────────────────────────────────────────────────────────────
# overfit: LR=1e-3 SCHEDULER=constant
LR=1.672772947691402e-05
SCHEDULER=cosine
WEIGHT_DECAY=0.0010489285940590723

# ── Loss weights ──────────────────────────────────────────────────────────────
#LOSSES="ita evid ortho dice" # swap "sim" -> "evid" for the LGDEA prototype variant
# Optional per-loss stage control; OVERRIDES $LOSSES when non-empty. Space-separated
# NAME:STAGES tokens over ita/sim/ortho/dice/rec/evid, stages from {1,2} or 0/none.
# e.g. LOSS_STAGES="dice:1,2 ortho:1,2 rec:1 ita:2 evid:2 sim:none"
LOSS_STAGES="dice:1,2 ortho:1,2 ita:2 sim:2 rec:none evid:none"
LEARN_LOSS_WEIGHTS=true
LAMBDA_SIM=1.0           # only used when LEARN_LOSS_WEIGHTS=false
LAMBDA_ORTHO=1.0
LAMBDA_DICE=1.0
LAMBDA_REC=1.0
LAMBDA_EVID=1.0

# ── Soft-target / t2i ────────────────────────────────────────────────────────
TAU_S_BEUR=0.03
TAU_S_BEF=0.045
TAU_S_IMG_FULL=0.03
LAMBDA_T2I=0.5
SAME_IMAGE_BOOST=20.0
REWEIGHT_BY_N_PHRASES=true
T2I_MODE=image_image  #image_image | "text_text" | "descriptor" | "infonce"

SEED=42
# ─────────────────────────────────────────────────────────────────────────────


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
    --image_encoder     $IMAGE_ENCODER \
    $( [ -z "$CV_DIR" ] && echo "--splits $SPLITS" ) \
    $( [ -n "$CV_DIR" ] && echo "--cv_dir $CV_DIR" ) \
    $( [ -n "$CV_DIR" ] && [ -n "$CV_PATTERN" ] && echo "--cv_pattern $CV_PATTERN" ) \
    --btxrd_images      $home_dir/Project/data/BTXRD/images \
    --btxrd_annots      $home_dir/Project/data/BTXRD/Annotations \
    --out_dir           $home_dir/Project/results \
    --lora_layers       $LORA_LAYERS \
    --lora_r            $LORA_R \
    --lora_alpha        $LORA_ALPHA \
    --embed_dim         $EMBED_DIM \
    --unfreeze_layers   $UNFREEZE_LAYERS \
    --n_mask_tokens     $N_MASK_TOKENS \
    --n_mask_heads      $N_MASK_HEADS \
    --gauss_sigma       $GAUSS_SIGMA \
    --mask_head_tau     $MASK_HEAD_TAU \
    --sim_attn_tau      $SIM_ATTN_TAU \
    --n_prototypes      $N_PROTOTYPES \
    --tau_proto         $TAU_PROTO \
    --lambda_mu         $LAMBDA_MU \
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
    --lambda_sim        $LAMBDA_SIM \
    --lambda_ortho      $LAMBDA_ORTHO \
    --lambda_dice       $LAMBDA_DICE \
    --lambda_rec        $LAMBDA_REC \
    --lambda_evid       $LAMBDA_EVID \
    --tau_s_beur        $TAU_S_BEUR \
    --tau_s_bef         $TAU_S_BEF \
    --tau_s_img_full    $TAU_S_IMG_FULL \
    --lambda_t2i        $LAMBDA_T2I \
    --same_image_boost  $SAME_IMAGE_BOOST \
    --t2i_mode          $T2I_MODE \
    $( [ "$LEARN_LOSS_WEIGHTS"     = "true" ] && echo "--learn_loss_weights" ) \
    $( [ "$REWEIGHT_BY_N_PHRASES"  = "true" ] && echo "--reweight_by_n_phrases" ) \
    $( [ "$WARM_START_PROJECTIONS" = "true" ] && echo "--warm_start_projections" ) \
    $( [ "$USE_BTXRD"              != "true" ] && echo "--no_btxrd" ) \
    $( [ -n "$OVERFIT_N" ] && echo "--overfit_n $OVERFIT_N" ) \
    $( [ -n "$LOSS_STAGES" ] && echo "--loss_stages $LOSS_STAGES" ) \
    --seed              $SEED \
    --wandb \
    --wandb_project lace-v2-pretrain \
    --wandb_entity  philipp-wiese
    #--losses            $LOSSES \
