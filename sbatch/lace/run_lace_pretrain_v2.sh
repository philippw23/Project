#!/bin/bash
#SBATCH --job-name=lace_v2_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
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

# LACE v2: L_ITA (stage 1) + L_seg (stage 2) + L_sim v2 with phrases (stage 3)
python $home_dir/Project/src/lace_pretrain_v2.py \
    --splits       $home_dir/Project/data/internal_dataset/split.json \
    --btxrd_images $home_dir/Project/data/BTXRD/images \
    --btxrd_annots $home_dir/Project/data/BTXRD/Annotations \
    --out_dir      $home_dir/Project/results \
    --lora_layers  4 \
    --lora_r       8 \
    --lora_alpha   16 \
    --embed_dim    256 \
    --n_mask_tokens 16 \
    --tau_spatial   0.1 \
    --batch_size   32 \
    --btxrd_batch_size 16 \
    --max_text_len 128 \
    --text_mode    phrase_attn \
    --max_bef_phrases  16 \
    --max_beur_phrases 16 \
    --stage1_epochs 10 \
    --stage2_epochs 15 \
    --stage3_epochs 15 \
    --lr           5e-5 \
    --weight_decay 0.2 \
    --lambda_seg   1.0 \
    --lambda_sim   1.0 \
    --patience     20 \
    --seed         42 \
    --wandb \
    --wandb_project lace-v2-pretrain \
    --wandb_entity  philipp-wiese
