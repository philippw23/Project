#!/bin/bash
#SBATCH --job-name=llm_extractor_sep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

home_dir="/mnt/nfs/homedirs/$USER"
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

# Activate conda environment
MY_CONDA_ENV="master"
export CONDA_EXE=$home_dir/miniconda3/bin/conda
source $home_dir/miniconda3/etc/profile.d/conda.sh
conda activate $MY_CONDA_ENV
echo Environment activated

# Redirect HuggingFace cache to NFS home (compute nodes have no /home)
export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers

# Run the separated extractor
# Two-stage per-section pipeline (atomic extraction → importance ranking).
# Input is English-only: the loader reads befund_en / beurteilung_en.
python_path=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python
$python_path $home_dir/Project/src/llm_extractor_seperated.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --reports $home_dir/Project/data/internal_dataset/text/translated_reports.json \
    --out_dir $home_dir/Project/results \
    --output $home_dir/Project/data/internal_dataset/test/full_report_test_15.json \
    --two_stage \
    --batch_size 4 \
    --quantize_8bit \
    --max 15
