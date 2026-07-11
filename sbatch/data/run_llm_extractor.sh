#!/bin/bash
#SBATCH --job-name=llm_extractor
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
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

MODEL=Qwen/Qwen2.5-14B-Instruct #Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct-AWQ

TWO_STAGE=false    # true = two-stage (atomic extract → classify+rank); false = one-shot joint
# Use a distinct output per methodology so results are not mixed/overwritten.
if [ "$TWO_STAGE" = "true" ]; then
    OUTPUT=$home_dir/Project/data/internal_dataset/text/full_reports_two_stage.json
else
    OUTPUT=$home_dir/Project/data/internal_dataset/test/full_reports_14B_260710_1406.json
fi

# Run the extractor
python_path=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python
$python_path $home_dir/Project/src/llm_extractor.py \
    --model $MODEL \
    --input  $home_dir/Project/data/internal_dataset/text/translated_reports.json \
    --output $OUTPUT \
    --english \
    --batch_size 4 \
    --quantize_8bit \
    $( [ "$TWO_STAGE" = "true" ] && echo "--two_stage" ) \
    #--max 15

