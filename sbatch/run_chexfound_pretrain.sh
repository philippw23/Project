#!/bin/bash
#SBATCH --job-name=chexfound_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --exclude=aioserver3
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"
#SBATCH --error="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.err"


home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export WANDB_DIR=$home_dir/Project/logs
export PYTHONPATH=$home_dir/Project/src
# Avoid PyTorch NVML allocator crashes on nodes where libnvidia-ml is visible
# but nvmlInit fails. This must be set before Python imports torch.
unset PYTORCH_NVML_BASED_CUDA_CHECK
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
echo Environment activated

echo "CUDA/NVML preflight:"
nvidia-smi || echo "WARNING: nvidia-smi failed on $(hostname) (NVML mismatch) — CUDA may still work."

# Run the Python script
NGPUS=${SLURM_GPUS_ON_NODE:-1}
MASTER_PORT=$(( 29500 + SLURM_JOBID % 10000 ))
$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python -m torch.distributed.run --nproc_per_node=$NGPUS --master_port=$MASTER_PORT \
    $home_dir/Project/src/chexfound/train/pretrain.py \
    --config     $home_dir/Project/src/chexfound/configs/chexfound_vitl16_bonetumor.yaml \
    --base_cfg   $home_dir/Project/src/chexfound/data/config.yaml \
    --out_dir    $home_dir/Project/results/chexfound_pretrain/job_${SLURM_JOBID} \
    --batch_size 4
