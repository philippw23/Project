#!/bin/bash
#SBATCH --job-name=inspect_shapes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

home_dir="/mnt/nfs/homedirs/$USER"
cd "${SLURM_SUBMIT_DIR:-$home_dir/Project}"

echo "Starting job ${SLURM_JOBID:-local}"
echo "SLURM assigned me these nodes:"
squeue -j "${SLURM_JOBID}" -O nodelist 2>/dev/null | tail -n +2

MY_CONDA_ENV="master"
export SSL_CERT_FILE=$home_dir/miniconda3/envs/$MY_CONDA_ENV/ssl/cert.pem
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME="$home_dir/.cache/huggingface"
export TRANSFORMERS_CACHE="$home_dir/.cache/huggingface/transformers"

export PATH="$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH"
export PYTHONPATH="$home_dir/Project/src"
unset PYTORCH_NVML_BASED_CUDA_CHECK
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
echo "Environment activated"

python_path="$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python"
"$python_path" -u "$home_dir/Project/src/inspect_tensor_shapes.py" \
    --model       chexfound \
    --config      "$home_dir/Project/src/chexfound/configs/chexfound_vitl16_bonetumor.yaml" \
    --base_cfg    "$home_dir/Project/src/chexfound/data/config.yaml" \
    --lora_layers 6
