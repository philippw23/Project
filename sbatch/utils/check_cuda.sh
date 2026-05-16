#!/bin/bash
#SBATCH --job-name=check_cuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

home_dir="/mnt/nfs/homedirs/$USER"

echo "=== nvidia-smi ==="
nvidia-smi

echo ""
echo "=== CUDA Driver Version ==="
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader

echo ""
echo "=== PyTorch CUDA Check ==="
MY_CONDA_ENV="master"
export CONDA_EXE=$home_dir/miniconda3/bin/conda
source $home_dir/miniconda3/etc/profile.d/conda.sh
conda activate $MY_CONDA_ENV

python_path=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python
$python_path - <<'EOF'
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA version    : {torch.version.cuda} (not usable)")
EOF
