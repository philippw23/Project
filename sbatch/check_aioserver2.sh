#!/bin/bash
#SBATCH --job-name=check_aioserver2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=aioserver2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"
#SBATCH --error="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.err"

home_dir="/mnt/nfs/homedirs/$USER"

echo "=== Node info ==="
uname -a
echo ""

echo "=== NVIDIA driver ==="
nvidia-smi 2>&1
echo ""

echo "=== NVML version check ==="
python3 -c "import ctypes; lib = ctypes.CDLL('libnvidia-ml.so.1'); print('NVML load: OK')" 2>&1
echo ""

echo "=== PyTorch CUDA ==="
$home_dir/miniconda3/envs/master/bin/python -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    x = torch.ones(3,3).cuda()
    print('Tensor on GPU: OK')
" 2>&1
