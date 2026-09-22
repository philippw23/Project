#!/bin/bash
#SBATCH --job-name=nvml_diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --partition=all_nodes
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j_nvml_diag.out"

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir

MY_CONDA_ENV="master"
export SSL_CERT_FILE=$home_dir/miniconda3/envs/$MY_CONDA_ENV/ssl/cert.pem
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
python=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python

echo "=== nvidia-smi ==="
nvidia-smi

echo ""
echo "=== /dev/nvidia* permissions ==="
ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* found"

echo ""
echo "=== libnvidia-ml.so search ==="
find /usr /mnt/nfs/homedirs/$USER/miniconda3 -name "libnvidia-ml.so*" 2>/dev/null

echo ""
echo "=== pynvml test ==="
$python -c "
import pynvml
try:
    pynvml.nvmlInit()
    print('NVML OK — driver version:', pynvml.nvmlSystemGetDriverVersion())
    count = pynvml.nvmlDeviceGetCount()
    print('GPU count:', count)
    for i in range(count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        print(f'  GPU {i}: {pynvml.nvmlDeviceGetName(h)}, total={mem.total//1024**3}GB, free={mem.free//1024**3}GB')
    pynvml.nvmlShutdown()
except Exception as e:
    print('NVML FAILED:', e)
"

echo ""
echo "=== torch CUDA info ==="
$python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}:', torch.cuda.get_device_name(i))
    total = torch.cuda.get_device_properties(i).total_memory
    print(f'    total memory: {total//1024**3} GB')
"

echo ""
echo "=== LD_LIBRARY_PATH ==="
echo $LD_LIBRARY_PATH
