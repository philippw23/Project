#!/bin/bash
#SBATCH --job-name=check_aioserver3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=aioserver3
#SBATCH --time=00:05:00
#SBATCH --output="/mnt/nfs/homedirs/%u/Project/logs/slurm-%j.out"

home_dir="/mnt/nfs/homedirs/$USER"
echo "=== Node info ==="
uname -a
echo ""

echo "=== OS release ==="
cat /etc/os-release
echo ""

echo "=== /home/philippw symlink ==="
ls -la /home/philippw 2>&1
echo ""

echo "=== conda binary exists? ==="
ls -la $home_dir/miniconda3/bin/conda 2>&1
echo ""

echo "=== conda binary ldd ==="
ldd $home_dir/miniconda3/bin/conda 2>&1
echo ""

echo "=== python binary exists? ==="
ls -la $home_dir/miniconda3/envs/master/bin/python 2>&1
echo ""

echo "=== python binary ldd ==="
ldd $home_dir/miniconda3/envs/master/bin/python 2>&1
echo ""

echo "=== dynamic linker present? ==="
ls -la /lib64/ld-linux-x86-64.so.2 2>&1
ls -la /lib/x86_64-linux-gnu/libc.so.6 2>&1
echo ""

echo "=== df (disk/mount check) ==="
df -h $home_dir 2>&1
