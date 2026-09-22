#!/bin/bash
#SBATCH --job-name=token_count
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

home_dir="/mnt/nfs/homedirs/$USER"
export HOME=$home_dir
cd ${SLURM_SUBMIT_DIR}

LOG_FILE="$home_dir/Project/logs/slurm-${SLURM_JOBID}_token_count.out"
exec > "$LOG_FILE" 2>&1

echo "Starting job ${SLURM_JOBID}"
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

MY_CONDA_ENV="master"
export SSL_CERT_FILE=$home_dir/miniconda3/envs/$MY_CONDA_ENV/ssl/cert.pem
export PYTHONUNBUFFERED=1
export HF_HOME=$home_dir/.cache/huggingface
export TRANSFORMERS_CACHE=$home_dir/.cache/huggingface/transformers
export PATH=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin:$home_dir/miniconda3/bin:$PATH
echo "Environment: $MY_CONDA_ENV"

python << 'PYTHON'
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
)

text = "New formation: Bone and articular cartilage, not further specified"

tokens = tokenizer.tokenize(text)

print("Text:", text)
print("Tokens:", tokens)
print("Number of tokens:", len(tokens))
PYTHON