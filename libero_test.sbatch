#!/bin/bash
#SBATCH --job-name=BPT
#SBATCH --output=results/libero-%J.out
#SBATCH --error=results/libero-%J.err
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH --account=bfbo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=220G
#SBATCH --nodes=1

set -euo pipefail

# Activate project environment.
export XDG_CACHE_HOME=/work/nvme/bfbo/xzhang42/.cache
export OPENPI_DATA_HOME=/work/nvme/bfbo/xzhang42/.cache/openpi
source /work/nvme/bfbo/xzhang42/openpi/.venv/bin/activate

export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/opt/nvidia/hpc_sdk/Linux_aarch64/24.3' | paste -sd:)
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export TORCH_COMPILE_DISABLE=1
export PYTHONPATH="/work/nvme/bfbo/xzhang42/openpi/third_party/libero:${PYTHONPATH:-}"

cd /work/nvme/bfbo/xzhang42/openpi
python libero_test.py
