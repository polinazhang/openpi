#!/bin/bash
#SBATCH --job-name=BPT
#SBATCH --output=results/run-%J.out
#SBATCH --error=results/run-%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --account=bfbo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=220G
#SBATCH --nodes=1

# module load cuda/12.6.1
source ~/.bashrc
cd /work/nvme/bfbo/xzhang42/openpi
source /work/nvme/bfbo/xzhang42/openpi/.venv/bin/activate
# export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME=/work/nvme/bfbo/xzhang42/.cache
export OPENPI_DATA_HOME=/work/nvme/bfbo/xzhang42/openpi/.cache

# EXTREMELY IMPORTANT: completely remove 12.3 SDK paths
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/opt/nvidia/hpc_sdk/Linux_aarch64/24.3' | paste -sd:)

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"

python - <<'PY'
import jax, jaxlib, jax.numpy as jnp
print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("jaxlib file", jaxlib.__file__)
print("devices", jax.devices())
x = jnp.ones((2048, 2048), dtype=jnp.float16)
y = x @ x
print(y.block_until_ready())
PY

# python scripts/compute_norm_stats.py \
# --config-name pi05_openarm_tea_continuous \
# --repo-root /work/nvme/bfbo/xzhang42/datasets/openarm_processed/openarm/tea_continuous

# python scripts/compute_norm_stats.py \
# --config-name pi05_openarm_tea_discrete \
# --repo-root /work/nvme/bfbo/xzhang42/datasets/openarm_processed/openarm/tea_discrete

# python scripts/train.py \
# pi05_openarm_tea_continuous \
# --data.repo_id /work/nvme/bfbo/xzhang42/datasets/openarm_processed/openarm/tea_continuous \
# --exp-name test

python scripts/train.py \
pi05_openarm_tea_discrete \
--data.repo_id /work/nvme/bfbo/xzhang42/datasets/openarm_processed/openarm/tea_discrete \
--exp-name run0