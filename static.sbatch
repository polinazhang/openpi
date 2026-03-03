#!/bin/bash
#SBATCH --job-name=Openpi-Convert
#SBATCH --output=results/convert-%J.out
#SBATCH --error=results/convert-%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --account=bfbo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=220G
#SBATCH --nodes=1

source ~/.bashrc
cd /work/nvme/bfbo/xzhang42/openpi
source /work/nvme/bfbo/xzhang42/openpi/.venv/bin/activate

set -euo pipefail

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

DATASETS=(use_steel_spoon)

for dataset in "${DATASETS[@]}"; do
  echo "=== Running static metrics for ${dataset} ==="
  python openarm/static_inference.py --dataset "${dataset}" --skip-frame 10
done

for dataset in "${DATASETS[@]}"; do
  echo "=== Printing static metrics for ${dataset} ==="
  python openarm/static_print.py --data-dir "/work/nvme/bfbo/xzhang42/static/${dataset}"
done
