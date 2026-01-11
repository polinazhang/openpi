#!/bin/bash
#SBATCH --job-name=jax_cuda_build_test
#SBATCH --output=results/jax_cuda_build_test-%j.out
#SBATCH --error=results/jax_cuda_build_test-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --nodes=1
#SBATCH --account=bfbo-dtai-gh

set -euo pipefail

source ~/.bashrc

module load cuda/12.4
module load gcc/11.4.0

export XDG_CACHE_HOME=/work/nvme/bfbo/xzhang42/.cache
source /work/nvme/bfbo/xzhang42/openpi/.venv/bin/activate

export HERMETIC_CUDA_VERSION=12.4.0
export HERMETIC_CUDNN_VERSION=9.1.1

mkdir -p /work/nvme/bfbo/xzhang42/tmp
export TMPDIR=/work/nvme/bfbo/xzhang42/tmp
export TEMP=/work/nvme/bfbo/xzhang42/tmp
export TMP=/work/nvme/bfbo/xzhang42/tmp


python -m pip install --no-deps --force-reinstall \
    /work/nvme/bfbo/xzhang42/jax/dist/jaxlib-0.5.3.dev20260110-cp311-cp311-manylinux2014_aarch64.whl \
    /work/nvme/bfbo/xzhang42/jax/dist/jax_cuda12_plugin-0.5.3.dev20260110-cp311-cp311-manylinux2014_aarch64.whl \
    /work/nvme/bfbo/xzhang42/jax/dist/jax_cuda12_pjrt-0.5.3.dev20260110-py3-none-manylinux2014_aarch64.whl \
    /work/nvme/bfbo/xzhang42/jax/dist/jax_cuda12_4_pjrt-0.5.3.dev20260110-py3-none-manylinux2014_aarch64.whl

# completely remove 12.3 SDK paths
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/opt/nvidia/hpc_sdk/Linux_aarch64/24.3' | paste -sd:)
# then prepend 12.4
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
