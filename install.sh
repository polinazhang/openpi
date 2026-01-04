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

source /work/nvme/bfbo/xzhang42/openpi/.venv/bin/activate

export HERMETIC_CUDA_VERSION=12.4.0
export HERMETIC_CUDNN_VERSION=9.1.1

mkdir -p /work/nvme/bfbo/xzhang42/tmp
export TMPDIR=/work/nvme/bfbo/xzhang42/tmp
export TEMP=/work/nvme/bfbo/xzhang42/tmp
export TMP=/work/nvme/bfbo/xzhang42/tmp

cd /work/nvme/bfbo/xzhang42/jax

python build/build.py build \
--wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt \
--cuda_version=12.4.0 \
--cudnn_version=9.1.1 \
--cuda_compute_capabilities=sm_90,compute_90 \
--bazel_path=/work/nvme/bfbo/xzhang42/bin/bazel \
--use_clang=false \
--gcc_path=$(which gcc) \
--bazel_options="--sandbox_tmpfs_path=$TMPDIR" \
--bazel_options="--action_env=TMPDIR=$TMPDIR"

/work/nvme/bfbo/xzhang42/bin/bazelisk build //jaxlib/tools:build_gpu_plugin_wheel \
  --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
  --repo_env=TF_CUDA_VERSION=12.4 \
  --repo_env=TF_CUDNN_VERSION=9.1.1 \
  --repo_env=CUDA_TOOLKIT_PATH=/usr/local/cuda-12.4 \
  --repo_env=PLATFORM_VERSION=12.4

/work/nvme/bfbo/xzhang42/jax/bazel-bin/jaxlib/tools/build_gpu_plugin_wheel \
  --output_path=/work/nvme/bfbo/xzhang42/jax/dist \
  --cpu=aarch64 \
  --enable-cuda=True \
  --platform_version=12.4 \
  --jaxlib_git_hash=$(git rev-parse HEAD)


python -m pip install --no-deps /work/nvme/bfbo/xzhang42/jax/dist/jaxlib-*.whl
python -m pip install --no-deps /work/nvme/bfbo/xzhang42/jax/dist/jax_cuda_plugin-*.whl
python -m pip install --no-deps /work/nvme/bfbo/xzhang42/jax/dist/jax_cuda_pjrt-*.whl

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
