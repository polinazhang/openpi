#!/bin/bash
#SBATCH --job-name=BPT
#SBATCH --output=results/run-%J.out
#SBATCH --error=results/run-%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
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

# Usage: modify CONFIG_NAME, RUN_NAME, DATASET_NAME, then run: bash convert_all_checkpoints.sh


# ====== User-defined variables ======
CONFIG_NAME="tea_use_spoon_openpi" 
RUN_NAME="run0"
DATASET_NAME="qrafty-ai/tea_use_spoon_openpi"
# ====================================

BASE_DIR="/work/nvme/bfbo/xzhang42/openpi"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints/${CONFIG_NAME}/${RUN_NAME}"
OUTPUT_BASE="${BASE_DIR}/checkpoints/${CONFIG_NAME}_pytorch"
ASSET_SRC="${BASE_DIR}/assets/${CONFIG_NAME}/${DATASET_NAME}"

mkdir -p "${OUTPUT_BASE}"

# ====== Conversion loop ======
echo "Converting checkpoints for ${CONFIG_NAME}/${RUN_NAME} ..."
for ckpt_subdir in "${CHECKPOINT_DIR}"/*/; do
    ckpt_name=$(basename "${ckpt_subdir}")
    output_path="${OUTPUT_BASE}/${RUN_NAME}_${ckpt_name}"
    mkdir -p "${output_path}"

    echo "→ Converting checkpoint: ${ckpt_name}"
    python examples/convert_jax_model_to_pytorch.py \
        --checkpoint-dir "${ckpt_subdir}" \
        --config-name "${CONFIG_NAME}" \
        --output-path "${output_path}"

    echo "→ Syncing assets to ${output_path}/assets/${CONFIG_NAME}/"
    rsync -a "${ASSET_SRC}/" "${output_path}/assets/${CONFIG_NAME}/"
done

echo "✅ Conversion complete. All converted checkpoints stored in: ${OUTPUT_BASE}"
