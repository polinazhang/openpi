#!/bin/bash
#SBATCH --job-name=Pi-Static
#SBATCH --output=results/static-%J.out
#SBATCH --error=results/static-%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --account=bfbo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=220G
#SBATCH --nodes=1

source ~/.bashrc
set -euo pipefail


# module load cuda/12.6.1
source ~/.bashrc
cd /work/nvme/bfbo/xzhang42/openpi
PY=/work/nvme/bfbo/xzhang42/openpi/.venv/bin/python
# export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME=/work/nvme/bfbo/xzhang42/.cache
export HF_HOME=/work/nvme/bfbo/xzhang42/huggingface
export TRANSFORMERS_CACHE=/work/nvme/bfbo/xzhang42/huggingface
export OPENPI_DATA_HOME=/work/nvme/bfbo/xzhang42/openpi/.cache

# EXTREMELY IMPORTANT: completely remove 12.3 SDK paths
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/opt/nvidia/hpc_sdk/Linux_aarch64/24.3' | paste -sd:)

export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"

unset JAX_PLUGINS JAX_PLATFORM_NAME JAX_PLATFORMS
unset NCCL_NET NCCL_NET_PLUGIN NCCL_NET_OFI_PROVIDER NCCL_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_CROSS_NIC NCCL_PLUGIN_P2P

"$PY" - <<'PY'
import jax, jaxlib, jax.numpy as jnp
print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("jaxlib file", jaxlib.__file__)
print("devices", jax.devices())
x = jnp.ones((2048, 2048), dtype=jnp.float16)
y = x @ x
print(y.block_until_ready())
PY


DATASET="${DATASET:-franka_object_action_ood}"
# OUTPUT_ROOT="${OUTPUT_ROOT:-/coc/testnvme/xzhang3205/static/franka_full/${DATASET}/gradient-training}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/coc/testnvme/xzhang3205/static/franka_full/${DATASET}/perturbance-noise}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/work/nvme/bfbo/xzhang42/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
# METRIC="${METRIC:-gradient}"
METRIC="${METRIC:-perturbance-noise}"
# CONDITION="${CONDITION:-training}"
CONDITION="${CONDITION:-inference}"
# SAVE_META="${SAVE_META:-True}"
SAVE_META="${SAVE_META:-False}"
SKIP_FRAME="${SKIP_FRAME:-10}"
NUM_STEPS="${NUM_STEPS:-10}"
MAX_FRAMES="${MAX_FRAMES:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
PERTURBANCE_STEP_NUM="${PERTURBANCE_STEP_NUM:-0}"
PERTURBANCE_STEP_SIZE="${PERTURBANCE_STEP_SIZE:-1e-2}"
# EMBEDDING_TYPE="${EMBEDDING_TYPE:-vision+action}"
EMBEDDING_TYPE="${EMBEDDING_TYPE:-vision}"

mkdir -p "${OUTPUT_ROOT}"

# "$PY" - <<'PY' || true
# import ctypes
# for candidates in [
#     ["libavutil.so.56", "libavutil.so.57", "libavutil.so.58"],
#     ["libavcodec.so.58", "libavcodec.so.59", "libavcodec.so.60"],
#     ["libavformat.so.58", "libavformat.so.59", "libavformat.so.60"],
# ]:
#     for lib in candidates:
#         try:
#             ctypes.CDLL(lib)
#             print("OK", lib)
#             break
#         except OSError:
#             continue
# PY

declare -A DATASET_PROMPTS=(
  [franka_object]=""
  [franka_object_plus]=""
  [franka_object_two]="place the pink block in the bin, then place the blue block in the bin"
  [franka_on_top]="place the pink block on top of the blue block"
  [franka_object_action_ood]=""
  [franka_object_vision_ood]=""
  [pick_cup]=""
  [pour_ice]=""
  [use_spoon]=""
  [use_steel_spoon]=""
)

echo "=== Running static metrics dataset=${DATASET} metric=${METRIC} condition=${CONDITION} ==="
prompt="${DATASET_PROMPTS[$DATASET]:-}"
cmd=(
  "$PY" static_inference/static_inference.py
  --dataset "${DATASET}"
  --metric "${METRIC}"
  --condition "${CONDITION}"
  --save_meta="${SAVE_META}"
  --num-steps "${NUM_STEPS}"
  --skip-frame "${SKIP_FRAME}"
  --perturbance_step_num "${PERTURBANCE_STEP_NUM}"
  --perturbance_step_size "${PERTURBANCE_STEP_SIZE}"
  --embedding_type "${EMBEDDING_TYPE}"
  --video-backend "${VIDEO_BACKEND}"
  --output-root "${OUTPUT_ROOT}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
)

if [[ "${MAX_FRAMES}" != "0" ]]; then
  cmd+=(--max-frames "${MAX_FRAMES}")
fi
if [[ -n "${prompt}" ]]; then
  cmd+=(--data.default_prompt "${prompt}")
fi
"${cmd[@]}"

# static_inference.py currently writes to <output-root>/<dataset>; flatten to keep requested
# layout where <output-root> itself is the condition directory.
if [[ -d "${OUTPUT_ROOT}/${DATASET}" ]]; then
  find "${OUTPUT_ROOT}/${DATASET}" -mindepth 1 -maxdepth 1 -exec mv {} "${OUTPUT_ROOT}/" \;
  rmdir "${OUTPUT_ROOT}/${DATASET}" || true
fi
