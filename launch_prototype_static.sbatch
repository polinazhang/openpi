#!/bin/bash
#SBATCH --job-name=Pi-Prototype
#SBATCH --output=results/prototype-static-%J.out
#SBATCH --error=results/prototype-static-%J.err
#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu=45G

source ~/.bashrc
set -euo pipefail

export XDG_CACHE_HOME=/coc/testnvme/xzhang3205/.cache
OPENPI_ROOT="${OPENPI_ROOT:-/coc/testnvme/xzhang3205/vla-adaptation/models/openpi}"
source "${OPENPI_ROOT}/.venv/bin/activate"
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/static_inference${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/coc/testnvme/xzhang3205/huggingface
export TRANSFORMERS_CACHE=/coc/testnvme/xzhang3205/huggingface
export OPENPI_DATA_HOME="${OPENPI_ROOT}/.cache"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

DATASET="${DATASET:?DATASET must be set}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set}"
NORM_STATS_DIR="${NORM_STATS_DIR:-}"
MESA_ROOT="${MESA_ROOT:-/coc/testnvme/xzhang3205/vla-adaptation/envs/mesa-env}"
ROBOCASA_BASE="${ROBOCASA_BASE:-/coc/pskynet4/chuang475/datasets/robocasa}"
SKIP_FRAME="${SKIP_FRAME:-1}"
NUM_STEPS="${NUM_STEPS:-10}"
MAX_FRAMES="${MAX_FRAMES:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
EMBEDDING_TYPE="${EMBEDDING_TYPE:-vision}"

mkdir -p "${OUTPUT_ROOT}"

python - <<'PY' || true
import ctypes
for candidates in [
    ["libavutil.so.56", "libavutil.so.57", "libavutil.so.58"],
    ["libavcodec.so.58", "libavcodec.so.59", "libavcodec.so.60"],
    ["libavformat.so.58", "libavformat.so.59", "libavformat.so.60"],
]:
    for lib in candidates:
        try:
            ctypes.CDLL(lib)
            print("OK", lib)
            break
        except OSError:
            continue
PY

cmd=(
  python static_inference/static_inference_single.py
  --dataset "${DATASET}"
  --use-first-episode=True
  --metric perturbance-noise
  --condition inference
  --save_meta=True
  --save_cosine=True
  --save_displacement_trace=True
  --num-steps "${NUM_STEPS}"
  --skip-frame "${SKIP_FRAME}"
  --embedding_type "${EMBEDDING_TYPE}"
  --video-backend "${VIDEO_BACKEND}"
  --output-root "${OUTPUT_ROOT}"
  --output-root-final=True
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --mesa-root "${MESA_ROOT}"
  --robocasa-base "${ROBOCASA_BASE}"
)

if [[ "${MAX_FRAMES}" != "0" ]]; then
  cmd+=(--max-frames "${MAX_FRAMES}")
fi
if [[ -n "${NORM_STATS_DIR}" ]]; then
  cmd+=(--norm-stats-dir "${NORM_STATS_DIR}")
fi

echo "=== Running prototype static inference dataset=${DATASET} output=${OUTPUT_ROOT} ==="
"${cmd[@]}"
