#!/bin/bash
#SBATCH --job-name=Pi-Static
#SBATCH --output=results/static-%J.out
#SBATCH --error=results/static-%J.err
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
source /coc/testnvme/xzhang3205/openpi/.venv/bin/activate
export PYTHONPATH=/coc/testnvme/xzhang3205/openpi/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/coc/testnvme/xzhang3205/huggingface
export TRANSFORMERS_CACHE=/coc/testnvme/xzhang3205/huggingface
export OPENPI_DATA_HOME=/coc/testnvme/xzhang3205/openpi/.cache
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

DATASET="${DATASET:-franka_object}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/coc/testnvme/xzhang3205/static/franka_full/${DATASET}/gradient-training}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/coc/testnvme/xzhang3205/openpi/checkpoints/torch_30000}"
METRIC="${METRIC:-gradient}"
CONDITION="${CONDITION:-training}"
SAVE_META="${SAVE_META:-True}"
SKIP_FRAME="${SKIP_FRAME:-10}"
NUM_STEPS="${NUM_STEPS:-10}"
MAX_FRAMES="${MAX_FRAMES:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

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

declare -A DATASET_PROMPTS=(
  [franka_object]=""
  [franka_object_plus]=""
  [franka_object_two]="place the pink block in the bin, then place the blue block in the bin"
  [franka_on_top]="place the pink block on top of the blue block"
)

echo "=== Running static metrics dataset=${DATASET} metric=${METRIC} condition=${CONDITION} ==="
prompt="${DATASET_PROMPTS[$DATASET]:-}"
cmd=(
  python static_inference/static_inference.py
  --dataset "${DATASET}"
  --metric "${METRIC}"
  --condition "${CONDITION}"
  --save_meta="${SAVE_META}"
  --num-steps "${NUM_STEPS}"
  --skip-frame "${SKIP_FRAME}"
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
