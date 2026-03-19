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
set -euo pipefail
export XDG_CACHE_HOME=/coc/testnvme/xzhang3205/.cache
source /coc/testnvme/xzhang3205/openpi/.venv/bin/activate
export TRANSFORMERS_CACHE=/coc/testnvme/xzhang3205/huggingface
export OPENPI_DATA_HOME=/coc/testnvme/xzhang3205/openpi/.cache
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
OUTPUT_ROOT="${OUTPUT_ROOT:-/coc/testnvme/xzhang3205/static}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/coc/testnvme/xzhang3205/openpi/checkpoints/torch_30000}"
DATASET="${DATASET:-franka_object}"
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
  [franka_object_two]="place the pink block in the bin, then place the blue block in the bin"
  [franka_on_top]="place the pink block on top of the blue block"
)

echo "=== Running static metrics for ${DATASET} ==="
prompt="${DATASET_PROMPTS[$DATASET]:-}"
cmd=(python openarm/static_inference.py --dataset "${DATASET}" --skip-frame 10 --output-root "${OUTPUT_ROOT}" --checkpoint-dir "${CHECKPOINT_DIR}")
if [[ -n "${prompt}" ]]; then
  cmd+=(--data.default_prompt "${prompt}")
fi
"${cmd[@]}"

echo "=== Printing static metrics for ${DATASET} ==="
python openarm/static_print.py --data-dir "${OUTPUT_ROOT}/${DATASET}"
