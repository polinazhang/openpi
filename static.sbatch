#!/bin/bash
#SBATCH --job-name=Pi-Static
#SBATCH --output=results/static-%J.out
#SBATCH --error=results/static-%J.err
#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu=45G

source ~/.bashrc
export XDG_CACHE_HOME=/coc/testnvme/xzhang3205/.cache
source /coc/testnvme/xzhang3205/openpi/.venv/bin/activate
export TRANSFORMERS_CACHE=/coc/testnvme/xzhang3205/huggingface
export OPENPI_DATA_HOME=/coc/testnvme/xzhang3205/openpi/.cache
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
OUTPUT_ROOT=/coc/testnvme/xzhang3205/static
CHECKPOINT_DIR=/coc/testnvme/xzhang3205/openpi/checkpoints/torch_30000

python - <<'PY'
import ctypes
for lib in ["libavutil.so.56", "libavcodec.so.58", "libavformat.so.58"]:
    ctypes.CDLL(lib)
    print("OK", lib)
PY


# DATASETS=(franka_object)
DATASETS=(franka_object_plus)
# DATASETS=(franka_object_two)
# DATASETS=(franka_on_top)
# DATASETS=(franka_object franka_object_two franka_on_top)

# local jax: /coc/testnvme/xzhang3205/openpi/.cache/openpi-assets/checkpoints/pi05_base/params


declare -A DATASET_PROMPTS=(
  [franka_object]=""
  [franka_object_two]="place the pink block in the bin, then place the blue block in the bin"
  [franka_on_top]="place the pink block on top of the blue block"
)

for dataset in "${DATASETS[@]}"; do
  echo "=== Running static metrics for ${dataset} ==="
  prompt="${DATASET_PROMPTS[$dataset]:-}"
  cmd=(python openarm/static_inference.py --dataset "${dataset}" --skip-frame 10 --output-root "${OUTPUT_ROOT}" --checkpoint-dir "${CHECKPOINT_DIR}")
  if [[ -n "${prompt}" ]]; then
    cmd+=(--data.default_prompt "${prompt}")
  fi
  "${cmd[@]}"
done

for dataset in "${DATASETS[@]}"; do
  echo "=== Printing static metrics for ${dataset} ==="
  python openarm/static_print.py --data-dir "${OUTPUT_ROOT}/${dataset}"
done
