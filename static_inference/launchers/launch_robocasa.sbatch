#!/bin/bash
#SBATCH --job-name=Pi-Static-Robocasa
#SBATCH --output=results/static-robocasa-%J.out
#SBATCH --error=results/static-robocasa-%J.err
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
cd "${OPENPI_ROOT}"
source /coc/testnvme/xzhang3205/miniconda3/etc/profile.d/conda.sh
conda activate robocasa
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/coc/testnvme/xzhang3205/huggingface
export TRANSFORMERS_CACHE=/coc/testnvme/xzhang3205/huggingface
export OPENPI_DATA_HOME="${OPENPI_ROOT}/.cache"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

: "${DATASET:?DATASET must be set}"
: "${ROBOCASA_TASK:?ROBOCASA_TASK must be set}"
: "${ROBOCASA_BASE:?ROBOCASA_BASE must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set}"
: "${NORM_STATS_DIR:?NORM_STATS_DIR must be set}"
: "${DIM_REMAP:?DIM_REMAP must be set}"

MAX_EPISODES="${MAX_EPISODES:-50}"
SKIP_FRAME="${SKIP_FRAME:-1}"
NUM_STEPS="${NUM_STEPS:-10}"
SAVE_META="${SAVE_META:-True}"

mkdir -p "${OUTPUT_ROOT}"

python static_inference/static_inference.py \
  --dataset "${DATASET}" \
  --robocasa-task "${ROBOCASA_TASK}" \
  --robocasa-max-episodes "${MAX_EPISODES}" \
  --robocasa-base "${ROBOCASA_BASE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --norm-stats-dir "${NORM_STATS_DIR}" \
  --dim-remap "${DIM_REMAP}" \
  --output-root "${OUTPUT_ROOT}" \
  --num-steps "${NUM_STEPS}" \
  --skip-frame "${SKIP_FRAME}" \
  --save_meta="${SAVE_META}"

# Preserve the previous task output layout by removing the temporary wrappers.
TASK_DIR="$(dirname "${OUTPUT_ROOT}")"
if [[ -d "${OUTPUT_ROOT}/${DATASET}" ]]; then
  find "${OUTPUT_ROOT}/${DATASET}" -mindepth 1 -maxdepth 1 -exec mv {} "${TASK_DIR}/" \;
  rmdir "${OUTPUT_ROOT}/${DATASET}" || true
fi
find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -exec mv {} "${TASK_DIR}/" \;
rmdir "${OUTPUT_ROOT}" || true
