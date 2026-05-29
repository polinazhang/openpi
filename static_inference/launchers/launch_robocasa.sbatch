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

: "${DATASET:?DATASET env var must be set (atomic-seen|composite-seen|composite-unseen)}"
: "${ROBOCASA_TASK:?ROBOCASA_TASK env var must be set to a single task name}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT env var must be set}"
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR env var must be set}"
: "${NORM_STATS_DIR:?NORM_STATS_DIR env var must be set}"
CONFIG_NAME="${CONFIG_NAME:-pi05_robocasa_checkpoint}"
ROBOCASA_BASE="${ROBOCASA_BASE:-/coc/pskynet4/chuang475/datasets/robocasa}"
MAX_EPISODES="${MAX_EPISODES:-100}"
METRIC="${METRIC:-perturbance-noise}"
CONDITION="${CONDITION:-inference}"
SAVE_META="${SAVE_META:-True}"
SAVE_COSINE="${SAVE_COSINE:-True}"
SAVE_DISPLACEMENT_TRACE="${SAVE_DISPLACEMENT_TRACE:-True}"
SKIP_FRAME="${SKIP_FRAME:-10}"
NUM_STEPS="${NUM_STEPS:-10}"
EMBEDDING_TYPE="${EMBEDDING_TYPE:-vision}"
PERTURBANCE_STEP_NUM="${PERTURBANCE_STEP_NUM:-0}"
PERTURBANCE_STEP_SIZE="${PERTURBANCE_STEP_SIZE:-1e-2}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

mkdir -p "${OUTPUT_ROOT}"

echo "=== Static inference dataset=${DATASET} task=${ROBOCASA_TASK} max_episodes=${MAX_EPISODES} ==="
python static_inference/static_inference.py \
  --dataset "${DATASET}" \
  --config-name "${CONFIG_NAME}" \
  --data.assets.assets-dir "${NORM_STATS_DIR}" \
  --robocasa-task "${ROBOCASA_TASK}" \
  --robocasa-max-episodes "${MAX_EPISODES}" \
  --metric "${METRIC}" \
  --condition "${CONDITION}" \
  --save_meta="${SAVE_META}" \
  --save_cosine="${SAVE_COSINE}" \
  --save_displacement_trace="${SAVE_DISPLACEMENT_TRACE}" \
  --num-steps "${NUM_STEPS}" \
  --skip-frame "${SKIP_FRAME}" \
  --perturbance_step_num "${PERTURBANCE_STEP_NUM}" \
  --perturbance_step_size "${PERTURBANCE_STEP_SIZE}" \
  --embedding_type "${EMBEDDING_TYPE}" \
  --video-backend "${VIDEO_BACKEND}" \
  --output-root "${OUTPUT_ROOT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --norm-stats-dir "${NORM_STATS_DIR}" \
  --robocasa-base "${ROBOCASA_BASE}"

# static_inference.py writes under <output-root>/<dataset>; OUTPUT_ROOT ends in
# .../<task>/perturbance-all (required so static_inference doesn't rewrite our
# path). Collapse both wrappers so the final layout is <task>/<artifacts>.
TASK_DIR="$(dirname "${OUTPUT_ROOT}")"
if [[ -d "${OUTPUT_ROOT}/${DATASET}" ]]; then
  find "${OUTPUT_ROOT}/${DATASET}" -mindepth 1 -maxdepth 1 -exec mv {} "${TASK_DIR}/" \;
  rmdir "${OUTPUT_ROOT}/${DATASET}" || true
fi
find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -exec mv {} "${TASK_DIR}/" \;
rmdir "${OUTPUT_ROOT}" || true
