#!/bin/bash
#SBATCH --job-name=robocasa-finetune
#SBATCH --output=/coc/testnvme/xzhang3205/vla-adaptation/models/openpi/results/finetune/robocasa-finetune-%J.out
#SBATCH --error=/coc/testnvme/xzhang3205/vla-adaptation/models/openpi/results/finetune/robocasa-finetune-%J.err
#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node="a40:8"
#SBATCH --cpus-per-task=8
#SBATCH --qos="long"
#SBATCH --time="6-00:00:00"
#SBATCH --exclude="sonny"
#SBATCH --mem-per-gpu=45G

set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/coc/testnvme/xzhang3205/vla-adaptation/models/openpi}"
cd "${OPENPI_ROOT}"
source /coc/testnvme/xzhang3205/miniconda3/etc/profile.d/conda.sh
conda activate robocasa
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"

python scripts/train.py "$RUN_BASE_CONFIG" \
  --exp-name="$RUN_EXP_NAME" \
  --checkpoint-base-dir="$RUN_CHECKPOINT_BASE_DIR" \
  --data.assets.assets-dir="$RUN_START_CHECKPOINT/assets" \
  --data.data-dirs-json="$RUN_DATA_DIRS_JSON" \
  --weight-loader.params-path="$RUN_START_CHECKPOINT/params"
