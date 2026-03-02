#!/bin/bash
#SBATCH --job-name=OpenPi
#SBATCH --output=results/pi-%J.out
#SBATCH --error=results/pi-%J.err
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

python - <<'PY'
import ctypes
for lib in ["libavutil.so.56", "libavcodec.so.58", "libavformat.so.58"]:
    ctypes.CDLL(lib)
    print("OK", lib)
PY


# python scripts/train.py \
#     pi05_franka_object \
#     --data.repo_id franka_object \
#     --exp-name full1 \
#     --fsdp-devices 4 \
#     --batch-size 16

python scripts/train.py \
    pi05_franka_object_lora \
    --data.repo_id franka_object \
    --exp-name lora0 \
    --fsdp-devices 4 \
    --batch-size 32


# --data.default_prompt "pick up the pink block and place it on the blue block" \  # don't use it for multiple task dataset
