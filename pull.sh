#!/bin/bash
set -euo pipefail

HF_REPO="PolinAvA/pi05_franka_base"
HF_REPO="RomanYakunin/pi05_franka_object_single_torch_30000"
HF_REPO="PolinAvA/openarm_static"
DEST_DIR="/coc/testnvme/xzhang3205/static/openarm_prior"

mkdir -p "$DEST_DIR"

echo "[pull.sh] Downloading EVERYTHING from $HF_REPO…"
hf download "$HF_REPO" \
  --repo-type dataset \
  --local-dir "$DEST_DIR" \
  --cache-dir "/work/nvme/bfbo/xzhang42/.cache/huggingface" \
  --revision main

echo "[pull.sh] Done (old local files are NOT deleted)."
