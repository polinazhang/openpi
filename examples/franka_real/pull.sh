#!/bin/bash
set -euo pipefail

HF_REPO="PolinAvA/pi05_franka_jax"
DEST_DIR="/home/ripl/openpi/checkpoints"

mkdir -p "$DEST_DIR"

echo "[pull.sh] Downloading EVERYTHING from $HF_REPO…"
hf download "$HF_REPO" \
  --repo-type model \
  --local-dir "$DEST_DIR" \
  --cache-dir "/home/ripl/openpi/.cache/huggingface" \
  --revision main

echo "[pull.sh] Done (old local files are NOT deleted)."