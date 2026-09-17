#!/bin/bash
set -euo pipefail

HF_REPO="PolinAvA/pi05_franka_jax"
OPENPI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST_DIR="$OPENPI_DIR/checkpoints"

mkdir -p "$DEST_DIR"

echo "[pull.sh] Downloading EVERYTHING from $HF_REPO…"
hf download "$HF_REPO" \
  --repo-type model \
  --local-dir "$DEST_DIR" \
  --cache-dir "$OPENPI_DIR/.cache/huggingface" \
  --revision main

echo "[pull.sh] Done (old local files are NOT deleted)."