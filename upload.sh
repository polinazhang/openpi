# ====== User-defined variables ======
PUBLISH_NAME="qrafty-ai/pi05_tea_use_spoon_openpi"
DATASET_DIR="/work/nvme/bfbo/xzhang42/openpi/checkpoints/pi05_tea_use_spoon_openpi"
# ====================================

source /work/nvme/bfbo/xzhang42/openpi/.venv/bin/activate
hf upload-large-folder "${PUBLISH_NAME}" \
  "${DATASET_DIR}" \
  --repo-type model
