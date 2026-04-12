# ====== User-defined variables ======
PUBLISH_NAME="PolinAvA/pi05_pick_cup_200_demos"
DATASET_DIR="/work/nvme/bfbo/xzhang42/openpi/checkpoints/pi05_tea_pick_cup_pytorch"
# ====================================

source /work/nvme/bfbo/xzhang42/openpi/.venv/bin/activate
hf upload-large-folder "${PUBLISH_NAME}" \
  "${DATASET_DIR}" \
  --repo-type model
