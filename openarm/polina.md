## To do list before sending checkpoints to cc
Convert from jax to torch; attach norm stats

## To do list after receiving checkpoints from cc
Compute stats, plot

## Convert checkpoints to torch

python examples/convert_jax_model_to_pytorch.py \
--checkpoint-dir /work/nvme/bfbo/xzhang42/openpi/checkpoints/pi05_openarm_tea_continuous/test/15000/ \
--config-name pi05_openarm_tea_continuous \
--output-path /work/nvme/bfbo/xzhang42/openpi/checkpoints/pi05_openarm_tea_continuous/test/15000_pytorch

rsync -a /work/nvme/bfbo/xzhang42/openpi/checkpoints/pi05_openarm_tea_continuous/test/15000/assets/ \
/work/nvme/bfbo/xzhang42/openpi/checkpoints/pi05_openarm_tea_continuous/test/15000_pytorch/assets/

rm -f /work/nvme/bfbo/xzhang42/openpi/checkpoints/pi05_openarm_tea_continuous/test/15000_pytorch/assets/norm_stats.json
mkdir -p /work/nvme/bfbo/xzhang42/openpi/checkpoints/pi05_openarm_tea_continuous/test/15000_pytorch/assets/openarm/tea_continuous
rsync -a /work/nvme/bfbo/xzhang42/openpi/assets/pi05_openarm_tea_continuous/openarm/tea_continuous/ /work/nvme/bfbo/xzhang42/openpi/checkpoints/pi05_openarm_tea_continuous/test/15000_pytorch/assets/openarm/tea_continuous/


python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --config-name pi05_libero \
  --output-path /work/nvme/bfbo/xzhang42/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch

PYTHONPATH=src python - <<'PY'
from openpi.shared import download
path = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero/assets")
print(f"Assets cached at: {path}")
PY

rsync -a /work/nvme/bfbo/xzhang42/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/ /work/nvme/bfbo/xzhang42/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/assets/physical-intelligence/libero/