## Openpi repo installation
pip install -e . --no-deps


## Integrate Openpi into your pipeline

Two new arguments are required whenever you create a policy:

- `evaluation_suite_name`: A short identifier for the evaluation name (e.g. Use_spoon_eval_Jan10_ckpt_150000). It becomes the folder name under which metadata is stored.
- `data_dir`: Root directory that will hold the recorded metadata.

Example 

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

policy = policy_config.create_trained_policy(
    config,
    checkpoint_dir,
    evaluation_suite_name="libero_object_debug",
    data_dir="/tmp/openpi_metadata",
)
```

Better Example 
`../libero_test.py`


## Convert to hd5
python convert_hd5.py --workers n --source_dir source_dir --target_dir target_dir
