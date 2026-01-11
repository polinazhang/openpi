## Fine-tuning and Data processing

This directory contains helper scripts plus the training configs needed to fine‑tune Pi models on the OpenArm dataset.

### Downloading datasets
Default directory: `/work/nvme/bfbo/xzhang42/datasets/<repo-id>` 

```bash
python openarm/download_hf_dataset.py qrafty-ai/tea_use_spoon
```

## Preprocess into Pi-ready datasets
```bash
python openarm/preprocess.py \
  --repo-id qrafty-ai/tea_use_spoon \
  --input-root /work/nvme/bfbo/xzhang42/datasets \
  --output-root /work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_use_spoon_processed
```
Automatically creates `<repo-id>_processed` next to the raw dataset (e.g., `/work/nvme/.../qrafty-ai/tea_use_spoon_processed`).
Use `--skip-videos` for debugging.
By default only the discrete dataset is produced; pass `--variant both` (as in the sbatch script) if you also need the continuous version.

The converter mirrors the LeRobot `task` string into a `prompt` field so the Hub task-metadata lookup can be skipped.

## Compute normalization stats 
The OpenArm assets live under `./assets/<config>/<asset_id>`:
  ```bash
  python scripts/compute_norm_stats.py --config-name pi05_openarm_tea_continuous
  python scripts/compute_norm_stats.py --config-name pi05_openarm_tea_discrete
  ```

## Fine-tuning
```bash
python -m openpi.training.main --config-name pi05_openarm_tea_continuous --exp-name <run-name>
```
The processed datasets to be available under the same repo IDs (push them to the Hub or override `repo_id` when invoking Tyro if you keep them local).
