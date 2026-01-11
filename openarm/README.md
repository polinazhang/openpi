## Fine-tuning and Data processing

This directory contains helper scripts plus the training configs needed to fine‑tune Pi models on the OpenArm dataset.

### Downloading datasets
Default saving directory: `/work/nvme/bfbo/xzhang42/datasets/<repo-id>` 

python openarm/download_hf_dataset.py qrafty-ai/tea_use_spoon

### Config
Add a new config in `src/openpi/training/config.py` for the new dataset.

```
TrainConfig(
    name="tea_use_spoon",
    project_name="tea_use_spoon",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        max_token_len=220,
    ),
    data=LeRobotOpenArmDataConfig(
        repo_id="qrafty-ai/tea_use_spoon_openpi",
        assets=AssetsConfig(asset_id="qrafty-ai/tea_use_spoon_openpi"),
        base_config=DataConfig(prompt_from_task=False),
        dataset_action_dim=16,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=20_000,
    batch_size=32,
),
```

### Downloading norm stats
python scripts/compute_norm_stats.py \
  --config-name tea_use_spoon \
  --repo-root /work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_use_spoon_openpi

### Training
python scripts/train.py \
  --config-name tea_use_spoon \
  --exp-name spoon_run0 \