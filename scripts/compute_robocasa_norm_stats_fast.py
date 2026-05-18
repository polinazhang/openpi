"""Compute RoboCasa norm stats from official dataset metadata.

This is a RoboCasa-specific fast path for normalization stats. It reuses the
OpenPI RoboCasa metadata-stat helpers instead of streaming samples through the
image-loading training dataset, which is unnecessarily slow for state/action
normalization.
"""

import pathlib

import numpy as np
import tyro

import openpi.groot_utils.groot_openpi_dataset as _groot_openpi_dataset
import openpi.shared.normalize as normalize
import openpi.training.config as _config


def _dataset_weight(ds_meta: dict, alpha: float) -> float:
    info_path = pathlib.Path(ds_meta["path"]) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info file not found: {info_path}")
    import json

    info = json.loads(info_path.read_text())
    return float(info["total_frames"]) ** alpha


def main(
    config_name: str,
    dataset_weights_alpha: float = 0.4,
    force: bool = False,
) -> None:
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    data_dirs = getattr(data_config, "data_dirs", None)
    if not data_dirs:
        raise ValueError(f"Config {config_name!r} does not define RoboCasa data_dirs.")
    if data_config.repo_id is None:
        raise ValueError(f"Config {config_name!r} does not define repo_id.")

    output_path = config.assets_dirs / data_config.repo_id
    norm_stats_path = output_path / "norm_stats.json"
    if norm_stats_path.exists() and not force:
        print(f"Norm stats already exist: {norm_stats_path}")
        print("Pass --force to overwrite.")
        return

    per_dataset_norm_stats = [
        _groot_openpi_dataset._load_norm_stats_from_groot_dataset(ds_meta)
        for ds_meta in data_dirs
    ]
    dataset_weights = np.array(
        [_dataset_weight(ds_meta, dataset_weights_alpha) for ds_meta in data_dirs],
        dtype=np.float64,
    )
    norm_stats = _groot_openpi_dataset.compute_overall_statistics(
        per_dataset_norm_stats,
        dataset_sampling_weights=dataset_weights,
    )

    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
