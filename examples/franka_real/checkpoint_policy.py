"""Inference-only Cartesian checkpoint contract; no training config is modified."""
import dataclasses
import hashlib
import logging
from pathlib import Path

import numpy as np


def load_checkpoint_stats(checkpoint_dir):
    """Return only the selected checkpoint's own validated statistics."""
    if checkpoint_dir is None:
        raise ValueError("Set checkpoint_dir to an absolute path in repo-configs/config.py before inference.")
    checkpoint = Path(checkpoint_dir)
    if not checkpoint.is_absolute():
        raise ValueError("checkpoint_dir must be an absolute local path")
    checkpoint = checkpoint.resolve(strict=True)
    stats_path = checkpoint / "assets" / "franka" / "norm_stats.json"
    if not stats_path.resolve(strict=True).is_relative_to(checkpoint):
        raise ValueError("Checkpoint norm_stats.json must not link to another assets directory")
    from openpi.shared import normalize
    contents = stats_path.read_bytes()
    stats = normalize.deserialize_json(contents.decode())
    if set(stats) != {"state", "actions"}:
        raise ValueError("Checkpoint statistics must contain exactly state and actions")
    for name in ("state", "actions"):
        item = stats[name]
        for field in ("mean", "std", "q01", "q99"):
            values = getattr(item, field)
            if values is None or np.asarray(values).shape != (8,) or not np.isfinite(values).all():
                raise ValueError(f"Checkpoint {name}.{field} must contain eight finite values")
        if np.any(item.q99 < item.q01):
            raise ValueError(f"Invalid checkpoint {name} quantile bounds")
    digest = hashlib.sha256(contents).hexdigest()
    logging.info("Normalization source: %s (sha256=%s); no assets fallback", stats_path, digest)
    return checkpoint, stats_path, stats, digest


def make_inference_config(checkpoint, stats):
    from openpi import transforms
    from openpi.policies import openarm_policy
    from openpi.training import config as training

    @dataclasses.dataclass(frozen=True)
    class CartesianData(training.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            # Deliberately bypass create_base_config/_load_norm_stats: they search assets.
            return training.DataConfig(
                asset_id="franka",
                norm_stats=stats,
                use_quantile_norm=True,
                data_transforms=transforms.Group(inputs=[openarm_policy.OpenArmInputs()]),
                model_transforms=training.ModelTransformFactory()(model_config),
            )

    base = training.get_config("pi05_franka_object")
    return dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, pi05=True, action_dim=32, action_horizon=50, max_token_len=200),
        data=CartesianData(),
        assets_base_dir=str(checkpoint / "assets"),
        checkpoint_base_dir=str(checkpoint.parent),
        policy_metadata={
            **(base.policy_metadata or {}),
            "action_representation": "absolute_cartesian_xyz_quaternion_xyzw",
            "tag": "cartesian",
            "action_horizon": 50,
            "control_hz": 20,
        },
    )


def load_policy(cfg):
    # Validate before importing the model stack, downloading assets, or constructing a model.
    checkpoint, stats_path, stats, digest = load_checkpoint_stats(cfg.checkpoint_dir)
    if cfg.model_family.value != "pi05":
        raise ValueError("This Franka workflow requires the specified Cartesian Pi0.5 checkpoint")
    if cfg.norm_stats_path is not None and Path(cfg.norm_stats_path).resolve() != stats_path:
        raise ValueError("Normalization path must be checkpoint_dir/assets/franka/norm_stats.json")
    from openpi.policies import policy_config
    train_config = make_inference_config(checkpoint, stats)
    train_config = dataclasses.replace(train_config, policy_metadata={
        **train_config.policy_metadata, "norm_stats_path": str(stats_path), "norm_stats_sha256": digest,
    })
    return policy_config.create_trained_policy(
        train_config, checkpoint,
        evaluation_suite_name=cfg.evaluation_suite_name,
        data_dir=cfg.data_dir,
        default_prompt=cfg.default_prompt,
        norm_stats=stats,
    )
