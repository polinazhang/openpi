"""Example showing how to integrate the OpenPI policy with metadata logging.

This mirrors what the robot backend should do: every call to ``policy.infer``
logs a *step*, and the evaluation code must call ``policy.end_trajectory()``
once per episode so the consolidated ``npy-metadata`` outputs and
``metadata.json`` entries are produced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


def build_policy(config_name: str, suite: str, data_dir: Path):
    """Create a π0.5 policy that records metadata."""
    cfg = _config.get_config(config_name)
    ckpt_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{config_name}")
    return policy_config.create_trained_policy(
        cfg,
        ckpt_dir,
        evaluation_suite_name=suite,
        data_dir=str(data_dir),
    )


def fake_robot_observation() -> dict[str, np.ndarray | str]:
    """Create a placeholder observation; replace with real robot data."""
    # State is 14 joints + 2 grippers for OpenArm.
    state = np.zeros((16,), dtype=np.float32)
    dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
    return {
        "state": state,
        "left_wrist_image": dummy_img,
        "right_wrist_image": dummy_img,
        "head_image": dummy_img,
        "prompt": "demo prompt",
    }


def run_episode(policy, *, max_steps: int) -> None:
    """Step through a single episode and flush metadata at the end."""
    for step in range(max_steps):
        obs = fake_robot_observation()
        result = policy.infer(obs)
        action = result["actions"]
        print(f"[episode] step={step} action_shape={action.shape}")
        # Robot control code would publish ``action`` here.
    policy.end_trajectory()


def main() -> None:
    parser = argparse.ArgumentParser(description="π0.5 metadata logging example")
    parser.add_argument("--config", default="pi05_droid", help="OpenPI config name")
    parser.add_argument("--suite", default="debug_suite", help="Evaluation suite name")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/tmp/openpi_metadata"),
        help="Root folder that will store npy-metadata outputs",
    )
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to run")
    parser.add_argument("--steps-per-episode", type=int, default=5, help="Steps per episode")
    args = parser.parse_args()

    policy = build_policy(args.config, args.suite, args.data_dir)
    for episode_idx in range(args.episodes):
        print(f"=== episode {episode_idx} ===")
        run_episode(policy, max_steps=args.steps_per_episode)

    print(
        f"Metadata saved under {args.data_dir}/{args.suite}/"
        " and summarized in metadata.json"
    )


if __name__ == "__main__":
    main()
