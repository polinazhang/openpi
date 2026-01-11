#!/usr/bin/env python3
"""Minimal Libero evaluation to exercise Pi0.5 metadata logging.

Runs three CPU-only episodes in the `libero_object` suite, recording metadata
under /work/nvme/bfbo/xzhang42/libero_test/debug_suite/.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
import torch.serialization
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.shared import download


# Configuration (hard-coded per user request).
DATA_ROOT = Path("/work/nvme/bfbo/xzhang42/libero_test").resolve()
EVALUATION_SUITE = "debug_suite"
LIBERO_SUITE = "libero_object"
NUM_EPISODES = 3
REPLAN_STEPS = 5
WAIT_STEPS = 10
MAX_STEPS = 280
CONFIG_NAME = "pi05_libero"
CHECKPOINT_PATH = (
    Path("/work/nvme/bfbo/xzhang42/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch").resolve()
)

LIBERO_DUMMY_ACTION = np.array([0.0] * 6 + [-1.0], dtype=np.float32)

# Allow Libero's pickled initial-state tensors under PyTorch 2.6+.
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle, matching examples/libero/main.py."""
    w, x, y, z = quat
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(1 - w * w)
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([x, y, z]) / s
    return axis * angle


def _preprocess_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    image = np.ascontiguousarray(image[::-1, ::-1])
    image = image_tools.resize_with_pad(image, size, size)
    return image_tools.convert_to_uint8(image)


def _format_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )


def _get_libero_env(task) -> tuple[OffScreenRenderEnv, str]:
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def run_episode(
    env: OffScreenRenderEnv,
    policy,
    task_description: str,
    init_state,
) -> Tuple[bool, int]:
    env.reset()
    obs = env.set_init_state(init_state)
    action_plan: Deque[np.ndarray] = collections.deque()
    steps = 0
    success = False

    while steps < MAX_STEPS + WAIT_STEPS:
        if steps < WAIT_STEPS:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
            if done:
                break
            steps += 1
            continue

        img = _preprocess_image(obs["agentview_image"])
        wrist_img = _preprocess_image(obs["robot0_eye_in_hand_image"])
        state = _format_state(obs)

        if not action_plan:
            inputs = {
                "observation/state": state,
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "prompt": task_description,
            }
            result = policy.infer(inputs)
            chunk = np.asarray(result["actions"])
            if chunk.shape[0] < REPLAN_STEPS:
                raise RuntimeError(
                    f"Policy returned {chunk.shape[0]} actions but replan window is {REPLAN_STEPS}."
                )
            action_plan.extend(chunk[:REPLAN_STEPS])

        action = np.asarray(action_plan.popleft(), dtype=np.float32)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break
        steps += 1

    return success, steps


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    suite_dir = DATA_ROOT / EVALUATION_SUITE
    config = _config.get_config(CONFIG_NAME)
    checkpoint_dir = CHECKPOINT_PATH
    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        evaluation_suite_name=EVALUATION_SUITE,
        data_dir=str(DATA_ROOT),
        pytorch_device="cpu",
    )

    task_suite = benchmark.get_benchmark_dict()[LIBERO_SUITE]()
    rng = np.random.default_rng(0)
    task_id = int(rng.integers(task_suite.n_tasks))
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, description = _get_libero_env(task)

    successes = 0
    try:
        for episode_idx in range(NUM_EPISODES):
            init_state = initial_states[episode_idx % len(initial_states)]
            success, steps = run_episode(env, policy, str(description), init_state)
            print(f"Episode {episode_idx + 1}: success={success} steps={steps}")
            if success:
                successes += 1
            policy.end_trajectory()
    finally:
        env.close()

    print(f"Completed {NUM_EPISODES} episodes on task '{description}'.")
    print(f"Successes: {successes}/{NUM_EPISODES}")
    print(f"Metadata saved under {DATA_ROOT / EVALUATION_SUITE}")


if __name__ == "__main__":
    main()
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
