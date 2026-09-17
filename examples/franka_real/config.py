from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import warnings

if __package__:
    from .repo_paths import checkpoint_dir, openpi_root, evaluation_suite_name, task_prompt
else:
    from repo_paths import checkpoint_dir, openpi_root, evaluation_suite_name, task_prompt


class ModelFamily(str, Enum):
    PI0 = "pi0"
    PI05 = "pi05"


# User-editable paths for Franka inference.
POLICY_CHECKPOINT_DIR = checkpoint_dir
POLICY_NORM_STATS_PATH = (
    str(Path(checkpoint_dir) / "assets" / "franka" / "norm_stats.json")
    if checkpoint_dir is not None else None
)
POLICY_EVALUATION_SUITE_NAME = evaluation_suite_name
# Preferred root directory where latent/action metadata is saved.
POLICY_METADATA_SAVE_DIR_PREFERRED = "/data3/openpi"
# Language instruction used for policy inference.
POLICY_LANGUAGE_INSTRUCTION = task_prompt

## Base task:
# "place the pink block in the bin"
# "place the blue block in the bin"

## On top:
# "place the pink block on top of the blue block"

## Object Two
# "place the pink block in the bin, then place the blue block in the bin"


def _resolve_policy_metadata_dir() -> str:
    override = os.getenv("OPENPI_POLICY_METADATA_DIR")
    if override:
        return override

    preferred = Path(POLICY_METADATA_SAVE_DIR_PREFERRED).expanduser()
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".openpi_write_test"
        probe.touch(exist_ok=True)
        probe.unlink()
        return str(preferred)
    except OSError:
        fallback = (_REPO_ROOT / ".runtime_data" / "openpi").resolve()
        fallback.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            f"Metadata dir {preferred} is not writable; falling back to {fallback}. "
            "Set OPENPI_POLICY_METADATA_DIR to override.",
            stacklevel=2,
        )
        return str(fallback)


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = openpi_root
POLICY_METADATA_SAVE_DIR = _resolve_policy_metadata_dir()


@dataclass(frozen=True)
class PolicyServerConfig:
    # Required by this customized OpenPI fork (see custom_openpi.md).
    evaluation_suite_name: str = POLICY_EVALUATION_SUITE_NAME
    data_dir: str = POLICY_METADATA_SAVE_DIR

    host: str = "0.0.0.0"
    port: int = 8000
    infer_path: str = "/infer"
    metadata_path: str = "/metadata"
    health_path: str = "/health"
    begin_episode_path: str = "/begin_episode"
    end_trajectory_path: str = "/end_trajectory"

    model_family: ModelFamily = ModelFamily.PI05
    checkpoint_dir: str | None = None
    norm_stats_path: str | None = None
    default_prompt: str | None = POLICY_LANGUAGE_INSTRUCTION


@dataclass(frozen=True)
class RobotRuntimeConfig:
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    infer_path: str = "/infer"
    metadata_path: str = "/metadata"
    health_path: str = "/health"
    begin_episode_path: str = "/begin_episode"
    end_trajectory_path: str = "/end_trajectory"
    request_timeout_sec: float = 20.0

    action_horizon: int = 50
    max_hz: float = 20.0
    num_episodes: int = 1
    max_episode_steps: int = 1200
    max_allowed_inferences_per_episode: int = 200
    max_allowed_episode_seconds: float = 120.0
    test_inference_count: int = 40

    prompt: str = POLICY_LANGUAGE_INSTRUCTION

    camera_host: str = "172.16.0.1"
    side_camera_port: str = "10005"
    wrist_camera_port: str = "10006"
    front_camera_port: str = "10007"

    # Match the Franka preprocessing in the existing LeRobot integration.
    mask_front_left_cols: int = 140
    mask_front_right_start: int = 500

    render_height: int = 224
    render_width: int = 224


POLICY_SERVER = PolicyServerConfig(
    evaluation_suite_name=POLICY_EVALUATION_SUITE_NAME,
    data_dir=POLICY_METADATA_SAVE_DIR,
    checkpoint_dir=POLICY_CHECKPOINT_DIR,
    norm_stats_path=POLICY_NORM_STATS_PATH,
    default_prompt=POLICY_LANGUAGE_INSTRUCTION,
)
ROBOT_RUNTIME = RobotRuntimeConfig()
