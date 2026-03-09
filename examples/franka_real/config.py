from dataclasses import dataclass
from enum import Enum


class ModelFamily(str, Enum):
    PI0 = "pi0"
    PI05 = "pi05"


# User-editable paths for Franka inference.
POLICY_CHECKPOINT_DIR = "/home/ripl/openpi/checkpoints/franka_base_torch/30000"
POLICY_NORM_STATS_PATH = "/home/ripl/openpi/checkpoints/franka_base_torch/30000/norm_stats.json"
POLICY_EVALUATION_SUITE_NAME = "franka_eval"
# custom_openpi.md `data_dir`: root directory where latent/action metadata is saved.
POLICY_METADATA_SAVE_DIR = "/data3/openpi"


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

    model_family: ModelFamily = ModelFamily.PI05
    checkpoint_dir: str | None = None
    norm_stats_path: str | None = None
    default_prompt: str | None = None


@dataclass(frozen=True)
class RobotRuntimeConfig:
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    infer_path: str = "/infer"
    metadata_path: str = "/metadata"
    health_path: str = "/health"
    request_timeout_sec: float = 20.0

    action_horizon: int = 10
    max_hz: float = 20.0
    num_episodes: int = 1
    max_episode_steps: int = 1200

    prompt: str = ""

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
)
ROBOT_RUNTIME = RobotRuntimeConfig()
