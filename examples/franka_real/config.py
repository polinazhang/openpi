from dataclasses import dataclass
from enum import Enum


class ModelFamily(str, Enum):
    PI0 = "pi0"
    PI05 = "pi05"


@dataclass(frozen=True)
class PolicyServerConfig:
    # Required by this customized OpenPI fork (see custom_openpi.md).
    evaluation_suite_name: str = "franka_eval"
    data_dir: str = "/tmp/openpi_metadata"

    host: str = "0.0.0.0"
    port: int = 8000

    model_family: ModelFamily = ModelFamily.PI05
    checkpoint_dir: str | None = None
    default_prompt: str | None = None


@dataclass(frozen=True)
class RobotRuntimeConfig:
    policy_host: str = "0.0.0.0"
    policy_port: int = 8000

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


POLICY_SERVER = PolicyServerConfig()
ROBOT_RUNTIME = RobotRuntimeConfig()
