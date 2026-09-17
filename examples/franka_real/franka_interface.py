import os

if __package__:
    from . import config as _config
    from .repo_paths import use_robot_repositories
else:
    import config as _config
    from repo_paths import use_robot_repositories

use_robot_repositories()

from easydict import EasyDict
import numpy as np
from openteach.components.operators.franka import CONFIG_ROOT
from openteach.components.operators.franka import FrankaArmOperator
from openteach.utils.network import ZMQCameraSubscriber
import yaml


class FrankaInterface:
    """Thin Franka hardware adapter using the same OpenTeach interfaces as LeRobot."""

    def __init__(self, config: _config.RobotRuntimeConfig = _config.ROBOT_RUNTIME) -> None:
        self._config = config
        self._side_subscriber = ZMQCameraSubscriber(
            host=config.camera_host,
            port=config.side_camera_port,
            topic_type="RGB",
        )
        self._wrist_subscriber = ZMQCameraSubscriber(
            host=config.camera_host,
            port=config.wrist_camera_port,
            topic_type="RGB",
        )
        self._front_subscriber = ZMQCameraSubscriber(
            host=config.camera_host,
            port=config.front_camera_port,
            topic_type="RGB",
        )

        with open(os.path.join(CONFIG_ROOT, "network.yaml"), "r", encoding="utf-8") as f:
            network_cfg = EasyDict(yaml.safe_load(f))

        self._operator = FrankaArmOperator(
            network_cfg["host_address"],
            None,
            None,
            None,
            use_filter=False,
            arm_resolution_port=None,
            teleoperation_reset_port=None,
            record="openpi_franka",
            control_mode="absolute_eef_pose_to_delta",
        )

    def is_connected(self) -> bool:
        return self._operator.robot_interface.last_q is not None

    def get_observation(self) -> dict:
        if not self.is_connected():
            raise ConnectionError("Franka interface is not connected (last_q is None).")

        side_img, _ = self._side_subscriber.recv_rgb_image()
        wrist_img, _ = self._wrist_subscriber.recv_rgb_image()
        front_img, _ = self._front_subscriber.recv_rgb_image()

        # OpenTeach camera streams are BGR; convert to RGB.
        side_img = np.copy(side_img[:, :, ::-1])
        wrist_img = np.copy(wrist_img[:, :, ::-1])
        front_img = np.copy(front_img[:, :, ::-1])

        # Keep camera preprocessing identical to the existing Franka stack.
        front_img[:, : self._config.mask_front_left_cols] = 0
        front_img[:, self._config.mask_front_right_start :] = 0

        joint_pos = np.asarray(self._operator.robot_interface.last_q, dtype=np.float32)
        gripper_pos = np.asarray([self._operator.robot_interface.last_gripper_q], dtype=np.float32)

        return {
            "images": {
                "camera_front": front_img,
                "camera_wrist": wrist_img,
                "camera_side": side_img,
            },
            "state": np.concatenate([joint_pos, gripper_pos], axis=0),
        }

    def send_action(self, action: np.ndarray) -> None:
        # Policy output is already unnormalized; padding is not a robot command.
        action = np.asarray(action, dtype=np.float64)
        if action.shape not in ((8,), (32,)):
            raise ValueError(f"Expected action shape (8,) or padded (32,), got {action.shape}")
        action = action[:8].copy()
        if not np.isfinite(action).all():
            raise ValueError("Action contains nonfinite pose or gripper values")
        quaternion_norm = np.linalg.norm(action[3:7])
        if not np.isfinite(quaternion_norm) or quaternion_norm <= 1e-8:
            raise ValueError("Predicted quaternion has zero or invalid norm")
        action[3:7] /= quaternion_norm
        gripper = -1.0 if action[7] < 0.0 else 1.0
        self._operator.arm_control(target_pose=action[:7], gripper_cmd=gripper)
