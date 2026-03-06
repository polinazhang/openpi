import os

from easydict import EasyDict
import numpy as np
from openteach.components.operators.franka import CONFIG_ROOT
from openteach.components.operators.franka import FrankaArmOperator
from openteach.utils.network import ZMQCameraSubscriber
import yaml

from examples.franka_real import config as _config


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
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (8,):
            raise ValueError(f"Expected action shape (8,), got {action.shape}")

        abs_eef_pose = action[:7].tolist()
        gripper = float(action[7])
        self._operator.arm_control(abs_eef_pose, gripper)
