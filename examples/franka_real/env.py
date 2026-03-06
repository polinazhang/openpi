from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.franka_real import config as _config
from examples.franka_real import franka_interface as _franka_interface


class FrankaRealEnvironment(_environment.Environment):
    """Real Franka environment that emits OpenArm-style observations for OpenPI."""

    def __init__(self, config: _config.RobotRuntimeConfig = _config.ROBOT_RUNTIME) -> None:
        self._config = config
        self._robot = _franka_interface.FrankaInterface(config)

    @override
    def reset(self) -> None:
        # Franka reset behavior is controlled by the OpenTeach operator setup.
        if not self._robot.is_connected():
            raise ConnectionError("Franka robot not connected (robot_interface.last_q is None).")

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        obs = self._robot.get_observation()

        front = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                obs["images"]["camera_front"], self._config.render_height, self._config.render_width
            )
        )
        wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                obs["images"]["camera_wrist"], self._config.render_height, self._config.render_width
            )
        )
        side = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["images"]["camera_side"], self._config.render_height, self._config.render_width)
        )

        return {
            "head_image": front,
            "left_wrist_image": wrist,
            "right_wrist_image": side,
            "state": obs["state"],
            "prompt": self._config.prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._robot.send_action(action["actions"])
