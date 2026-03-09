"""Standalone Franka robot communication client.

Run this script in the OpenTeach environment as:
    /home/jeremiah/miniforge3/envs/openteach/bin/python /home/ripl/openpi/examples/franka_real/robot_communicator.py
"""

from __future__ import annotations

import collections
import json
import logging
import pathlib
import pickle
import sys
import time
from typing import Any

import cv2
import numpy as np
import requests


_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config as _config
import franka_interface as _franka_interface


logger = logging.getLogger(__name__)


def _ensure_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image with 3 dims, got shape {image.shape}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 255.0)
        else:
            image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
    return image


def _resize_with_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    image = _ensure_uint8_hwc(image)
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape {image.shape}")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _build_policy_observation(raw_obs: dict[str, Any], cfg: _config.RobotRuntimeConfig) -> dict[str, Any]:
    front = _resize_with_pad(raw_obs["images"]["camera_front"], cfg.render_height, cfg.render_width)
    wrist = _resize_with_pad(raw_obs["images"]["camera_wrist"], cfg.render_height, cfg.render_width)
    side = _resize_with_pad(raw_obs["images"]["camera_side"], cfg.render_height, cfg.render_width)

    return {
        "head_image": front,
        "left_wrist_image": wrist,
        "right_wrist_image": side,
        "state": np.asarray(raw_obs["state"], dtype=np.float32),
        "prompt": cfg.prompt,
    }


class PolicyHttpClient:
    def __init__(self, cfg: _config.RobotRuntimeConfig) -> None:
        self._cfg = cfg
        self._base_url = f"http://{cfg.policy_host}:{cfg.policy_port}"

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def get_health(self) -> dict[str, Any]:
        resp = requests.get(self._url(self._cfg.health_path), timeout=self._cfg.request_timeout_sec)
        resp.raise_for_status()
        return resp.json()

    def get_metadata(self) -> dict[str, Any]:
        resp = requests.get(self._url(self._cfg.metadata_path), timeout=self._cfg.request_timeout_sec)
        resp.raise_for_status()
        return pickle.loads(resp.content)

    def infer(self, observation: dict[str, Any], request_id: int) -> tuple[Any, dict[str, Any]]:
        payload = pickle.dumps(
            {"request_id": request_id, "observation": observation},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        resp = requests.post(
            self._url(self._cfg.infer_path),
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._cfg.request_timeout_sec,
        )

        if resp.status_code >= 400:
            msg = resp.text
            try:
                msg = json.dumps(resp.json())
            except Exception:
                pass
            raise RuntimeError(f"Inference server returned HTTP {resp.status_code}: {msg}")

        data = pickle.loads(resp.content)
        if not isinstance(data, dict) or not data.get("ok", False):
            raise RuntimeError(f"Inference response is invalid: {data}")
        return data["action"], data.get("server_timing", {})


def _action_queue_from_response(action_payload: Any, action_horizon: int) -> collections.deque[np.ndarray]:
    if isinstance(action_payload, dict) and "actions" in action_payload:
        actions = np.asarray(action_payload["actions"], dtype=np.float32)
    else:
        actions = np.asarray(action_payload, dtype=np.float32)

    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"Expected action payload with 1D or 2D shape, got {actions.shape}")
    if actions.shape[1] < 8:
        raise ValueError(f"Expected action dim >= 8, got {actions.shape}")

    horizon = max(1, min(action_horizon, actions.shape[0]))
    queue: collections.deque[np.ndarray] = collections.deque()
    for i in range(horizon):
        queue.append(np.asarray(actions[i, :8], dtype=np.float32))
    return queue


def _wait_for_robot_connection(robot: _franka_interface.FrankaInterface) -> None:
    logger.info("Waiting for Franka robot connection...")
    while not robot.is_connected():
        time.sleep(0.2)
    logger.info("Franka robot is connected.")


def _wait_for_server(client: PolicyHttpClient) -> None:
    logger.info("Waiting for inference server at %s...", client._base_url)
    while True:
        try:
            health = client.get_health()
            logger.info("Inference server health: %s", health)
            metadata = client.get_metadata()
            logger.info("Inference server metadata: %s", metadata)
            return
        except Exception as exc:
            logger.info("Inference server not ready yet (%s). Retrying...", exc)
            time.sleep(1.0)


def run(cfg: _config.RobotRuntimeConfig = _config.ROBOT_RUNTIME) -> None:
    if cfg.max_hz <= 0:
        raise ValueError(f"max_hz must be > 0, got {cfg.max_hz}")

    client = PolicyHttpClient(cfg)
    _wait_for_server(client)

    robot = _franka_interface.FrankaInterface(cfg)
    _wait_for_robot_connection(robot)

    dt = 1.0 / cfg.max_hz
    request_id = 0

    for episode_idx in range(cfg.num_episodes):
        logger.info("Starting episode %d/%d", episode_idx + 1, cfg.num_episodes)
        action_queue: collections.deque[np.ndarray] = collections.deque()
        next_tick = time.monotonic()

        for step_idx in range(cfg.max_episode_steps):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick = max(next_tick + dt, time.monotonic())

            if not action_queue:
                raw_obs = robot.get_observation()
                obs = _build_policy_observation(raw_obs, cfg)
                action_payload, timing = client.infer(obs, request_id=request_id)
                request_id += 1
                action_queue = _action_queue_from_response(action_payload, cfg.action_horizon)
                logger.info(
                    "step=%d request_id=%d chunk=%d infer_ms=%s",
                    step_idx,
                    request_id - 1,
                    len(action_queue),
                    timing.get("infer_ms"),
                )

            action = action_queue.popleft()
            robot.send_action(action)

        logger.info("Episode %d complete.", episode_idx + 1)

    logger.info("Completed all episodes.")


def main() -> None:
    run(_config.ROBOT_RUNTIME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
