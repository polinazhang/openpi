"""Standalone Franka robot communication client.

Run in OpenTeach env:
    /home/jeremiah/miniforge3/envs/openteach/bin/python /home/ripl/openpi/examples/franka_real/robot_communicator.py
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import pickle
import select
import sys
import termios
import time
import tty
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


def _log(msg: str) -> None:
    print(msg, flush=True)


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
    return {
        "head_image": _resize_with_pad(raw_obs["images"]["camera_front"], cfg.render_height, cfg.render_width),
        "left_wrist_image": _resize_with_pad(raw_obs["images"]["camera_wrist"], cfg.render_height, cfg.render_width),
        "right_wrist_image": _resize_with_pad(raw_obs["images"]["camera_side"], cfg.render_height, cfg.render_width),
        "state": np.asarray(raw_obs["state"], dtype=np.float32),
        "prompt": cfg.prompt,
    }


class _KeyPressMonitor:
    def __init__(self) -> None:
        self._fd = None
        self._old_settings = None
        self._enabled = False

    def __enter__(self) -> "_KeyPressMonitor":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._enabled = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled and self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        self._enabled = False

    def is_pressed(self) -> bool:
        if not self._enabled:
            return False
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return False
        _ = sys.stdin.read(1)
        return True


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

    def begin_episode(self, evaluation_suite_name: str) -> None:
        resp = requests.post(
            self._url(self._cfg.begin_episode_path),
            data=json.dumps({"evaluation_suite_name": evaluation_suite_name}),
            headers={"Content-Type": "application/json"},
            timeout=self._cfg.request_timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok", False):
            raise RuntimeError(f"begin_episode failed: {data}")

    def end_trajectory(self) -> None:
        resp = requests.post(
            self._url(self._cfg.end_trajectory_path),
            data=b"",
            headers={"Content-Type": "application/json"},
            timeout=self._cfg.request_timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok", False):
            raise RuntimeError(f"end_trajectory failed: {data}")

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


def _wait_for_server(client: PolicyHttpClient) -> None:
    _log(f"Waiting for inference server at {client._base_url}...")
    while True:
        try:
            health = client.get_health()
            metadata = client.get_metadata()
            _log(f"Inference server health: {health}")
            _log(f"Inference server metadata: {metadata}")
            return
        except Exception as exc:
            _log(f"Inference server not ready yet ({exc}). Retrying...")
            time.sleep(1.0)


def _wait_for_robot_connection(robot: _franka_interface.FrankaInterface) -> None:
    _log("Waiting for Franka robot connection...")
    while not robot.is_connected():
        time.sleep(0.2)
    _log("Franka robot is connected.")


def _wait_for_reset_confirmation() -> None:
    while True:
        answer = input("Please reset the robot. Has the reset finished? y/N ").strip().lower()
        if answer == "y":
            return


def _next_episode_suite_path(base_data_dir: pathlib.Path, suite_name: str) -> str:
    suite_dir = base_data_dir / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    existing_subdirs = [p for p in suite_dir.iterdir() if p.is_dir()]
    episode_idx = len(existing_subdirs)
    episode_dir = suite_dir / f"{episode_idx:04d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    return f"{suite_name}/{episode_idx:04d}"


def _run_one_episode(
    robot: _franka_interface.FrankaInterface,
    client: PolicyHttpClient,
    cfg: _config.RobotRuntimeConfig,
    *,
    request_id_start: int,
    max_inferences: int,
    max_episode_seconds: float,
    stop_on_keypress: bool,
) -> tuple[int, str, int]:
    dt = 1.0 / cfg.max_hz
    next_tick = time.monotonic()
    start_time = time.monotonic()
    request_id = request_id_start
    inference_count = 0
    action_queue: collections.deque[np.ndarray] = collections.deque()
    stop_reason = "max_episode_steps"

    with _KeyPressMonitor() as key_monitor:
        for step_idx in range(cfg.max_episode_steps):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick = max(next_tick + dt, time.monotonic())

            elapsed = time.monotonic() - start_time
            if elapsed >= max_episode_seconds:
                stop_reason = "max_allowed_episode_seconds"
                break
            if stop_on_keypress and key_monitor.is_pressed():
                stop_reason = "key_press"
                break

            if not action_queue:
                if inference_count >= max_inferences:
                    stop_reason = "max_allowed_inferences_per_episode"
                    break

                raw_obs = robot.get_observation()
                obs = _build_policy_observation(raw_obs, cfg)
                action_payload, timing = client.infer(obs, request_id=request_id)
                request_id += 1
                inference_count += 1
                action_queue = _action_queue_from_response(action_payload, cfg.action_horizon)
                _log(
                    f"step={step_idx} request_id={request_id - 1} infer_count={inference_count} "
                    f"chunk={len(action_queue)} infer_ms={timing.get('infer_ms')}"
                )

            action = action_queue.popleft()
            robot.send_action(action)

        else:
            stop_reason = "max_episode_steps"

    return request_id, stop_reason, inference_count


def run_eval_mode(cfg: _config.RobotRuntimeConfig) -> None:
    if cfg.max_hz <= 0:
        raise ValueError(f"max_hz must be > 0, got {cfg.max_hz}")

    client = PolicyHttpClient(cfg)
    _wait_for_server(client)

    robot = _franka_interface.FrankaInterface(cfg)
    _wait_for_robot_connection(robot)

    request_id = 0
    base_data_dir = pathlib.Path(_config.POLICY_METADATA_SAVE_DIR).expanduser()
    suite_name = _config.POLICY_EVALUATION_SUITE_NAME

    while True:
        episode_suite_path = _next_episode_suite_path(base_data_dir, suite_name)
        _log(f"Starting episode with metadata suite: {episode_suite_path}")
        client.begin_episode(episode_suite_path)

        request_id, stop_reason, inference_count = _run_one_episode(
            robot,
            client,
            cfg,
            request_id_start=request_id,
            max_inferences=cfg.max_allowed_inferences_per_episode,
            max_episode_seconds=cfg.max_allowed_episode_seconds,
            stop_on_keypress=True,
        )

        # Required by custom_openpi behavior: finalize trajectory artifacts.
        client.end_trajectory()
        _log(
            f"Episode finished: reason={stop_reason}, "
            f"inferences={inference_count}, episode_suite={episode_suite_path}"
        )
        _wait_for_reset_confirmation()


def run_test_mode(cfg: _config.RobotRuntimeConfig) -> None:
    if cfg.max_hz <= 0:
        raise ValueError(f"max_hz must be > 0, got {cfg.max_hz}")

    client = PolicyHttpClient(cfg)
    _wait_for_server(client)

    robot = _franka_interface.FrankaInterface(cfg)
    _wait_for_robot_connection(robot)

    test_dir = pathlib.Path(_config.POLICY_METADATA_SAVE_DIR).expanduser() / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    client.begin_episode("test")
    _log("Starting test mode episode with metadata suite: test")

    _, stop_reason, inference_count = _run_one_episode(
        robot,
        client,
        cfg,
        request_id_start=0,
        max_inferences=cfg.test_inference_count,
        max_episode_seconds=cfg.max_allowed_episode_seconds,
        stop_on_keypress=False,
    )

    client.end_trajectory()
    _log(
        f"Test mode finished: reason={stop_reason}, "
        f"inferences={inference_count}, metadata_suite=test"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode: execute exactly test_inference_count inferences then stop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test:
        run_test_mode(_config.ROBOT_RUNTIME)
    else:
        run_eval_mode(_config.ROBOT_RUNTIME)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("Interrupted by user.")
