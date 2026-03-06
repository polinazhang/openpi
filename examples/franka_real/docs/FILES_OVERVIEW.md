# Franka OpenPI Inference Stack: File Overview

This document describes the Franka inference files added under `examples/franka_real` and the one config addition in `src/openpi/training/config.py`.

## Goals of this stack

- Run remote OpenPI inference for Franka while following existing OpenPI conventions (`Runtime`, `WebsocketClientPolicy`, `ActionChunkBroker`).
- Keep robot communication independent from `lerobot`.
- Preserve the known-good Franka observation/action conventions from the previous LeRobot-based stack.
- Support both `pi0` and `pi05` with base checkpoint defaults unless a custom checkpoint path is set.
- Keep user-facing settings in a Python config file rather than CLI arguments.

## Added files

## `examples/franka_real/config.py`

Central, user-editable configuration.

Contains two dataclasses:

- `PolicyServerConfig`
  - `evaluation_suite_name`, `data_dir` (required by this customized OpenPI behavior)
  - `host`, `port`
  - `model_family` (`pi0` or `pi05`)
  - `checkpoint_dir` (optional override)
  - `default_prompt` (optional)

- `RobotRuntimeConfig`
  - policy server endpoint (`policy_host`, `policy_port`)
  - runtime control (`action_horizon`, `max_hz`, episode limits)
  - prompt
  - camera endpoint/ports
  - front-camera masking params
  - render size

Singletons:

- `POLICY_SERVER`
- `ROBOT_RUNTIME`

## `examples/franka_real/franka_interface.py`

Hardware bridge for Franka + cameras using OpenTeach APIs.

Responsibilities:

- Initialize ZMQ camera subscribers for side/wrist/front streams.
- Initialize `FrankaArmOperator` using OpenTeach `network.yaml` host settings.
- Read observations:
  - camera frames
  - joint state (`last_q`)
  - gripper state (`last_gripper_q`)
- Apply camera preprocessing compatible with existing Franka data/runtime:
  - BGR -> RGB conversion
  - front camera column masking (`:140`, `500:` by default)
- Send 8D actions `[x, y, z, quat_x, quat_y, quat_z, quat_w, gripper]` via `arm_control(...)`.

## `examples/franka_real/env.py`

OpenPI-client environment adapter (`openpi_client.runtime.environment.Environment`).

Responsibilities:

- Wrap `FrankaInterface` into runtime-compatible methods:
  - `reset`
  - `is_episode_complete`
  - `get_observation`
  - `apply_action`
- Convert raw camera images into policy-ready images:
  - resize+pad to configured resolution (default 224x224)
  - uint8 format
- Emit OpenArm-style policy input keys:
  - `head_image` (front)
  - `left_wrist_image` (wrist)
  - `right_wrist_image` (side)
  - `state` (7 joints + gripper)
  - `prompt`

## `examples/franka_real/main.py`

Robot-side runtime entrypoint.

Responsibilities:

- Connect to remote policy server (`WebsocketClientPolicy`).
- Wrap policy with `ActionChunkBroker`.
- Build and run `Runtime` with `FrankaRealEnvironment`.
- Use config-only settings from `ROBOT_RUNTIME`.

## `examples/franka_real/serve_policy.py`

Server-side policy serving entrypoint for Franka.

Responsibilities:

- Select config/checkpoint pair by `model_family`:
  - `pi0` -> `pi0_franka_object` + `gs://openpi-assets/checkpoints/pi0_base`
  - `pi05` -> `pi05_franka_object` + `gs://openpi-assets/checkpoints/pi05_base`
- Allow custom checkpoint override via `checkpoint_dir`.
- Create policy with required custom arguments:
  - `evaluation_suite_name`
  - `data_dir`
- Serve over websocket with metadata.

## Modified file

## `src/openpi/training/config.py`

Added config:

- `pi0_franka_object`

Why:

- `pi05_franka_object` already existed.
- `pi0` support required a corresponding train/inference config so `get_config("pi0_franka_object")` works.

## Observation/action schema used by this stack

- Observation state: 8D (`joint1..joint7`, `gripper_pos`)
- Observation images: front/wrist/side
- Policy output consumed: first 8 dims
- Robot command: Cartesian EE pose (7D quaternion pose) + gripper (1D)

## Design notes

- No `lerobot` imports are used.
- Architecture intentionally mirrors existing OpenPI examples (minimal changes).
- All frequent user edits live in `config.py`.
