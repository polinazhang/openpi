# Franka Inference: Setup and Run Guide

This guide explains how to run OpenPI inference on Franka using the files in `examples/franka_real`.

## 1) Prerequisites

You need two Python environments/machines (can be the same machine if resources allow):

- Policy server environment (GPU recommended)
  - OpenPI installed
- Robot control environment (connected to Franka + cameras)
  - OpenPI client runtime + OpenTeach dependencies available

You also need:

- OpenTeach network/camera stack running and reachable.
- Franka camera ZMQ streams reachable at configured host/ports.
- Franka operator connection working (`robot_interface.last_q` should become non-`None`).

## 2) Install OpenPI (customized repo behavior)

From `openpi/`:

```bash
pip install -e . --no-deps
```

This follows your custom instruction file (`custom_openpi.md`).

## 3) Configure settings

Edit:

- `examples/franka_real/config.py`

### Server settings (`POLICY_SERVER`)

- `evaluation_suite_name`: required metadata suite name.
- `data_dir`: required metadata root directory.
- `model_family`: choose `ModelFamily.PI0` or `ModelFamily.PI05`.
- `checkpoint_dir`: optional custom checkpoint path. If `None`, defaults are:
  - PI0: `gs://openpi-assets/checkpoints/pi0_base`
  - PI05: `gs://openpi-assets/checkpoints/pi05_base`

### Robot runtime settings (`ROBOT_RUNTIME`)

- Policy endpoint: `policy_host`, `policy_port`
- Prompt: `prompt`
- Control loop/horizon: `action_horizon`, `max_hz`, episode limits
- Camera routing:
  - `camera_host`
  - `side_camera_port`, `wrist_camera_port`, `front_camera_port`
- Front camera masking:
  - `mask_front_left_cols`
  - `mask_front_right_start`

## 4) Start policy server

From `openpi/` (server machine):

```bash
python -m examples.franka_real.serve_policy
```

Server behavior:

- Loads config from `POLICY_SERVER`.
- Creates trained policy using required custom args (`evaluation_suite_name`, `data_dir`).
- Serves over websocket on configured host/port.

## 5) Start Franka runtime client

From `openpi/` (robot machine):

```bash
python -m examples.franka_real.main
```

Client behavior:

- Connects to policy websocket server.
- Streams Franka observations (state + images).
- Receives action chunks and executes one action at a time via `ActionChunkBroker`.

## 6) Data/IO conventions used

Observation sent to policy:

- `head_image` <- front camera
- `left_wrist_image` <- wrist camera
- `right_wrist_image` <- side camera
- `state` <- `[joint1..joint7, gripper_pos]`
- `prompt`

Action applied to robot:

- `actions[0:8]` interpreted as:
  - `[x, y, z, quat_x, quat_y, quat_z, quat_w, gripper]`

Image preprocessing:

- BGR -> RGB conversion
- front camera horizontal masking (same as prior Franka stack)
- resize/pad to configured size (default `224x224`)
- uint8 conversion

## 7) Troubleshooting

## `ConnectionError: Franka interface is not connected (last_q is None).`

- OpenTeach Franka operator is not fully connected.
- Check OpenTeach network config and robot process state.

## No camera images / blocked inference

- Verify `camera_host` and ports in `config.py`.
- Verify ZMQ camera publishers are up.

## Server reachable but actions fail shape check

- Franka client expects 8D action vectors.
- Ensure you are serving a Franka-compatible config (`pi0_franka_object` or `pi05_franka_object`) or compatible custom checkpoint/config.

## Prompt not affecting behavior

- Ensure `ROBOT_RUNTIME.prompt` is non-empty.
- If using server-side default prompt, set `POLICY_SERVER.default_prompt`.

## 8) Minimal run checklist

1. Edit `examples/franka_real/config.py`.
2. Start server: `python -m examples.franka_real.serve_policy`.
3. Start robot client: `python -m examples.franka_real.main`.
4. Confirm websocket connection and incoming actions.
5. Confirm robot moves with valid 8D EE+gripper commands.
