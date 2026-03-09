# Franka OpenPI Stack: File Overview (Current)

This directory now uses a split-process design:

1. OpenPI inference server in uv env.
2. Robot communicator in OpenTeach env.

## Core files

## `examples/franka_real/config.py`

Single place for user configuration.

Important user-editable variables:

- `POLICY_CHECKPOINT_DIR`
- `POLICY_NORM_STATS_PATH`
- `POLICY_EVALUATION_SUITE_NAME`
- `POLICY_METADATA_SAVE_DIR`

`POLICY_SERVER` contains inference server settings.
`ROBOT_RUNTIME` contains robot runtime settings (server endpoint, hz, horizon, camera ports, prompt).

## `examples/franka_real/inference_server.py`

Standalone OpenPI inference server (run in uv env).

- Loads policy from explicit checkpoint path (no default checkpoint fallback).
- Loads norm stats from explicit `norm_stats.json` path.
- Uses custom_openpi-required args:
  - `evaluation_suite_name`
  - `data_dir` (metadata root)
- Serves HTTP endpoints:
  - `GET /health`
  - `GET /metadata`
  - `POST /infer`

## `examples/franka_real/test_inference_server.py`

Standalone server responsiveness test client.

- Calls `/health`.
- Sends synthetic observations to `/infer`.
- Verifies response shape and prints latency stats.

## `examples/franka_real/franka_interface.py`

OpenTeach-based Franka hardware adapter.

- Subscribes to side/wrist/front camera streams.
- Reads joint+gripper state.
- Applies front-camera masking and BGR->RGB conversion.
- Sends 8D Cartesian pose + gripper command.

## `examples/franka_real/robot_communicator.py`

Standalone robot communication runtime (run with OpenTeach python executable).

- Waits for inference server and robot connection.
- Builds policy observations from live cameras/state.
- Requests action chunks from `/infer`.
- Executes one action per control step on Franka.

## Legacy files

These files are from the earlier websocket/openpi-client flow and are not the primary run path now:

- `examples/franka_real/main.py`
- `examples/franka_real/env.py`
- `examples/franka_real/serve_policy.py`

## External launcher

New 5-process launcher (outside this repo):

- `~/openteach/franka_openpi_eval.bash`

It launches:

1. camera launcher
2. arm controller launcher
3. gripper controller launcher
4. OpenPI inference server (`TORCHDYNAMO_DISABLE=1`)
5. robot communicator
