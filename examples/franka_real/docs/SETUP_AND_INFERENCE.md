# Franka Inference: Quick Run Guide

This is the updated split-environment flow:

- OpenPI inference server runs in the OpenPI uv env.
- Robot communicator runs in the OpenTeach env (`/home/jeremiah/miniforge3/envs/openteach/bin/python`).

## 1) Configure paths (required)

Edit:

- `examples/franka_real/config.py`

Main user-editable variables:

- `POLICY_CHECKPOINT_DIR`
- `POLICY_NORM_STATS_PATH`
- `POLICY_EVALUATION_SUITE_NAME`
- `POLICY_METADATA_SAVE_DIR` (custom_openpi `data_dir`, where metadata/latents are saved)

Current defaults are already set to your local paths.

## 2) Start robot-side dependencies (3 essential processes)

You need these running (same as identified from OpenTeach launcher):

1. camera launcher: `robot_camera.py --config-name=camera`
2. arm controller launcher: `auto_arm.sh`
3. gripper controller launcher: `auto_gripper.sh`

## 3) Start inference server (OpenPI uv env)

```bash
cd /home/ripl/openpi
source /home/ripl/openpi/.venv/bin/activate
TORCHDYNAMO_DISABLE=1 python /home/ripl/openpi/examples/franka_real/inference_server.py
```

Why `TORCHDYNAMO_DISABLE=1`:

- On this machine (RTX 2080 Ti), `torch.compile`/Triton can fail at first inference with bf16/f16 compile errors.
- Disabling Dynamo avoids that compile path and keeps GPU inference working.

## 4) (Optional but recommended) test server responsiveness

In another terminal:

```bash
cd /home/ripl/openpi
source /home/ripl/openpi/.venv/bin/activate
python /home/ripl/openpi/examples/franka_real/test_inference_server.py --host 127.0.0.1 --port 8000 --num-requests 3 --timeout 180
```

## 5) Start robot communicator (OpenTeach env python)

```bash
cd /home/ripl/openpi
/home/jeremiah/miniforge3/envs/openteach/bin/python /home/ripl/openpi/examples/franka_real/robot_communicator.py
```

## 6) One-command launcher for all 5 processes

You can launch all required panes (3 OpenTeach processes + inference server + robot communicator) with:

```bash
~/openteach/franka_openpi_eval.bash
```

Optional NUC override:

```bash
~/openteach/franka_openpi_eval.bash 172.16.0.3
```

## 7) Data IO conventions

Observation sent to policy:

- `head_image` <- front camera
- `left_wrist_image` <- wrist camera
- `right_wrist_image` <- side camera
- `state` <- `[joint1..joint7, gripper_pos]`
- `prompt`

Action applied to robot:

- first 8 dims: `[x, y, z, quat_x, quat_y, quat_z, quat_w, gripper]`

## Troubleshooting

`ModuleNotFoundError: examples.franka_real` from communicator:

- Use latest files where local imports are fixed (`import config`, `import franka_interface`).

`transformers_replace is not installed correctly`:

```bash
cd /home/ripl/openpi
source /home/ripl/openpi/.venv/bin/activate
TRANSFORMERS_DIR=$(python - <<'PY'
import pathlib, transformers
print(pathlib.Path(transformers.__file__).resolve().parent)
PY
)
cp -r src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR"/
```

Server starts but first inference crashes with Triton bf16/f16 error:

- Confirm server was started with `TORCHDYNAMO_DISABLE=1`.
