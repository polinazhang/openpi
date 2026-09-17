# Franka inference and robot communication

This workflow uses two existing environments: OpenPI's `.venv` for inference and
OpenTeach's `.conda-env` for hardware communication. Training is unchanged.

## Configure paths

Edit `repo-configs/config.py` in OpenPI:

- `openteach_root`: absolute path of the OpenTeach checkout.
- `checkpoint_dir`: absolute path of the Cartesian-tag Pi0.5 checkpoint. It is
  initially `None`; inference and the combined launcher refuse to start until set.
- `openpi_root`: derived from the config file location.
- `task_prompt`: instruction sent to the policy.
- `evaluation_suite_name`: evaluation output folder name.

In OpenTeach's `repo-configs/config.py`, set `openpi_root` to this OpenPI checkout.
The robot client locates `franka-control` through OpenTeach's configured `repo_root`.
Keep that local checkout on `main`; the `nuc` branch is a source backup.

Normalization is read ONLY from:

```
<checkpoint_dir>/assets/franka/norm_stats.json
```

No default/base checkpoint, training assets, alternate normalization file, or
recomputed statistics are used. Missing/invalid statistics stop startup. The
loader rejects a stats symlink pointing outside the selected checkpoint, checks
finite eight-value state/action statistics and quantiles, and records the source
path and SHA-256 in logs/policy metadata. Standard policy output transforms apply
quantile unnormalization once; the robot client does not apply it again.

## Run

From the OpenPI checkout (the wrapper also works from any cwd):

```bash
bash repo-configs/run_franka.bash server
bash repo-configs/run_franka.bash robot
```

Run these in separate terminals, with cameras and the existing NUC controllers
available. The wrapper selects the appropriate Python directly and removes an
inherited `PYTHONPATH` so unrelated ROS/user checkouts cannot supply imports.
`TORCHDYNAMO_DISABLE=1` retains the existing workstation inference setting.

For the combined five-process launcher, from the OpenTeach checkout:

```bash
bash franka_openpi_eval.bash
```

It reads `openpi_root`, validates the two configs agree and the checkpoint stats
exist before opening terminals, starts cameras/existing NUC controllers/inference,
then starts communication after server health responds. The NUC installation is
unchanged. The optional second argument `test` executes real robot actions; it is
not an offline test.

Task prompt and evaluation suite are in `repo-configs/config.py`.
Metadata destination, camera settings and episode limits remain in
`examples/franka_real/config.py`. Metadata keeps its existing
configured destination/fallback; datasets and caches are not moved.

## Supported checkpoint contract

Only Cartesian-tag Pi0.5: action dimension 32, chunk length 50, max token length 200.
The inference-only config leaves training definitions unchanged. No action-state
subtraction or delta-to-absolute output transform is applied.

After unnormalization, the first eight values are absolute
`[x, y, z, qx, qy, qz, qw, gripper]`; remaining coordinates are padding.
The configured robot frame, position units and tool frame are assumed to match the
recorder, as confirmed for this workflow. Joint-tag and delta checkpoints are not
supported by this adapter.

The client executes 50 actions at a target 20 Hz before requesting another chunk;
inference latency adds a pause between chunks. No targets are cumulatively summed.
Nonfinite robot commands and zero/near-zero quaternions are rejected. Quaternions
are normalized; gripper values below zero map to -1 (open), otherwise +1 (close).
OpenTeach's `absolute_eef_pose_to_delta` computes errors against current measured
pose and uses its current rotation/translation clipping and OSC_POSE configuration.
Measured input state remains seven joint angles plus measured gripper position.
Existing BGR-to-RGB conversion, front-camera mask, and front/wrist/side mapping are
preserved. No claim of checkpoint quality or hardware rollout validation is made.

## Environment audit

The existing `.venv` was retained. Against this checkout's installation guide:
Python 3.11.15, torch 2.7.1, JAX 0.5.3, transformers 4.53.2, editable local OpenPI
and openpi-client, and the pinned LeRobot Git revision matched. All five documented
Transformers replacement Python files matched byte-for-byte. `uv pip check`
passed. Installed distributions represented in `uv.lock` had matching versions;
extra distributions were pip 24.0, pyserial 3.5, and accelerate 1.13.0. No packages
were installed, removed, or upgraded. This audit does not prove byte-for-byte
identity with a fresh environment. The uv-managed base Python and standard cache
locations remain external, as permitted.

After relocating the checkout, follow OpenPI's root README installation steps to
refresh editable package/environment paths; moving `.venv` is not a portable
installation procedure.

## Offline verification

From OpenPI:

```bash
PYTHONPATH= OPENPI_POLICY_METADATA_DIR=/tmp/openpi-test-metadata JAX_PLATFORMS=cpu \
  .venv/bin/python repo-configs/test_checkpoint_contract.py
```

Run `repo-configs/test_robot_adapter.py` using the configured OpenTeach environment.
It mocks camera subscribers and the robot operator. RealSense import still needs
OS udev access. Neither test loads model weights or sends robot commands.
