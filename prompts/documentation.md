# Documentation of the Static/Dynamic Inference Modes in OpenPI

# Schedule

**Training (JAX)**

Time sampling in training loss uses:

`time = Beta(1.5, 1.0) * 0.999 + 0.001`

- `src/openpi/models/pi0.py:197`

Then:

- `x_t = time * noise + (1 - time) * actions` (`src/openpi/models/pi0.py:199`)
- `u_t = noise - actions` (`src/openpi/models/pi0.py:200`)

**PyTorch static mode**

Time sampling uses the same distribution/scaling:

- `sample_time`: `src/openpi/models_pytorch/pi0_pytorch.py:182-185`
- `x_t` and `u_t`: `src/openpi/models_pytorch/pi0_pytorch.py:513-514`

**Dynamic inference rollout**

Rollout starts at `time = 1.0` and integrates with fixed Euler steps `dt = -1/num_steps`:

- `src/openpi/models_pytorch/pi0_pytorch.py:403-420`


# Static Inference

## Implementation

Main files:

- `openarm/static_inference.py`
- `openarm/static_print.py`
- `src/openpi/models_pytorch/pi0_pytorch.py` (`compute_static_targets`)

Main logic:

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

1. `compute_static_targets(...)` returns:
   - `vt_layers` (per action-expert layer projected velocity),
   - `target = u_t = noise - actions`,
   - `final_prediction`.
   - `src/openpi/models_pytorch/pi0_pytorch.py:497-559`
2. `openarm/static_inference.py` computes, per layer:
   - `cosine`,
   - `scale alpha`,
   - `scaled_norm` (EDR).
   - `openarm/static_inference.py:191-203`, `366-372`
3. Final layer raw residual norm is computed as:
   - `||final_prediction - target||_2`
   - `openarm/static_inference.py:374-375`

## Storage Layout and Format

Saved under:

- `<output_root>/<dataset>/<trajectory_id>/npy-metadata/`
- path construction: `openarm/static_inference.py:235-237`

Artifacts written:

- `static_edr_layer_{ll:02d}.npy`
- `static_cosine_layer_{ll:02d}.npy`
- `static_final_loss.npy`
- writing: `openarm/static_inference.py:261-265`

Format:

- saved as float16 `.npy`: `openarm/static_inference.py:247`
- metadata entry includes:
  - `artifacts`
  - `artifact_shapes`
  - `artifact_lengths`
  - `artifact_spans`
  - `openarm/static_inference.py:267-278`
- all trajectories summarized in `metadata.json`:
  - `openarm/static_inference.py:409-411`

Meaning:

- `static_edr_layer_*`: per-sample `||alpha*v_t^{(l)} - u_t||_2`
- `static_cosine_layer_*`: per-sample cosine between `v_t^{(l)}` and `u_t`
- `static_final_loss`: per-sample final-layer `||v_final - u_t||_2`

## How to Run

Compute and save static metrics:

- `python openarm/static_inference.py --dataset <dataset> --output-root <out> --checkpoint-dir <ckpt>`

Print aggregate mean/var/std from saved `static_*` arrays:

- `python openarm/static_print.py --data-dir <out>/<dataset>`

`static_print.py` is summary-only (it does not recompute from `vt`/noise):

- `openarm/static_print.py:93-113`, `125-147`


# Dynamic Inference

## Implementation

Main files:

- `src/openpi/models_pytorch/pi0_pytorch.py`
- `src/openpi/policies/policy.py`
- `src/openpi/policies/metadata_logger.py`
- post-hoc metrics: `openarm/convert_hd5.py`, `openarm/compute.py`

Main logic:

1. During `sample_actions`, model records:
   - `extra:diffusion_noise`
   - `extra:predicted_action_chunk`
   - `src/openpi/models_pytorch/pi0_pytorch.py:383`, `421`
2. Activation hook records per-layer projected velocity:
   - `action_expert_vt -> vt_layer_{k}`
   - `src/openpi/models_pytorch/pi0_pytorch.py:472-481`
3. Policy capture maps/collects these tensors per step:
   - `src/openpi/policies/policy.py:48-61`, `151-155`
4. Metadata logger stacks per-step arrays to trajectory arrays and appends `metadata.json`:
   - `src/openpi/policies/metadata_logger.py:39-77`

## Storage Layout and Format

Dynamic mode saves raw latent artifacts first, not EDR:

- `vt_layer_{k}.npy`
- `diffusion_noise.npy`
- `actions.npy` (from predicted action chunk)

Where:

- `<data_dir>/<evaluation_suite>/<trajectory_id>/npy-metadata/*.npy`
- plus `<data_dir>/<evaluation_suite>/metadata.json`

How:

- each step saved as float16 `.npy`: `src/openpi/policies/metadata_logger.py:34-37`
- trajectory-level stacking at end: `src/openpi/policies/metadata_logger.py:62-67`

Meaning:

- `vt_layer_k`: projected velocity at layer `k`
- `diffusion_noise`: sampled initial noise
- `actions`: rollout predicted action chunk

## How to Run

Step 1: run evaluation/policy inference with metadata logging enabled (this produces `metadata.json` + per-trajectory `.npy` artifacts).

Step 2: convert to HD5:

- `python openarm/convert_hd5.py --source_dir <suite_dir> --target_dir <suite_hd5_dir>`

Step 3: compute post-hoc EDR/cosine:

- `python openarm/compute.py --data_dir <suite_hd5_dir>`

`openarm/compute.py` computes from saved `vt_layer`, `diffusion_noise`, and `actions`:

- `target = diffusion - actions`: `openarm/compute.py:107-110`
- cosine / scale / scaled_norm(EDR): `openarm/compute.py:75-97`
- per-layer output files:
  - `norm_layer_{k}.h5`
  - `scaled_norm_layer_{k}.h5`
  - `cosine_layer_{k}.h5`
  - `scale_layer_{k}.h5`
  - `openarm/compute.py:117-120`


# Key Distinction (Static vs Dynamic)

- **Static mode** computes and saves EDR/cosine directly as `static_*` artifacts.
- **Dynamic mode** saves raw tensors (`vt_layer`, `diffusion_noise`, `actions`) during rollout, then computes EDR/cosine later via `openarm/compute.py`.
