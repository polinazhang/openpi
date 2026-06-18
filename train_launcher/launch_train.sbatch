#!/bin/bash
# Canonical RoboCasa finetune training sbatch (adapted for the job-manager pipeline).
# This is a LAUNCH SCRIPT (adapting it is permitted by colleague_instructions rule 1);
# the actual training is the unchanged, verified scripts/train.py.
#
# Driven ENTIRELY by env vars set via an explicit --export allowlist by the submitter
# (jm_launch.py for fresh launch, jm_resume.py for resume; or train_launcher/launch.py
# for a manual fresh launch). No --export=ALL. Resource directives (partition, gpus,
# cpus, time, mem, job-name, output) are supplied by the submitter on the sbatch CLI
# and override the conservative #SBATCH defaults below.
#
#SBATCH --job-name=robocasa-finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=kira-lab
#SBATCH --qos=long

set -euo pipefail

# --- required contract (fail-fast; per-user env is parametrized, review point 4) ---
: "${RUN_BASE_CONFIG:?missing RUN_BASE_CONFIG}"
: "${RUN_EXP_NAME:?missing RUN_EXP_NAME}"
: "${RUN_CHECKPOINT_BASE_DIR:?missing RUN_CHECKPOINT_BASE_DIR}"
: "${RUN_DATA_DIRS_JSON:?missing RUN_DATA_DIRS_JSON}"
: "${RUN_START_CHECKPOINT:?missing RUN_START_CHECKPOINT}"
: "${OPENPI_ROOT:?missing OPENPI_ROOT}"
: "${JM_CONDA_SH:?missing JM_CONDA_SH}"
: "${JM_CONDA_ENV:?missing JM_CONDA_ENV}"

# --- optional contract (with defaults) ---
RUN_RESUME="${RUN_RESUME:-0}"
RUN_FSDP_DEVICES="${RUN_FSDP_DEVICES:-8}"
RUN_ASSET_ID="${RUN_ASSET_ID:-}"
RUN_PROJECT_NAME="${RUN_PROJECT_NAME:-openpi}"
# Tier-2 step-count overrides as SEPARATE single ints (no spaces/commas → safe via
# --export). NEVER set in production (preflight refuses go-live if any is set).
RUN_NUM_TRAIN_STEPS="${RUN_NUM_TRAIN_STEPS:-}"
RUN_SAVE_INTERVAL="${RUN_SAVE_INTERVAL:-}"
RUN_KEEP_PERIOD="${RUN_KEEP_PERIOD:-}"

# --- env hygiene (Gate 2) ---
unset DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR SSH_AUTH_SOCK 2>/dev/null || true

cd "${OPENPI_ROOT}"
# shellcheck disable=SC1090
source "${JM_CONDA_SH}"
conda activate "${JM_CONDA_ENV}"
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "=== finetune job context ==="
echo "host=$(hostname) job_id=${SLURM_JOB_ID:-?} gpus=${SLURM_JOB_GPUS:-?}"
echo "config=${RUN_BASE_CONFIG} exp=${RUN_EXP_NAME}"
echo "ckpt_base=${RUN_CHECKPOINT_BASE_DIR} start=${RUN_START_CHECKPOINT}"
echo "resume=${RUN_RESUME} fsdp_devices=${RUN_FSDP_DEVICES} asset_id='${RUN_ASSET_ID}' project=${RUN_PROJECT_NAME}"
echo "step_overrides: num_train_steps='${RUN_NUM_TRAIN_STEPS}' save_interval='${RUN_SAVE_INTERVAL}' keep_period='${RUN_KEEP_PERIOD}'"

# --- assemble train.py args ---
ARGS=(
  "${RUN_BASE_CONFIG}"
  --exp-name="${RUN_EXP_NAME}"
  --checkpoint-base-dir="${RUN_CHECKPOINT_BASE_DIR}"
  --data.assets.assets-dir="${RUN_START_CHECKPOINT}/assets"
  --data.data-dirs-json="${RUN_DATA_DIRS_JSON}"
  --weight-loader.params-path="${RUN_START_CHECKPOINT}/params"
  --fsdp-devices="${RUN_FSDP_DEVICES}"
  --project-name="${RUN_PROJECT_NAME}"
)
# pi05_base writes norm stats under assets/<asset_id>/ — select it explicitly here.
if [[ -n "${RUN_ASSET_ID}" ]]; then
  ARGS+=( --data.assets.asset-id="${RUN_ASSET_ID}" )
fi
# Resume from the latest Orbax checkpoint in the (existing) run dir. A FRESH launch
# targets a new timestamped dir that does not exist, so it needs NO flag (and we never
# pass --overwrite, so a real checkpoint can never be wiped).
if [[ "${RUN_RESUME}" == "1" ]]; then
  ARGS+=( --resume )
fi
# Tier-2 step-count overrides (each a single int; empty in production).
[[ -n "${RUN_NUM_TRAIN_STEPS}" ]] && ARGS+=( --num-train-steps="${RUN_NUM_TRAIN_STEPS}" )
[[ -n "${RUN_SAVE_INTERVAL}" ]] && ARGS+=( --save-interval="${RUN_SAVE_INTERVAL}" )
[[ -n "${RUN_KEEP_PERIOD}" ]] && ARGS+=( --keep-period="${RUN_KEEP_PERIOD}" )

echo "=== launching: python scripts/train.py ${ARGS[*]} ==="
# srun re-imposes SLURM_EXPORT_ENV (the sbatch --export allowlist, which carries the
# submission-time PATH) onto the step, CLOBBERING the conda activate above -> the step's
# `python` reverts to the submitter's PATH (wrong interpreter: no etils/openpi). Unset it
# so srun propagates THIS (already sanitized + conda-activated) batch env to the step.
# Safe: the batch env was built from the --export allowlist (no login-env leak) and D-Bus
# is unset above. Tier-2 finding; reproduced + verified in jm_t2/diag.sbatch.
unset SLURM_EXPORT_ENV
srun -u python -u scripts/train.py "${ARGS[@]}"
