# RoboCasa Finetuning: Standalone Implementation Plan

# Important Caveat

The separate openpi / robocasa environment requirement is not in place right now. The code only uses the robocasa environment, and openpi is installed in robocasa separately. This works cleaner. If any later requirements mentioned maintaining two different environments, ignore those requirements.

The registry separation is still in place, so the launch script still needs to be run within the robocasa env. This is to ensure minimal adjustment to the original data loading structure to support task-level individual finetuning jobs, not purely for environment separation.

## 1) Objective
Implement a clean, reproducible RoboCasa finetuning launcher stack in this `openpi` fork with strict validation, no config explosion, and minimal intrusive changes to core training code.

This document is self-contained and is the only plan required for implementation.

---

## 2) Requirements

1. Milestone gating
- Work in milestone order.
- Stop after each milestone and request explicit approval before moving forward.

2. Task/group resolution and range semantics
- YAML must define canonical group names and expected `max_index`.
- Actual task lists must be resolved from RoboCasa registry at launch time.
- `task_range` is 1-based inclusive over each resolved group list.
- Launcher must fail before submission if:
  - any task index is out of bounds,
  - resolved group size does not equal YAML `max_index`,
  - requested demos exceed availability for that split.

3. Task selection and demo controls
- Task choice must be runtime-driven by YAML, not static per-task `TrainConfig`s.
- User-controlled demos per task must be allowed in `[1, 500]`.
- Fail fast on invalid demo counts with clear error messages.

4. Launcher UX
- Final launcher command has no required CLI args.
- Run definitions and per-run overrides live in YAML.
- YAML supports both single index and index ranges.
- YAML controls start checkpoint and run options.

5. Checkpoint behavior
- Arbitrary start checkpoint path must be supported.
- All RoboCasa run artifacts and checkpoints must be rooted under:
  `/coc/testnvme/xzhang3205/checkpoints/robocasa`

6. Runtime and cluster constraints
- Worker execution must activate and use repo `.venv`.
- Slurm workflow is required.
- Launcher code and sbatch script live under `train_launcher/`.

7. Code quality constraints
- Keep changes to original `openpi` minimal.
- Keep campaign/sweep logic out of `src/openpi/training/config.py`.
- Emit strict, specific, actionable validation errors.

8. Norm stats policy
- Do not recompute norm stats on downstream finetuning data.
- Finetuning must reuse checkpoint-provided norm stats.
- Worker must not call `scripts/compute_norm_stats.py`.

---

## 3) Architecture

Use a two-phase architecture with explicit environment boundaries.

1. Launcher phase (control plane)
- Responsibility:
  - parse YAML,
  - resolve groups/tasks from RoboCasa registry,
  - validate all runs,
  - materialize per-run JSON manifests,
  - submit sbatch jobs.
- Environment:
  - RoboCasa-capable environment where registry imports are available.

2. Worker phase (training plane)
- Responsibility:
  - read per-run manifest,
  - run `scripts/train.py` with resolved overrides.
- Environment:
  - repo `.venv` only.
- Constraint:
  - no RoboCasa registry imports in worker.

Data handoff between phases:
- one `dataset_meta.json` file per run directory.

---

## 4) Files To Create or Update

1. `train_launcher/config.yaml`
- Single source of truth for launcher defaults, groups, runs, and slurm defaults.

2. `train_launcher/registry.py`
- Group resolution and dataset-meta builder utilities.

3. `train_launcher/launch.py`
- Zero-required-args launcher entrypoint.

4. `train_launcher/launch_train.sbatch`
- Worker script for one resolved run.
- For slurm specific settings you should copy from /coc/testnvme/xzhang3205/vla-adaptation/models/openpi/convert.sh only except the .out and .err location and the job name. Those logs should appear in results/finetune

5. `train_launcher/README.md`
- Schema, behavior, dry-run usage, and failure mode documentation.

6. `src/openpi/training/config.py` (minimal change only)
- Add/verify runtime manifest hook for resolved dataset metadata input if missing.
- Keep all campaign logic outside this file.

---

## 5) YAML Specification (`train_launcher/config.yaml`)

Required top-level keys:
1. `defaults`
- `base_config_name` (string)
- `checkpoint_save_base_dir` (absolute path)
- `start_checkpoint` (absolute path)
- `ckpt_tag` (string)
- `num_demos` (int)

2. `slurm`
- `partition`, `qos`, `gpus`, `cpus_per_task`, `mem_per_gpu`
- optional: `exclude`, `time`

3. `groups`
- mapping of group name -> `{ max_index: <int> }`

4. `runs`
- list of run entries:
  - `name` (slurm job name stem)
  - `group`
  - `task_range`
  - optional overrides: `num_demos`, `start_checkpoint`, `ckpt_tag`, slurm override subset

`task_range` grammar:
- single index: `"7"`
- inclusive range: `"3-11"`

Canonical groups to support:
- `pretrain_all_300`
- `pretrain_only_266`
- `atomic_seen`
- `composite_seen`
- `composite_unseen`
- `eval_target_50`

---

## 6) Registry Resolution Rules (`train_launcher/registry.py`)

Implement these functions exactly.

1. `resolve_group(group_name: str, expected_max_index: int) -> list[str]`
- Resolve ordered task list from RoboCasa registry sources.
- Raise `ValueError` if `group_name` is unknown.
- Raise `ValueError` if `len(resolved_tasks) != expected_max_index`.

Group mapping:
- `pretrain_all_300`: `TASK_SET_REGISTRY["pretrain300"]`
- `pretrain_only_266`: ordered `pretrain300` minus `eval_target_50`
- `atomic_seen`: `TARGET_TASKS["atomic_seen"]`
- `composite_seen`: `TARGET_TASKS["composite_seen"]`
- `composite_unseen`: `TARGET_TASKS["composite_unseen"]`
- `eval_target_50`: concatenation of `atomic_seen + composite_seen + composite_unseen` in canonical order

2. `group_split_cap(group_name: str) -> int`
- Return demo cap by split:
  - pretrain groups: `100`
  - target groups: `500`

3. `build_meta(task_name: str, group_name: str, num_demos: int) -> dict`
- Validate `num_demos`:
  - global bound: `1 <= num_demos <= 500`
  - split bound: `num_demos <= group_split_cap(group_name)`
- Create dataset meta dict using RoboCasa dataset registry helper.
- Force exact demo selection by setting:
  - `meta["filter_key"] = f"{num_demos}_demos"`
- Return meta dict.

Do not perform filesystem existence checks in `registry.py`.

---

## 7) Launcher Behavior (`train_launcher/launch.py`)

Entry command:
- `python train_launcher/launch.py`
- no required CLI arguments.

Execution algorithm:
1. Load YAML config.
2. Generate single batch timestamp `mm-dd-hh-mm`; reuse for all expanded runs.
3. Expand each run into concrete units `(run_name, group, task_idx, overrides)`.
4. Perform full-batch validation before any submission:
- group exists in YAML and resolver
- resolved group size matches `max_index`
- `task_range` parse succeeds
- all indices satisfy `1 <= idx <= max_index`
- effective `num_demos` valid under global and split bounds
- effective `start_checkpoint` is non-empty string
5. If any unit fails validation:
- abort entire batch,
- submit nothing,
- print one clear error including run name, group, task_idx (if available), and reason.
6. For each validated unit:
- resolve `task_name = tasks[idx - 1]`
- sanitize `task_name` for filesystem safety
- sanitize `ckpt_tag` (lowercase, no slash)
- construct `exp_name`:
  `<group>/<idx>_<task_name>/demo<num_demos>_<ckpt_tag>_<timestamp>`
- construct run directory exactly:
  `<checkpoint_save_base_dir>/<group>/<idx>_<task_name>/demo<num_demos>_<ckpt_tag>_<timestamp>/`
- create run directory
- write `<run_dir>/dataset_meta.json` as JSON list containing one meta object
- write `<run_dir>/run_config.json` audit file containing all resolved effective values
7. Build sbatch command with env vars and submit one job per run unit.
8. Print one submission line per job:
- `[submitted jobid=<id>] <group>/<idx>_<task_name> demo=<num_demos>`

Optional flag:
- `--dry-run` prints resolved runs and sbatch commands, writes no jobs.

---

## 8) Worker Behavior (`train_launcher/launch_train.sbatch`)

Worker receives env vars:
- `RUN_BASE_CONFIG`
- `RUN_EXP_NAME`
- `RUN_CHECKPOINT_BASE_DIR`
- `RUN_DATA_DIRS_JSON`
- `RUN_START_CHECKPOINT`
- slurm-related environment as needed

Worker steps:
1. `cd` repository root.
2. `source .venv/bin/activate`.
3. Execute training command only:

```bash
python scripts/train.py "$RUN_BASE_CONFIG" \
  --exp-name="$RUN_EXP_NAME" \
  --checkpoint-base-dir="$RUN_CHECKPOINT_BASE_DIR" \
  --data.data-dirs-json="$RUN_DATA_DIRS_JSON" \
  --weight-loader.params-path="$RUN_START_CHECKPOINT"
```

Hard prohibition:
- No call to `scripts/compute_norm_stats.py` in worker.

Norm stats source:
- must come from checkpoint asset loading/fallback path in config behavior.

---

## 9) Path and Naming Contracts

1. Checkpoint base root
- `/coc/testnvme/xzhang3205/checkpoints/robocasa`

2. Per-run path format (exact)
- `checkpoint_save_base_dir/<group>/<idx>_<task_name>/demo<num_demos>_<ckpt_tag>_<timestamp>/`

3. Timestamp contract
- one timestamp per launcher invocation; shared across all runs in the batch.

4. Batch atomicity contract
- any invalid expanded run blocks all submissions.

---

## 10) Milestones (Approval Required Between Milestones)

Milestone 0: Baseline audit
- Inspect current launcher/config state.
- Produce exact gap list against this plan.
- Stop for approval.

Milestone 1: Minimal core config hook
- Implement or verify runtime JSON dataset meta input hook in `src/openpi/training/config.py`.
- Ensure compatible base finetune config entry exists and is stable.
- Run config-load sanity checks.
- Stop for approval.

Milestone 2: Launcher implementation
- Implement `registry.py`, `launch.py`, and `launch_train.sbatch`.
- Implement validate-all-then-submit behavior.
- Implement canonical path construction.
- Add `--dry-run`.
- Stop for approval.

Milestone 3: Verification and docs
- Execute positive and negative tests listed below.
- Write `train_launcher/README.md`.
- Update `finetune.md` launcher section.
- Stop for approval.

---

## 11) Verification Plan (Must Execute)

Positive cases:
1. Single index run expands and submits one job.
2. Range run expands and submits one job per index.
3. All runs in one invocation share identical timestamp.
4. `dataset_meta.json` and `run_config.json` created for each run.

Negative cases:
1. Unknown group -> fail before submission.
2. Group size mismatch vs `max_index` -> fail before submission.
3. Out-of-bounds index -> fail before submission.
4. `num_demos` outside `[1,500]` -> fail before submission.
5. Pretrain group with `num_demos > 100` -> fail before submission.

Policy cases:
1. Worker script contains no `compute_norm_stats.py` invocation.
2. Output path exactly matches required format.
3. Worker activates `.venv` before training command.

Evidence to collect:
- dry-run output logs,
- one successful sbatch submission log line,
- one failure log line for each negative category.

---

## 12) Non-Goals

1. No static per-task `TrainConfig` expansion.
2. No model architecture or objective changes.
3. No broad refactor outside RoboCasa finetuning path.

---

## 13) Implementation Notes

1. Keep errors explicit and deterministic.
2. Prefer small helper functions over large monolithic launcher function.
3. Keep registry-dependent logic isolated to `train_launcher/registry.py`.
4. Keep worker script dumb: consume env vars and run training command.
