# RoboCasa Train Launcher

Standalone launcher for RoboCasa finetuning runs.

- Control plane: `train_launcher/launch.py` (run in RoboCasa-capable environment).
- Training plane: `train_launcher/launch_train.sbatch` (runs `scripts/train.py` in repo `.venv`).

## Entry Command

```bash
python train_launcher/launch.py
```

Optional:

```bash
python train_launcher/launch.py --dry-run
python train_launcher/launch.py --config /path/to/config.yaml
```

## Config Schema (`train_launcher/config.yaml`)

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
- mapping: `group_name -> { max_index: int }`

4. `runs`
- list entries with:
  - `name`
  - `group`
  - `task_range` (`"7"` or `"3-11"`)
  - optional overrides: `num_demos`, `start_checkpoint`, `ckpt_tag`, slurm overrides

Canonical groups:

- `pretrain_all_300`
- `pretrain_only_266`
- `atomic_seen`
- `composite_seen`
- `composite_unseen`
- `eval_target_50`

## Behavior

`launch.py` does the following:

1. Loads YAML.
2. Resolves group task lists from RoboCasa registry.
3. Expands each run into concrete run units.
4. Validates the full batch before any submission.
5. If valid and not dry-run:
- creates run dir (mirrors `scripts/train.py` checkpoint layout):
  `checkpoint_save_base_dir/<base_config_name>/<group>/<idx>_<task_name>/demo<num_demos>_<ckpt_tag>_<timestamp>/`
- writes `dataset_meta.json` (JSON list with one meta dict)
- writes `run_config.json`
- submits one `sbatch` job per run unit.

Batch timestamp format: `mm-dd-hh-mm` and is shared by all units in the same invocation.

## Validation Rules

Validation is fail-fast and atomic (no partial submissions):

- group must exist in YAML and resolver
- resolved group size must equal `max_index`
- `task_range` must parse and stay in bounds
- `num_demos` must be in `[1, 500]`
- pretrain groups cap demos at `100`
- target groups cap demos at `500`
- `start_checkpoint` must be non-empty

## Worker Contract

Worker receives env vars:

- `RUN_BASE_CONFIG`
- `RUN_EXP_NAME`
- `RUN_CHECKPOINT_BASE_DIR`
- `RUN_DATA_DIRS_JSON`
- `RUN_START_CHECKPOINT`

Worker behavior:

1. `cd` repo root
2. `source .venv/bin/activate`
3. run only:

```bash
python scripts/train.py "$RUN_BASE_CONFIG" \
  --exp-name="$RUN_EXP_NAME" \
  --checkpoint-base-dir="$RUN_CHECKPOINT_BASE_DIR" \
  --data.data-dirs-json="$RUN_DATA_DIRS_JSON" \
  --weight-loader.params-path="$RUN_START_CHECKPOINT"
```

`RUN_DATA_DIRS_JSON` is the file path to the launcher-written `dataset_meta.json`.

## Dry-Run Output

`--dry-run` prints:

- resolved unit count and shared timestamp
- one resolved target line per unit
- one `sbatch` command line per unit

No directories are created and no jobs are submitted in dry-run mode.

## Failure Modes

Typical failures are explicit and include run/group/task context:

- unknown group
- group size mismatch vs `max_index`
- out-of-bounds index
- invalid `num_demos`
- split cap exceeded (`pretrain > 100`)
- malformed `task_range`
- missing required config keys

## Paths and Logs

- checkpoint/artifact root: `/coc/pskynet4/xzhang3205/robocasa`
- slurm logs: `results/finetune/<run-name>-%J.out|err`
