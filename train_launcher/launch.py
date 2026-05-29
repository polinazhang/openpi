#!/usr/bin/env python3
"""Standalone RoboCasa finetune launcher with strict pre-submit validation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml

try:
    from train_launcher.registry import build_meta, group_split_cap, resolve_group
except ModuleNotFoundError:
    # Allow direct execution without requiring manual PYTHONPATH exports.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from train_launcher.registry import build_meta, group_split_cap, resolve_group

SLURM_KEYS = {
    "partition",
    "qos",
    "gpus",
    "cpus_per_task",
    "mem_per_gpu",
    "exclude",
    "time",
}
REQUIRED_SLURM_KEYS = {"partition", "qos", "gpus", "cpus_per_task", "mem_per_gpu"}
REQUIRED_TOP_LEVEL_KEYS = {"defaults", "slurm", "groups", "runs"}


class LauncherConfigError(ValueError):
    """Raised when launcher config or resolved units are invalid."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config.yaml"


def _sbatch_script_path() -> Path:
    return Path(__file__).resolve().parent / "launch_train.sbatch"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch RoboCasa finetuning runs from YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate runs, but do not submit jobs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to launcher YAML config (defaults to train_launcher/config.yaml).",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LauncherConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise LauncherConfigError("Top-level config must be a mapping.")

    missing = REQUIRED_TOP_LEVEL_KEYS - set(cfg.keys())
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise LauncherConfigError(f"Missing required top-level config key(s): {missing_keys}")

    return cfg


def _parse_task_range(task_range: str) -> list[int]:
    if re.fullmatch(r"\d+", task_range):
        return [int(task_range)]

    match = re.fullmatch(r"(\d+)-(\d+)", task_range)
    if match is None:
        raise LauncherConfigError(
            f"Invalid task_range '{task_range}'. Expected single index like '7' or range like '3-11'."
        )

    start = int(match.group(1))
    end = int(match.group(2))
    if start > end:
        raise LauncherConfigError(f"Invalid task_range '{task_range}': start index is greater than end index.")

    return list(range(start, end + 1))


def _sanitize_task_name(task_name: str) -> str:
    # Keep deterministic and filesystem-safe names while preserving readability.
    return re.sub(r"[^A-Za-z0-9._-]+", "_", task_name).strip("_")


def _sanitize_ckpt_tag(ckpt_tag: str) -> str:
    lowered = ckpt_tag.lower().replace("/", "_")
    sanitized = re.sub(r"[^a-z0-9._-]+", "_", lowered).strip("_")
    if not sanitized:
        raise LauncherConfigError(f"ckpt_tag '{ckpt_tag}' is empty after sanitization.")
    return sanitized


def _resolve_slurm(run: dict[str, Any], slurm_defaults: dict[str, Any]) -> dict[str, Any]:
    effective = dict(slurm_defaults)

    nested = run.get("slurm")
    if nested is not None:
        if not isinstance(nested, dict):
            raise LauncherConfigError("Run-level 'slurm' override must be a mapping.")
        invalid = set(nested.keys()) - SLURM_KEYS
        if invalid:
            bad = ", ".join(sorted(invalid))
            raise LauncherConfigError(f"Run-level slurm override has unsupported key(s): {bad}")
        effective.update(nested)

    inline = {k: run[k] for k in SLURM_KEYS if k in run}
    effective.update(inline)

    missing = REQUIRED_SLURM_KEYS - set(effective.keys())
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise LauncherConfigError(f"Missing required slurm setting(s): {missing_keys}")

    return effective


def _fail(run_name: str, group: str, task_idx: int | None, reason: str) -> LauncherConfigError:
    idx = f", task_idx={task_idx}" if task_idx is not None else ""
    return LauncherConfigError(f"Validation failed (run={run_name}, group={group}{idx}): {reason}")


def _expand_and_validate(cfg: dict[str, Any], timestamp: str) -> list[dict[str, Any]]:
    defaults = cfg["defaults"]
    groups_cfg = cfg["groups"]
    runs_cfg = cfg["runs"]
    slurm_defaults = cfg["slurm"]

    if not isinstance(defaults, dict):
        raise LauncherConfigError("'defaults' must be a mapping.")
    if not isinstance(groups_cfg, dict):
        raise LauncherConfigError("'groups' must be a mapping of group -> {max_index}.")
    if not isinstance(runs_cfg, list):
        raise LauncherConfigError("'runs' must be a list.")
    if not isinstance(slurm_defaults, dict):
        raise LauncherConfigError("'slurm' must be a mapping.")

    required_defaults = {
        "base_config_name",
        "checkpoint_save_base_dir",
        "start_checkpoint",
        "ckpt_tag",
        "num_demos",
    }
    missing_defaults = required_defaults - set(defaults.keys())
    if missing_defaults:
        raise LauncherConfigError(
            f"Missing required defaults key(s): {', '.join(sorted(missing_defaults))}"
        )

    resolved_groups: dict[str, list[str]] = {}
    for group_name, group_entry in groups_cfg.items():
        if not isinstance(group_entry, dict) or "max_index" not in group_entry:
            raise LauncherConfigError(
                f"Group '{group_name}' must be a mapping containing 'max_index'."
            )
        expected_max_index = group_entry["max_index"]
        if not isinstance(expected_max_index, int) or expected_max_index < 1:
            raise LauncherConfigError(
                f"Group '{group_name}' has invalid max_index={expected_max_index}; expected positive int."
            )
        try:
            resolved_groups[group_name] = resolve_group(group_name, expected_max_index)
        except ValueError as exc:
            raise LauncherConfigError(str(exc)) from exc

    checkpoint_save_base_dir = Path(defaults["checkpoint_save_base_dir"])
    if not checkpoint_save_base_dir.is_absolute():
        raise LauncherConfigError("defaults.checkpoint_save_base_dir must be an absolute path.")

    base_config_name = defaults["base_config_name"]
    if not isinstance(base_config_name, str) or not base_config_name.strip():
        raise LauncherConfigError("defaults.base_config_name must be a non-empty string.")

    units: list[dict[str, Any]] = []

    for run in runs_cfg:
        if not isinstance(run, dict):
            raise LauncherConfigError("Each runs[] entry must be a mapping.")

        run_name = run.get("name")
        group_name = run.get("group")
        task_range = run.get("task_range")

        if not isinstance(run_name, str) or not run_name.strip():
            raise LauncherConfigError("Each run must define a non-empty string 'name'.")
        if not isinstance(group_name, str) or not group_name.strip():
            raise LauncherConfigError(f"Run '{run_name}' must define a non-empty string 'group'.")
        if not isinstance(task_range, str) or not task_range.strip():
            raise LauncherConfigError(f"Run '{run_name}' must define a non-empty string 'task_range'.")

        if group_name not in groups_cfg:
            raise _fail(run_name, group_name, None, "group is not present in YAML groups")
        if group_name not in resolved_groups:
            raise _fail(run_name, group_name, None, "group is not resolvable by registry")

        indices = _parse_task_range(task_range)
        max_index = groups_cfg[group_name]["max_index"]

        for idx in indices:
            if idx < 1 or idx > max_index:
                raise _fail(
                    run_name,
                    group_name,
                    idx,
                    f"task index out of bounds; valid range is 1..{max_index}",
                )

            num_demos = run.get("num_demos", defaults["num_demos"])
            if not isinstance(num_demos, int):
                raise _fail(run_name, group_name, idx, f"num_demos must be int, got {type(num_demos).__name__}")
            if num_demos < 1 or num_demos > 500:
                raise _fail(run_name, group_name, idx, f"num_demos must be within [1, 500], got {num_demos}")

            split_cap = group_split_cap(group_name)
            if num_demos > split_cap:
                raise _fail(
                    run_name,
                    group_name,
                    idx,
                    f"num_demos={num_demos} exceeds split cap {split_cap} for group '{group_name}'",
                )

            start_checkpoint = run.get("start_checkpoint", defaults["start_checkpoint"])
            if not isinstance(start_checkpoint, str) or not start_checkpoint.strip():
                raise _fail(run_name, group_name, idx, "start_checkpoint must be a non-empty string")

            ckpt_tag_raw = run.get("ckpt_tag", defaults["ckpt_tag"])
            if not isinstance(ckpt_tag_raw, str) or not ckpt_tag_raw.strip():
                raise _fail(run_name, group_name, idx, "ckpt_tag must be a non-empty string")
            ckpt_tag = _sanitize_ckpt_tag(ckpt_tag_raw)

            slurm = _resolve_slurm(run, slurm_defaults)

            task_name = resolved_groups[group_name][idx - 1]
            task_name_sanitized = _sanitize_task_name(task_name)
            if not task_name_sanitized:
                raise _fail(run_name, group_name, idx, f"task name '{task_name}' is empty after sanitization")

            exp_name = (
                f"{group_name}/{idx}_{task_name_sanitized}/"
                f"demo{num_demos}_{ckpt_tag}_{timestamp}"
            )
            # Mirror scripts/train.py checkpoint_dir layout (<base>/<name>/<exp_name>)
            # so meta artifacts land alongside the actual checkpoint output.
            run_dir = checkpoint_save_base_dir / base_config_name / exp_name

            dataset_meta = build_meta(task_name=task_name, group_name=group_name, num_demos=num_demos)

            units.append(
                {
                    "run_name": run_name,
                    "group": group_name,
                    "task_idx": idx,
                    "task_name": task_name,
                    "task_name_sanitized": task_name_sanitized,
                    "num_demos": num_demos,
                    "start_checkpoint": start_checkpoint,
                    "ckpt_tag": ckpt_tag,
                    "timestamp": timestamp,
                    "exp_name": exp_name,
                    "run_dir": run_dir,
                    "checkpoint_save_base_dir": str(checkpoint_save_base_dir),
                    "base_config_name": base_config_name,
                    "dataset_meta": dataset_meta,
                    "slurm": slurm,
                }
            )

    return units


def _build_sbatch_cmd(unit: dict[str, Any], sbatch_script: Path, repo_root: Path) -> list[str]:
    slurm = unit["slurm"]
    env_vars = {
        "RUN_BASE_CONFIG": unit["base_config_name"],
        "RUN_EXP_NAME": unit["exp_name"],
        "RUN_CHECKPOINT_BASE_DIR": unit["checkpoint_save_base_dir"],
        "RUN_DATA_DIRS_JSON": str(unit["run_dir"] / "dataset_meta.json"),
        "RUN_START_CHECKPOINT": unit["start_checkpoint"],
        "OPENPI_ROOT": str(repo_root),
    }
    export_blob = "ALL," + ",".join(f"{k}={v}" for k, v in env_vars.items())

    log_dir = repo_root / "results" / "finetune"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sbatch",
        f"--job-name={unit['run_name']}",
        f"--output={log_dir}/{unit['run_name']}-%J.out",
        f"--error={log_dir}/{unit['run_name']}-%J.err",
        f"--partition={slurm['partition']}",
        "--nodes=1",
        "--ntasks-per-node=1",
        f"--cpus-per-task={slurm['cpus_per_task']}",
        f"--gpus-per-node={slurm['gpus']}",
        f"--qos={slurm['qos']}",
        f"--mem-per-gpu={slurm['mem_per_gpu']}",
    ]

    if slurm.get("time"):
        cmd.append(f"--time={slurm['time']}")
    if slurm.get("exclude"):
        cmd.append(f"--exclude={slurm['exclude']}")

    cmd.extend([f"--export={export_blob}", str(sbatch_script)])
    return cmd


def _write_run_artifacts(unit: dict[str, Any]) -> None:
    run_dir = unit["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset_meta_path = run_dir / "dataset_meta.json"
    dataset_meta_path.write_text(json.dumps([unit["dataset_meta"]], indent=2) + "\n", encoding="utf-8")

    run_config = {
        "run_name": unit["run_name"],
        "group": unit["group"],
        "task_idx": unit["task_idx"],
        "task_name": unit["task_name"],
        "task_name_sanitized": unit["task_name_sanitized"],
        "num_demos": unit["num_demos"],
        "start_checkpoint": unit["start_checkpoint"],
        "ckpt_tag": unit["ckpt_tag"],
        "timestamp": unit["timestamp"],
        "exp_name": unit["exp_name"],
        "run_dir": str(run_dir),
        "base_config_name": unit["base_config_name"],
        "checkpoint_save_base_dir": unit["checkpoint_save_base_dir"],
        "dataset_meta_json": str(dataset_meta_path),
        "slurm": unit["slurm"],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")


def _submit(unit: dict[str, Any], sbatch_script: Path, repo_root: Path) -> None:
    cmd = _build_sbatch_cmd(unit, sbatch_script, repo_root)
    try:
        result = subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        joined_cmd = " ".join(cmd)
        raise LauncherConfigError(
            "sbatch submission failed.\n"
            f"command: {joined_cmd}\n"
            f"returncode: {exc.returncode}\n"
            f"stdout: {stdout or '<empty>'}\n"
            f"stderr: {stderr or '<empty>'}"
        ) from exc
    stdout = result.stdout.strip()

    job_id_match = re.search(r"Submitted batch job (\d+)", stdout)
    job_id = job_id_match.group(1) if job_id_match else "unknown"

    print(
        f"[submitted jobid={job_id}] "
        f"{unit['group']}/{unit['task_idx']}_{unit['task_name_sanitized']} "
        f"demo={unit['num_demos']}"
    )


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    sbatch_script = _sbatch_script_path()

    if not sbatch_script.is_file():
        raise LauncherConfigError(f"Missing sbatch script: {sbatch_script}")

    cfg = _load_yaml(args.config)
    timestamp = dt.datetime.now().strftime("%m-%d-%H-%M")
    units = _expand_and_validate(cfg, timestamp)

    if args.dry_run:
        print(f"[dry-run] resolved_units={len(units)} timestamp={timestamp}")
        for unit in units:
            cmd = _build_sbatch_cmd(unit, sbatch_script, repo_root)
            print(
                f"[dry-run unit] run={unit['run_name']} "
                f"target={unit['group']}/{unit['task_idx']}_{unit['task_name_sanitized']} "
                f"demo={unit['num_demos']}"
            )
            print("[dry-run sbatch] " + " ".join(cmd))
        return 0

    for unit in units:
        _write_run_artifacts(unit)

    for unit in units:
        _submit(unit, sbatch_script, repo_root)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LauncherConfigError as exc:
        print(str(exc))
        raise SystemExit(1)
