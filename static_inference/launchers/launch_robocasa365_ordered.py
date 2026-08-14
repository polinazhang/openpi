#!/usr/bin/env python3
"""Maintain the ordered, 20-slot RoboCasa365 static-inference queue."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from launch_robocasa365_pair import COMPAT_BASE
from launch_robocasa365_pair import MODELS
from launch_robocasa365_pair import REPO_ROOT
from launch_robocasa365_pair import RUN_CONFIG
from launch_robocasa365_pair import SBATCH_SCRIPT
from launch_robocasa365_pair import TaskSplit
from launch_robocasa365_pair import load_task_splits
from launch_robocasa365_pair import prepare_compat_tree
from launch_robocasa365_pair import validate_inputs

MAX_ACTIVE_JOBS = 20
OUTPUT_BASE = Path("/coc/testnvme/xzhang3205/static")
STATE_PATH = Path(__file__).resolve().parent / "robocasa365_ordered_state.json"
ERROR_LOG = Path("/coc/testnvme/xzhang3205/vla-adaptation/models/openpi/static_inference/ERROR-log.txt")

# Strict queue order. Tasks 1 through 18 are appended within each entry.
RUN_ORDER = (
    ("pi05-base", "base-arm"),
    ("pi05-robocasa", "robocasa"),
    ("pi05-base", "base-arm-base-grip"),
    ("pi05-base", "base-arm-base"),
)


def now() -> datetime:
    return datetime.now(UTC).astimezone()


ACTIVE_SLURM_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "RUNNING",
    "SUSPENDED",
}


@dataclasses.dataclass(frozen=True)
class QueueEntry:
    sequence: int
    model: str
    mode: str
    task_number: int
    task_name: str
    task_output_name: str
    output_root: str


def build_queue(tasks: list[TaskSplit], timestamp: str) -> list[QueueEntry]:
    entries: list[QueueEntry] = []
    for model_name, mode in RUN_ORDER:
        model_root = OUTPUT_BASE / model_name / f"{mode}-{timestamp}"
        for task in tasks:
            entries.append(
                QueueEntry(
                    sequence=len(entries),
                    model=model_name,
                    mode=mode,
                    task_number=task.task_number,
                    task_name=task.task_name,
                    task_output_name=task.output_name,
                    output_root=str(model_root / task.output_name / "perturbance-all"),
                )
            )
    return entries


def initialize_state(timestamp: str | None = None) -> dict[str, Any]:
    if STATE_PATH.exists():
        existing = json.loads(STATE_PATH.read_text())
        if not existing.get("finished", False):
            raise RuntimeError(f"An ordered workflow is already active: {STATE_PATH}")

    tasks = load_task_splits()
    prepare_compat_tree(tasks)
    validate_inputs(sorted({model for model, _ in RUN_ORDER}))
    timestamp = timestamp or now().strftime("%m%d%H%M%S")
    queue = build_queue(tasks, timestamp)
    state = {
        "timestamp": timestamp,
        "created_at": now().isoformat(timespec="seconds"),
        "finished": False,
        "entries": [
            {
                **dataclasses.asdict(entry),
                "status": "pending",
                "job_id": "",
                "submitted_at": "",
                "finished_at": "",
                "slurm_state": "",
                "exit_code": "",
            }
            for entry in queue
        ],
    }
    write_state(state)
    print(f"Initialized {len(queue)} ordered jobs with timestamp {timestamp}")
    return state


def write_state(state: dict[str, Any]) -> None:
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE_PATH)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def query_slurm(job_ids: list[str]) -> dict[str, tuple[str, str]]:
    """Return exact job-id -> (state, exit code) for active or historical jobs."""
    if not job_ids:
        return {}
    wanted = set(job_ids)
    states: dict[str, tuple[str, str]] = {}

    active = _run(["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T"])
    if active.returncode == 0:
        for line in active.stdout.splitlines():
            parts = line.strip().split("|", maxsplit=1)
            if len(parts) == 2 and parts[0] in wanted:
                states[parts[0]] = (parts[1].upper(), "")

    missing = [job_id for job_id in job_ids if job_id not in states]
    if missing:
        history = _run(["sacct", "-n", "-P", "-j", ",".join(missing), "--format=JobIDRaw,State,ExitCode"])
        if history.returncode == 0:
            for line in history.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) < 3 or parts[0] not in wanted:
                    continue
                # Slurm may append qualifiers such as COMPLETED+.
                state = parts[1].split()[0].rstrip("+").upper()
                states[parts[0]] = (state, parts[2])
    return states


def append_failure(entry: dict[str, Any], reason: str) -> None:
    timestamp = now().isoformat(timespec="seconds")
    line = (
        f"{timestamp} sequence={entry['sequence']} model={entry['model']} mode={entry['mode']} "
        f"task={entry['task_number']:02d}:{entry['task_name']} job_id={entry.get('job_id') or 'none'} "
        f"failure={reason}\n"
    )
    with ERROR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line)


def submit_entry(entry: dict[str, Any]) -> str:
    model = MODELS[entry["model"]]
    env = {
        **RUN_CONFIG,
        "ROBOCASA_TASK": entry["task_name"],
        "ROBOCASA_BASE": str(COMPAT_BASE),
        "OUTPUT_ROOT": entry["output_root"],
        "CHECKPOINT_DIR": str(model["checkpoint"]),
        "NORM_STATS_DIR": str(model["norm_stats"]),
        "DIM_REMAP": entry["mode"],
    }
    export_blob = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    job_name = f"opi-static-{entry['sequence']:02d}-{entry['mode']}-t{entry['task_number']:02d}"[:255]
    result = _run(
        [
            "sbatch",
            f"--job-name={job_name}",
            f"--export={export_blob}",
            str(SBATCH_SCRIPT),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "sbatch failed")
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse sbatch output: {result.stdout!r}")
    return match.group(1)


def update_submitted_entries(state: dict[str, Any]) -> None:
    submitted = [entry for entry in state["entries"] if entry["status"] == "submitted"]
    observed = query_slurm([entry["job_id"] for entry in submitted])
    checked_at = now().isoformat(timespec="seconds")
    for entry in submitted:
        observation = observed.get(entry["job_id"])
        if observation is None:
            # Do not duplicate a job during the brief interval before it is
            # visible to squeue/sacct. It remains counted as active.
            continue
        slurm_state, exit_code = observation
        entry["slurm_state"] = slurm_state
        entry["exit_code"] = exit_code
        if slurm_state in ACTIVE_SLURM_STATES:
            continue
        entry["finished_at"] = checked_at
        if slurm_state == "COMPLETED" and (not exit_code or exit_code.startswith("0:0")):
            entry["status"] = "complete"
            print(
                f"Completed {entry['job_id']}: sequence={entry['sequence']} {entry['mode']} task={entry['task_number']}"
            )
        else:
            entry["status"] = "failed"
            reason = f"slurm_state={slurm_state} exit_code={exit_code or 'unknown'}"
            append_failure(entry, reason)
            print(f"Failed {entry['job_id']}: {reason}", file=sys.stderr)


def fill_available_slots(state: dict[str, Any]) -> None:
    active = sum(entry["status"] == "submitted" for entry in state["entries"])
    available = MAX_ACTIVE_JOBS - active
    if available <= 0:
        return
    pending = [entry for entry in state["entries"] if entry["status"] == "pending"]
    for entry in pending[:available]:
        try:
            job_id = submit_entry(entry)
        except Exception as error:  # submission failure is terminal by contract
            entry["status"] = "failed"
            entry["finished_at"] = now().isoformat(timespec="seconds")
            entry["slurm_state"] = "SUBMIT_FAILED"
            append_failure(entry, f"submission={type(error).__name__}: {error}")
            print(f"Submission failed for sequence={entry['sequence']}: {error}", file=sys.stderr)
            continue
        entry["status"] = "submitted"
        entry["job_id"] = job_id
        entry["submitted_at"] = now().isoformat(timespec="seconds")
        print(
            f"Submitted {job_id}: sequence={entry['sequence']} {entry['model']} "
            f"{entry['mode']} task={entry['task_number']}"
        )


def run_once() -> dict[str, int]:
    if not STATE_PATH.is_file():
        raise FileNotFoundError(f"Workflow state does not exist; initialize first: {STATE_PATH}")
    state = json.loads(STATE_PATH.read_text())
    if state.get("finished"):
        return summarize(state)
    update_submitted_entries(state)
    fill_available_slots(state)
    summary = summarize(state)
    state["finished"] = summary["pending"] == 0 and summary["submitted"] == 0
    state["last_checked_at"] = now().isoformat(timespec="seconds")
    write_state(state)
    print(f"Queue summary: {summary}")
    return summary


def summarize(state: dict[str, Any]) -> dict[str, int]:
    return {
        status: sum(entry["status"] == status for entry in state["entries"])
        for status in ("pending", "submitted", "complete", "failed")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--initialize", action="store_true")
    action.add_argument("--run-once", action="store_true")
    parser.add_argument("--timestamp", help="Optional MMDDHHMMSS override for initialization")
    args = parser.parse_args()
    if args.initialize:
        initialize_state(args.timestamp)
    else:
        run_once()
    return 0


if __name__ == "__main__":
    sys.exit(main())
