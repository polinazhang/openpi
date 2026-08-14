#!/usr/bin/env python3
"""Launch paired pi05-base/pi05-robocasa static inference on RoboCasa365 task splits.

This launcher intentionally reuses static_inference.py, robocasa_dataset.py, and
launch_robocasa.sbatch without modifying them. It creates a small symlink tree
that exposes each single-task LeRobot split through the RoboCasa target layout
expected by robocasa_dataset.py, then submits one Slurm job per model/task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SBATCH_SCRIPT = Path(__file__).resolve().parent / "launch_robocasa.sbatch"
SPLIT_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation/datasets/robocasa365/atomic-seen-splits")
MANIFEST = SPLIT_ROOT / "build_manifest.json"
COMPAT_BASE = SPLIT_ROOT / "_static_inference_robocasa_layout"
OUTPUT_BASE = Path("/coc/testnvme/xzhang3205/static")

MODELS = {
    "pi05-base": {
        "checkpoint": Path("/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/base/pi05_base_torch"),
        "norm_stats": Path(
            "/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/base/pi05_base_torch/assets/franka_robocasa_padded"
        ),
    },
    "pi05-robocasa": {
        "checkpoint": Path(
            "/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/robocasa-multi-models/pi05_pretrain_human300/multitask_learning/75000_torch"
        ),
        "norm_stats": Path(
            "/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/robocasa-multi-models/pi05_pretrain_human300/multitask_learning/75000_torch/assets"
        ),
    },
}

RUN_CONFIG = {
    "DATASET": "atomic-seen",
    "CONFIG_NAME": "pi05_robocasa_checkpoint",
    "MAX_EPISODES": "50",
    "METRIC": "perturbance-noise",
    "CONDITION": "inference",
    "SAVE_META": "True",
    "SAVE_COSINE": "True",
    "SAVE_DISPLACEMENT_TRACE": "True",
    "SKIP_FRAME": "1",
    "NUM_STEPS": "10",
    "EMBEDDING_TYPE": "vision",
    "PERTURBANCE_STEP_NUM": "0",
    "PERTURBANCE_STEP_SIZE": "1e-2",
    "VIDEO_BACKEND": "pyav",
}


@dataclass(frozen=True)
class TaskSplit:
    task_number: int
    task_name: str
    dataset_root: Path

    @property
    def output_name(self) -> str:
        return f"task_{self.task_number:02d}_{self.task_name}"


def load_task_splits() -> list[TaskSplit]:
    manifest = json.loads(MANIFEST.read_text())
    tasks: list[TaskSplit] = []
    for record in manifest["datasets"]:
        output_path = Path(record["output_path"])
        match = re.fullmatch(r"task_(\d+)_demo_50", output_path.name)
        if not match:
            continue
        if int(record["selected_episode_count"]) != 50:
            raise ValueError(f"Expected 50 episodes for {output_path}, got {record['selected_episode_count']}")
        tasks.append(
            TaskSplit(
                task_number=int(match.group(1)),
                task_name=record["source_task_name"],
                dataset_root=output_path,
            )
        )

    tasks.sort(key=lambda task: task.task_number)
    numbers = [task.task_number for task in tasks]
    if numbers != list(range(1, 19)):
        raise ValueError(f"Expected task_1_demo_50 through task_18_demo_50; got {numbers}")
    return tasks


def prepare_compat_tree(tasks: list[TaskSplit]) -> None:
    """Create v1.0/target/atomic/<Task>/20250822/lerobot symlinks."""
    for task in tasks:
        if not (task.dataset_root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata under {task.dataset_root}")
        version_dir = COMPAT_BASE / "v1.0" / "target" / "atomic" / task.task_name / "20250822"
        version_dir.mkdir(parents=True, exist_ok=True)
        link = version_dir / "lerobot"
        target = task.dataset_root.resolve()
        if link.is_symlink():
            if link.resolve() == target:
                continue
            link.unlink()
        elif link.exists():
            raise FileExistsError(f"Refusing to replace non-symlink path: {link}")
        link.symlink_to(target, target_is_directory=True)


def validate_inputs(models: list[str]) -> None:
    if not SBATCH_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing sbatch script: {SBATCH_SCRIPT}")
    for model_name in models:
        model = MODELS[model_name]
        for key in ("checkpoint", "norm_stats"):
            path = model[key]
            if not path.exists():
                raise FileNotFoundError(f"{model_name} {key} does not exist: {path}")
        if not (model["checkpoint"] / "model.safetensors").is_file():
            raise FileNotFoundError(f"{model_name} checkpoint lacks model.safetensors")
        if not (model["norm_stats"] / "norm_stats.json").is_file():
            raise FileNotFoundError(f"{model_name} norm stats lack norm_stats.json")


def submit(model_name: str, task: TaskSplit, *, dry_run: bool) -> str | None:
    model = MODELS[model_name]
    task_output = OUTPUT_BASE / model_name / task.output_name
    env = {
        **RUN_CONFIG,
        "ROBOCASA_TASK": task.task_name,
        "ROBOCASA_BASE": str(COMPAT_BASE),
        "OUTPUT_ROOT": str(task_output / "perturbance-all"),
        "CHECKPOINT_DIR": str(model["checkpoint"]),
        "NORM_STATS_DIR": str(model["norm_stats"]),
    }
    export_blob = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    job_name = f"static-{model_name}-{task.output_name}"[:255]
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--export={export_blob}",
        str(SBATCH_SCRIPT),
    ]
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return None

    task_output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    print(completed.stdout.strip())
    match = re.search(r"Submitted batch job (\d+)", completed.stdout)
    return match.group(1) if match else None


def main() -> int:
    args = parse_args()
    tasks = load_task_splits()
    if args.tasks:
        requested = set(args.tasks)
        tasks = [task for task in tasks if task.task_number in requested]
        missing = requested - {task.task_number for task in tasks}
        if missing:
            raise ValueError(f"Unknown task numbers: {sorted(missing)}")

    prepare_compat_tree(tasks)
    validate_inputs(args.models)
    print(f"Prepared compatibility tree: {COMPAT_BASE}")
    print(f"Launching {len(args.models) * len(tasks)} jobs: {len(args.models)} models x {len(tasks)} tasks")
    if args.prepare_only:
        return 0

    submitted: list[tuple[str, str, str | None]] = []
    for model_name in args.models:
        for task in tasks:
            job_id = submit(model_name, task, dry_run=args.dry_run)
            submitted.append((model_name, task.output_name, job_id))

    if not args.dry_run:
        manifest_path = OUTPUT_BASE / "robocasa365_static_jobs.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                [
                    {"model": model, "task": task, "job_id": job_id}
                    for model, task, job_id in submitted
                ],
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote job manifest: {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting.")
    parser.add_argument("--prepare-only", action="store_true", help="Create/validate the symlink tree and exit.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS),
        default=sorted(MODELS),
        help="Subset of models to launch (default: both).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        help="Optional subset of 1-based RoboCasa365 task numbers to launch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
