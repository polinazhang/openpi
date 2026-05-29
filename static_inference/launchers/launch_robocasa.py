#!/usr/bin/env python3
"""Submit one static-inference sbatch per RoboCasa held-out task.

Iterates the union of atomic_seen + composite_seen + composite_unseen (50 tasks
total) sourced from ``static_inference.robocasa_dataset.TARGET_SPLITS`` and
submits one slurm job per task via ``launch_robocasa.sbatch``. Each job loads
the dataset's first ``MAX_EPISODES`` episodes for a single task.

Outputs land at:
    <repo>/results/static/<split>-<launchtime>/<task_name>/
where ``launch_robocasa.sbatch`` flattens the inner ``<dataset>/`` subdir that
``static_inference.py`` creates.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SBATCH_SCRIPT = Path(__file__).resolve().parent / "launch_robocasa.sbatch"

# Source-of-truth task list lives next to the runtime loader.
sys.path.insert(0, str(REPO_ROOT / "static_inference"))
from robocasa_dataset import TARGET_SPLITS  # noqa: E402


CHECKPOINT_DIR = Path(
    "/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/robocasa-multi-models/"
    "pi05_pretrain_human300/multitask_learning/75000_torch"
)
NORM_STATS_DIR = CHECKPOINT_DIR / "assets"
ROBOCASA_BASE = Path("/coc/pskynet4/chuang475/datasets/robocasa")
OUTPUT_BASE = REPO_ROOT / "results" / "static"
MAX_EPISODES = 100
CONFIG_NAME = "pi05_robocasa_checkpoint"

# kebab dataset key in static_inference.DATASETS  ->  snake split key in TARGET_SPLITS
SPLITS: list[tuple[str, str]] = [
    ("atomic-seen", "atomic_seen"),
    ("composite-seen", "composite_seen"),
    ("composite-unseen", "composite_unseen"),
]

RUN_CONFIG: dict[str, str] = {
    "METRIC": "perturbance-noise",
    "CONDITION": "inference",
    "SAVE_META": "True",
    "SAVE_COSINE": "True",
    "SAVE_DISPLACEMENT_TRACE": "True",
    "SKIP_FRAME": "10",
    "NUM_STEPS": "10",
    "EMBEDDING_TYPE": "vision",
    "PERTURBANCE_STEP_NUM": "0",
    "PERTURBANCE_STEP_SIZE": "1e-2",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands instead of submitting.",
    )
    return parser.parse_args()


def _build_export_blob(env: dict[str, str]) -> str:
    return "ALL," + ",".join(f"{k}={v}" for k, v in env.items())


def _submit(env: dict[str, str], *, dry_run: bool) -> None:
    cmd = ["sbatch", f"--export={_build_export_blob(env)}", str(SBATCH_SCRIPT)]
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return
    print("Submitting:", env["DATASET"], env["ROBOCASA_TASK"])
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    args = _parse_args()
    if not SBATCH_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing sbatch script: {SBATCH_SCRIPT}")

    launch_time = _datetime.datetime.now().strftime("%m-%d-%H%M")
    total = 0
    for dataset_key, split_key in SPLITS:
        tasks = TARGET_SPLITS[split_key]
        split_root = OUTPUT_BASE / f"{dataset_key}-{launch_time}"
        for task in tasks:
            # `static_inference.py` rewrites output_root.name to "perturbance-all" when
            # metric=perturbance-noise; tuck that wrapper inside the task dir so it's a
            # no-op and the sbatch flatten step can strip it back out.
            output_root = split_root / task / "perturbance-all"
            if not args.dry_run:
                output_root.mkdir(parents=True, exist_ok=True)

            env = {
                "DATASET": dataset_key,
                "ROBOCASA_TASK": task,
                "OUTPUT_ROOT": str(output_root),
                "CHECKPOINT_DIR": str(CHECKPOINT_DIR),
                "NORM_STATS_DIR": str(NORM_STATS_DIR),
                "ROBOCASA_BASE": str(ROBOCASA_BASE),
                "MAX_EPISODES": str(MAX_EPISODES),
                "CONFIG_NAME": CONFIG_NAME,
                **RUN_CONFIG,
            }
            _submit(env, dry_run=args.dry_run)
            total += 1

    print(f"{'Would submit' if args.dry_run else 'Submitted'} {total} jobs (launch_time={launch_time}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
