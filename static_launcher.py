#!/usr/bin/env python3
"""Submit static inference jobs for gradient (current milestone)."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit static inference jobs.")
    parser.add_argument("--test", action="store_true", help="Run quick test jobs only.")
    return parser.parse_args()


def submit_job(
    *,
    root_dir: Path,
    folder_name: str,
    dataset: str,
    mode: str,
    metric: str,
    condition: str,
    checkpoint_dir: Path,
    skip_frame: int,
    max_frames: int,
) -> None:
    output_root = Path("/coc/testnvme/xzhang3205/static") / folder_name / dataset / mode
    output_root.mkdir(parents=True, exist_ok=True)

    export_vars = {
        "DATASET": dataset,
        "OUTPUT_ROOT": str(output_root),
        "CHECKPOINT_DIR": str(checkpoint_dir),
        "METRIC": metric,
        "CONDITION": condition,
        "SAVE_META": "True",
        "SKIP_FRAME": str(skip_frame),
        "NUM_STEPS": "10",
        "MAX_FRAMES": str(max_frames),
    }
    export_str = ",".join([f"{k}={v}" for k, v in export_vars.items()])
    cmd = ["sbatch", f"--export=ALL,{export_str}", "launch_static.sbatch"]
    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, cwd=root_dir, check=True)


if __name__ == "__main__":
    args = parse_args()

    root_dir = Path("/coc/testnvme/xzhang3205/openpi")
    folder_name = "franka_full"
    checkpoint_dir = Path(
        os.environ.get("CHECKPOINT_DIR", "/coc/testnvme/xzhang3205/openpi/checkpoints/torch_30000")
    )

    datasets = [
        "franka_object",
        "franka_object_plus",
        "franka_object_two",
        "franka_on_top",
    ]
    if args.test:
        folder_name = "test_currentime"
        datasets = ["franka_on_top"]

    # Gradient-first milestone:
    # submit condition-training and condition-inference only.
    runs = [
        ("gradient-training", "gradient", "training"),
        ("gradient-inference", "gradient", "inference"),
    ]

    # Test mode should finish quickly while still leaving enough processed samples:
    # at least ~5 processed inputs for training and therefore ~50 inference steps.
    skip_frame = 500 if args.test else 10
    max_frames = 3000 if args.test else 0

    for dataset in datasets:
        for mode, metric, condition in runs:
            submit_job(
                root_dir=root_dir,
                folder_name=folder_name,
                dataset=dataset,
                mode=mode,
                metric=metric,
                condition=condition,
                checkpoint_dir=checkpoint_dir,
                skip_frame=skip_frame,
                max_frames=max_frames,
            )
