#!/usr/bin/env python3
"""Submit static inference jobs for cosine + gradient + perturbance static metrics."""

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
    save_meta: bool,
    checkpoint_dir: Path,
    skip_frame: int,
    max_frames: int,
    perturbance_step_num: int = 0,
    perturbance_step_size: float = 1e-2,
    embedding_type: str = "vision+action",
) -> None:
    output_root = Path("/coc/testnvme/xzhang3205/static") / folder_name / dataset / mode
    output_root.mkdir(parents=True, exist_ok=True)

    export_vars = {
        "DATASET": dataset,
        "OUTPUT_ROOT": str(output_root),
        "CHECKPOINT_DIR": str(checkpoint_dir),
        "METRIC": metric,
        "CONDITION": condition,
        "SAVE_META": "True" if save_meta else "False",
        "SKIP_FRAME": str(skip_frame),
        "NUM_STEPS": "10",
        "MAX_FRAMES": str(max_frames),
        "PERTURBANCE_STEP_NUM": str(perturbance_step_num),
        "PERTURBANCE_STEP_SIZE": str(perturbance_step_size),
        "EMBEDDING_TYPE": embedding_type,
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
        "franka_object_action_ood",
        "franka_object_vision_ood",
    ]
    if args.test:
        folder_name = "test_currentime"
        datasets = ["franka_object_action_ood"]

    runs = [
        # ("cosine", "cosine", "training", True),
        # ("gradient-training", "gradient", "training", False),
        # ("gradient-inference", "gradient", "inference", False),
        ("perturbance", "perturbance", "training", True),
    ]

    # Test mode should finish quickly while still leaving enough processed samples.
    # With skip_frame=500 and max_frames=3000, selected OOD datasets keep at least ~5 evaluated frames.
    skip_frame = 500 if args.test else 10
    max_frames = 3000 if args.test else 0
    if args.test:
        runs = [("perturbance", "perturbance", "training", True)]

    for dataset in datasets:
        for mode, metric, condition, save_meta in runs:
            submit_job(
                root_dir=root_dir,
                folder_name=folder_name,
                dataset=dataset,
                mode=mode,
                metric=metric,
                condition=condition,
                save_meta=save_meta,
                checkpoint_dir=checkpoint_dir,
                skip_frame=skip_frame,
                max_frames=max_frames,
                perturbance_step_num=0,
                perturbance_step_size=1e-2,
                # embedding_type="time",
                # embedding_type="vision+action",
                embedding_type="vision+action+time",
            )
