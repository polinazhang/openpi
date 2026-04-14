#!/usr/bin/env python3
"""Submit static inference jobs for the 4 OOD conditions.

Conditions:
  in-dist             checkpoint=pi05_franka_object_single  dataset=franka_object_single
  vision-ood-replace  checkpoint=torch_30000                dataset=franka_object_vision_ood_replace
  vision-ood-addition checkpoint=torch_30000                dataset=franka_object_vision_ood_addition
  action-ood          checkpoint=torch_30000                dataset=franka_object_action_ood
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit OOD static inference jobs.")
    parser.add_argument("--test", action="store_true", help="Run quick test job only.")
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
    save_cosine: bool,
    save_displacement_trace: bool,
    checkpoint_dir: Path,
    skip_frame: int,
    max_frames: int,
    perturbance_step_num: int = 0,
    perturbance_step_size: float = 1e-2,
    embedding_type: str = "vision",
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
        "SAVE_COSINE": "True" if save_cosine else "False",
        "SAVE_DISPLACEMENT_TRACE": "True" if save_displacement_trace else "False",
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
    folder_name = "ood_full"

    CHECKPOINT_DEFAULT = Path("/coc/testnvme/xzhang3205/openpi/checkpoints/torch_30000")
    CHECKPOINT_INDIST = Path("/coc/testnvme/xzhang3205/openpi/checkpoints/pi05_franka_object_single")

    # (dataset, checkpoint_dir)
    conditions = [
        ("franka_object_single",           CHECKPOINT_INDIST),
        ("franka_object_vision_ood_replace", CHECKPOINT_DEFAULT),
        ("franka_object_vision_ood_addition", CHECKPOINT_DEFAULT),
        ("franka_object_action_ood",         CHECKPOINT_DEFAULT),
    ]

    runs = [
        ("perturbance-all", "perturbance-noise", "inference", True, True, True),
    ]

    skip_frame = 10
    max_frames = 0

    if args.test:
        folder_name = "test_currentime"
        # franka_object_single is a single 364-frame episode — smallest possible test
        conditions = [("franka_object_single", CHECKPOINT_INDIST)]
        skip_frame = 200
        max_frames = 1200

    for dataset, checkpoint_dir in conditions:
        for mode, metric, condition, save_meta, save_cosine, save_displacement_trace in runs:
            submit_job(
                root_dir=root_dir,
                folder_name=folder_name,
                dataset=dataset,
                mode=mode,
                metric=metric,
                condition=condition,
                save_meta=save_meta,
                save_cosine=save_cosine,
                save_displacement_trace=save_displacement_trace,
                checkpoint_dir=checkpoint_dir,
                skip_frame=skip_frame,
                max_frames=max_frames,
                perturbance_step_num=0,
                perturbance_step_size=1e-2,
                embedding_type="vision",
            )
