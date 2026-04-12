#!/usr/bin/env python3
"""Submit OpenArm static inference jobs through launch_static.sbatch."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT_DIR = Path("/work/nvme/bfbo/xzhang42/openpi")
RESULT_DIR = Path("/work/hdd/bfbo/xzhang42/static/openarm_full")
CHECKPOINT_DIR = Path("/work/nvme/bfbo/xzhang42/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch")
SKIP_FRAME = 40
MAX_FRAMES = 0
NUM_STEPS = 10
PERTURBANCE_STEP_NUM = 0
PERTURBANCE_STEP_SIZE = 1e-2
EMBEDDING_TYPE = "vision"

DATASETS: dict[str, dict[str, str]] = {
    "pick_cup": {
        "repo": "qrafty-ai/tea_pick_cup",
        "path": "/work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_pick_cup",
        "config": "pi05_tea_pick_cup",
        "skip_frame": "20",
        "max_frames": "1500",
    },
    "pour_ice": {
        "repo": "qrafty-ai/tea_pour_ice",
        "path": "/work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_pour_ice",
        "config": "pi05_tea_pour_ice",
        "skip_frame": "40",
        "max_frames": "5000",
    },
    "use_spoon": {
        "repo": "qrafty-ai/tea_use_spoon_openpi",
        "path": "/work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_use_spoon_openpi",
        "config": "pi05_tea_use_spoon",
        "skip_frame": "40",
        "max_frames": "4700",
    },
    "use_steel_spoon": {
        "repo": "qrafty-ai/tea_use_steel_spoon",
        "path": "/work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_use_steel_spoon",
        "config": "pi05_tea_use_steel_spoon",
        "skip_frame": "40",
        "max_frames": "5000",
    },
}

RUNS = [
    # ("cosine", "cosine", "training", False),
    # ("gradient-inference", "gradient", "inference", False),
    ("perturbance-noise", "perturbance-noise", "inference", False),
]


def submit_job(
    *,
    dataset_name: str,
    dataset_meta: dict[str, str],
    mode: str,
    metric: str,
    condition: str,
    save_meta: bool,
) -> None:
    output_root = RESULT_DIR / dataset_name / mode
    output_root.mkdir(parents=True, exist_ok=True)

    export_vars = {
        "DATASET": dataset_name,
        "DATASET_REPO": dataset_meta["repo"],
        "DATASET_PATH": dataset_meta["path"],
        "DATASET_CONFIG": dataset_meta["config"],
        "OUTPUT_ROOT": str(output_root),
        "CHECKPOINT_DIR": str(CHECKPOINT_DIR),
        "METRIC": metric,
        "CONDITION": condition,
        "SAVE_META": "True" if save_meta else "False",
        "SKIP_FRAME": dataset_meta.get("skip_frame", str(SKIP_FRAME)),
        "NUM_STEPS": str(NUM_STEPS),
        "MAX_FRAMES": dataset_meta.get("max_frames", str(MAX_FRAMES)),
        "PERTURBANCE_STEP_NUM": str(PERTURBANCE_STEP_NUM),
        "PERTURBANCE_STEP_SIZE": str(PERTURBANCE_STEP_SIZE),
        "EMBEDDING_TYPE": EMBEDDING_TYPE,
    }
    export_str = ",".join(f"{key}={value}" for key, value in export_vars.items())
    cmd = ["sbatch", f"--export=ALL,{export_str}", "launch_static.sbatch"]
    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main() -> None:
    for dataset_name, dataset_meta in DATASETS.items():
        for mode, metric, condition, save_meta in RUNS:
            submit_job(
                dataset_name=dataset_name,
                dataset_meta=dataset_meta,
                mode=mode,
                metric=metric,
                condition=condition,
                save_meta=save_meta,
            )


if __name__ == "__main__":
    main()
