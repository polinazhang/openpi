#!/usr/bin/env python3
"""Submit prototype one-episode static inference jobs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation/models/openpi")
PROTOTYPE_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation/prototype")

CHECKPOINT_PROFILES = {
    "base_pi05_with_franka_stats": {
        "checkpoint_dir": Path("/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/base/pi05_base_torch"),
        "norm_stats_dir": Path(
            "/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/base/pi05_base_torch/assets/franka"
        ),
    },
    "finetuned_franka_object_pi05": {
        "checkpoint_dir": Path("/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/franka/pi05_franka_object_30000"),
        "norm_stats_dir": None,
    },
    "finetuned_mesa_pi05": {
        "checkpoint_dir": Path("/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/mesa/mesa-pi05-torch"),
        "norm_stats_dir": None,
    },
}

BENCHMARKS = {
    "robocasa": {
        "checkpoint_profile": "base_pi05_with_franka_stats",
        "datasets": ["atomic-seen", "composite-seen", "composite-unseen"],
        "mesa_root": None,
        "robocasa_base": Path("/coc/pskynet4/chuang475/datasets/robocasa"),
    },
    "mesa": {
        "checkpoint_profile": "finetuned_mesa_pi05",
        "datasets": ["mesa-70", "mesa-instance", "mesa-spatial", "mesa-composite"],
        "mesa_root": Path("/coc/testnvme/xzhang3205/vla-adaptation/envs/mesa-env"),
        "robocasa_base": None,
    },
    "mesa-unfinetuned": {
        "checkpoint_profile": "base_pi05_with_franka_stats",
        "datasets": ["mesa-70", "mesa-instance", "mesa-spatial", "mesa-composite"],
        "mesa_root": Path("/coc/testnvme/xzhang3205/vla-adaptation/envs/mesa-env"),
        "robocasa_base": None,
    },
    "franka": {
        "checkpoint_profile": "finetuned_franka_object_pi05",
        "datasets": ["franka_object", "franka_object_plus", "franka_object_two", "franka_on_top"],
        "mesa_root": None,
        "robocasa_base": None,
    },
}

# Milestone 4 launches all prototype benchmarks.
# ACTIVE_BENCHMARKS = ["robocasa"]
ACTIVE_BENCHMARKS = ["robocasa", "mesa", "mesa-unfinetuned", "franka"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit prototype single-episode static inference jobs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-scope",
        choices=["tmp", "final"],
        default="tmp",
        help="Use tmp for RoboCasa verification, final for /prototype/static.",
    )
    parser.add_argument("--skip-frame", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    return parser.parse_args()


def submit_job(
    *,
    benchmark: str,
    dataset: str,
    checkpoint_dir: Path,
    norm_stats_dir: Path | None,
    mesa_root: Path | None,
    robocasa_base: Path | None,
    output_scope: str,
    skip_frame: int,
    max_frames: int,
    num_steps: int,
    dry_run: bool,
) -> None:
    root_name = "tmp/static" if output_scope == "tmp" else "static"
    output_root = PROTOTYPE_ROOT / root_name / benchmark / dataset / "perturbance-all"
    output_root.mkdir(parents=True, exist_ok=True)

    export_vars = {
        "DATASET": dataset,
        "OUTPUT_ROOT": str(output_root),
        "CHECKPOINT_DIR": str(checkpoint_dir),
        "SKIP_FRAME": str(skip_frame),
        "NUM_STEPS": str(num_steps),
        "MAX_FRAMES": str(max_frames),
        "EMBEDDING_TYPE": "vision",
    }
    if norm_stats_dir is not None:
        export_vars["NORM_STATS_DIR"] = str(norm_stats_dir)
    if mesa_root is not None:
        export_vars["MESA_ROOT"] = str(mesa_root)
    if robocasa_base is not None:
        export_vars["ROBOCASA_BASE"] = str(robocasa_base)

    export_str = ",".join(f"{key}={value}" for key, value in export_vars.items())
    cmd = ["sbatch", f"--export=ALL,{export_str}", "launch_prototype_static.sbatch"]
    print("Submitting:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for benchmark in ACTIVE_BENCHMARKS:
        benchmark_cfg = BENCHMARKS[benchmark]
        profile = CHECKPOINT_PROFILES[benchmark_cfg["checkpoint_profile"]]
        for dataset in benchmark_cfg["datasets"]:
            submit_job(
                benchmark=benchmark,
                dataset=dataset,
                checkpoint_dir=profile["checkpoint_dir"],
                norm_stats_dir=profile["norm_stats_dir"],
                mesa_root=benchmark_cfg["mesa_root"],
                robocasa_base=benchmark_cfg["robocasa_base"],
                output_scope=args.output_scope,
                skip_frame=args.skip_frame,
                max_frames=args.max_frames,
                num_steps=args.num_steps,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
