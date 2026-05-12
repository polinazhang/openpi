#!/usr/bin/env python3
"""Submit static inference jobs for the active benchmark config."""

from __future__ import annotations

import datetime as _datetime
import subprocess
from pathlib import Path


CHECKPOINT_PROFILES = {
    "base_pi05_with_franka_stats": {
        "checkpoint_dir": Path("/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/base/pi05_base_torch"),
        "norm_stats_dir": Path(
            "/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/base/pi05_base_torch/assets/franka"
        ),
    },
    "base_pi05_no_override": {
        "checkpoint_dir": Path("/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/base/pi05_base_torch"),
        "norm_stats_dir": None,
    },
    "mesa_pi05_no_override": {
        "checkpoint_dir": Path("/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/mesa/mesa-pi05-torch"),
        "norm_stats_dir": None,
    },
}


RUN_CONFIGS = {
    "robocasa": {
        "folder_type": "robocasa",
        "checkpoint_profile": "base_pi05_with_franka_stats",
        "datasets": [
            "atomic-seen",
            "composite-seen",
            "composite-unseen",
        ],
        "runs": [
            ("perturbance-all", "perturbance-noise", "inference", True, True, True),
        ],
        "skip_frame": 10,
        "max_frames": 0,
        "embedding_type": "vision",
        "mesa_root": None,
        "robocasa_base": Path("/coc/pskynet4/chuang475/datasets/robocasa"),
    },
    "mesa": {
        "folder_type": "mesa",
        "checkpoint_profile": "mesa_pi05_no_override",
        "datasets": [
            "mesa-70",
            "mesa-instance",
            "mesa-spatial",
            "mesa-composite",
        ],
        "runs": [
            ("perturbance-all", "perturbance-noise", "inference", True, True, True),
        ],
        "skip_frame": 10,
        "max_frames": 0,
        "embedding_type": "vision",
        "mesa_root": Path("/coc/testnvme/xzhang3205/vla-adaptation/envs/mesa-env"),
        "robocasa_base": None,
    },
    "franka": {
        "folder_type": "franka",
        "checkpoint_profile": "base_pi05_no_override",
        "datasets": [
            "franka_object",
            "franka_object_plus",
            "franka_object_two",
            "franka_on_top",
        ],
        "runs": [
            ("perturbance-all", "perturbance-noise", "inference", True, True, True),
        ],
        "skip_frame": 10,
        "max_frames": 0,
        "embedding_type": "vision",
        "mesa_root": None,
        "robocasa_base": None,
    },
}


ACTIVE_RUN = RUN_CONFIGS["robocasa"]
# ACTIVE_RUN = RUN_CONFIGS["mesa"]
# ACTIVE_RUN = RUN_CONFIGS["franka"]


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
    norm_stats_dir: Path | None,
    skip_frame: int,
    max_frames: int,
    mesa_root: Path | None = None,
    robocasa_base: Path | None = None,
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
    if norm_stats_dir is not None:
        export_vars["NORM_STATS_DIR"] = str(norm_stats_dir)
    if mesa_root is not None:
        export_vars["MESA_ROOT"] = str(mesa_root)
    if robocasa_base is not None:
        export_vars["ROBOCASA_BASE"] = str(robocasa_base)
    export_str = ",".join([f"{k}={v}" for k, v in export_vars.items()])
    cmd = ["sbatch", f"--export=ALL,{export_str}", "launch_static.sbatch"]
    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, cwd=root_dir, check=True)


if __name__ == "__main__":
    root_dir = Path("/coc/testnvme/xzhang3205/vla-adaptation/models/openpi")
    timestamp = _datetime.datetime.now().strftime("%m-%d-%H%M")
    checkpoint_profile_name = ACTIVE_RUN["checkpoint_profile"]
    folder_name = str(Path(ACTIVE_RUN["folder_type"]) / checkpoint_profile_name / timestamp)
    checkpoint_profile = CHECKPOINT_PROFILES[checkpoint_profile_name]

    for dataset in ACTIVE_RUN["datasets"]:
        for mode, metric, condition, save_meta, save_cosine, save_displacement_trace in ACTIVE_RUN["runs"]:
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
                checkpoint_dir=checkpoint_profile["checkpoint_dir"],
                norm_stats_dir=checkpoint_profile["norm_stats_dir"],
                skip_frame=ACTIVE_RUN["skip_frame"],
                max_frames=ACTIVE_RUN["max_frames"],
                mesa_root=ACTIVE_RUN["mesa_root"],
                robocasa_base=ACTIVE_RUN["robocasa_base"],
                perturbance_step_num=0,
                perturbance_step_size=1e-2,
                # embedding_type="time",
                # embedding_type="vision+action",
                # embedding_type="vision+action+time",
                embedding_type=ACTIVE_RUN["embedding_type"],
            )
