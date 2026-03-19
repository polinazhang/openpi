#!/usr/bin/env python3
"""Post-process static gradient outputs into aggregate CSV metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

STATIC_ROOT = Path("/coc/testnvme/xzhang3205/static")
FOLDER_NAME = "franka_full"
DATASETS = ["franka_object", "franka_object_plus", "franka_object_two", "franka_on_top"]
DEFAULT_OUTPUT_CSV = Path("/coc/testnvme/xzhang3205/openpi/static_results/result_gradient.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate gradient metrics from static outputs.")
    parser.add_argument(
        "--root-folder",
        default=FOLDER_NAME,
        help="Folder under static root to read from (default: franka_full).",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset list (default: all franka_full datasets).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path (default: static_results/result_gradient.csv).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=10.0,
        help="Beta used for post-hoc gradient scaling (default: 10.0).",
    )
    return parser.parse_args()


def _load_metadata(path: Path) -> list[dict]:
    with (path / "metadata.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_artifact(root: Path, entry: dict, name: str) -> np.ndarray:
    rel = entry["artifacts"][name]
    return np.load(root / rel).astype(np.float64, copy=False)


def _find_inference_steps(entry: dict) -> list[int]:
    steps: list[int] = []
    for key in entry["artifacts"]:
        if key.startswith("gradient_step_"):
            steps.append(int(key.split("_")[-1]))
    return sorted(steps)


def _safe_stats(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def _trajectory_metric_means(
    train_root: Path,
    infer_root: Path,
    train_entry: dict,
    infer_entry: dict,
    beta: float,
) -> dict[str, float]:
    v_action = _load_artifact(train_root, train_entry, "gradient_step_0")  # [F, H, D]
    final_train = _load_artifact(train_root, train_entry, "final_layer_loss")  # [F, 1, H]

    steps = _find_inference_steps(infer_entry)
    v_all_steps = [_load_artifact(infer_root, infer_entry, f"gradient_step_{s}") for s in steps]
    tau_steps = [_load_artifact(infer_root, infer_entry, f"tau_step_{s}") for s in steps]
    final_infer = _load_artifact(infer_root, infer_entry, "final_layer_loss")  # [F, S, H]

    if len(v_all_steps) == 0:
        raise ValueError("No inference gradient_step_* artifacts found.")
    if final_infer.shape[1] != len(v_all_steps):
        raise ValueError("Mismatch between inference final loss step count and gradient step count.")

    # Repeat condition-training gradient for each inference step so dimensions align.
    v_action_4d = np.repeat(v_action[:, None, :, :], len(v_all_steps), axis=1)  # [F, S, H, D]
    v_all_4d = np.stack(v_all_steps, axis=1)  # [F, S, H, D]
    tau_2d = np.stack([x.reshape(-1) for x in tau_steps], axis=1)  # [F, S]

    # Post-hoc scaling term, beta-clipped.
    tau_sq = np.square(tau_2d)
    one_minus = 1.0 - tau_2d
    r_tau_sq = np.square(one_minus) / (tau_sq + np.square(one_minus) + 1e-12)
    scale = np.minimum(beta, one_minus / (tau_2d * r_tau_sq + 1e-12))  # [F, S]
    scale = scale[:, :, None, None]

    v_action_scaled = scale * v_action_4d
    v_all_scaled = scale * v_all_4d
    v_vision_scaled = v_all_scaled - v_action_scaled

    def l2_and_l2sq(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        l2_sq = np.sum(np.square(v), axis=(-1, -2))  # [F, S]
        l2 = np.sqrt(l2_sq)
        return l2, l2_sq

    action_l2, action_l2_sq = l2_and_l2sq(v_action_scaled)
    vision_l2, vision_l2_sq = l2_and_l2sq(v_vision_scaled)
    all_l2, all_l2_sq = l2_and_l2sq(v_all_scaled)

    return {
        "v_action_l2_mean": float(np.mean(action_l2)),
        "v_action_l2_sq_mean": float(np.mean(action_l2_sq)),
        "v_vision_l2_mean": float(np.mean(vision_l2)),
        "v_vision_l2_sq_mean": float(np.mean(vision_l2_sq)),
        "v_all_l2_mean": float(np.mean(all_l2)),
        "v_all_l2_sq_mean": float(np.mean(all_l2_sq)),
        "final_loss_training_mean": float(np.mean(final_train)),
        "final_loss_inference_mean": float(np.mean(final_infer)),
    }


def process_dataset(root: Path, dataset: str, beta: float) -> list[dict]:
    train_root = root / dataset / "gradient-training"
    infer_root = root / dataset / "gradient-inference"
    if not train_root.exists() or not infer_root.exists():
        raise FileNotFoundError(f"Missing condition folders for dataset={dataset}: {train_root} or {infer_root}")

    train_meta = _load_metadata(train_root)
    infer_meta = _load_metadata(infer_root)
    if len(train_meta) != len(infer_meta):
        raise ValueError(f"Trajectory count mismatch for dataset={dataset}: training={len(train_meta)}, inference={len(infer_meta)}")

    traj_rows: list[dict] = []
    for train_entry, infer_entry in zip(train_meta, infer_meta, strict=True):
        traj_rows.append(_trajectory_metric_means(train_root, infer_root, train_entry, infer_entry, beta))

    metrics = sorted(traj_rows[0].keys()) if traj_rows else []
    rows: list[dict] = []
    for metric in metrics:
        mean, std = _safe_stats([row[metric] for row in traj_rows])
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "mean": mean,
                "std": std,
                "num_trajectories": len(traj_rows),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "metric", "mean", "std", "num_trajectories"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict]) -> None:
    print("dataset,metric,mean,std,num_trajectories")
    for row in rows:
        print(
            f"{row['dataset']},{row['metric']},"
            f"{row['mean']:.8f},{row['std']:.8f},{row['num_trajectories']}"
        )


def main() -> None:
    args = parse_args()
    root = STATIC_ROOT / args.root_folder
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    all_rows: list[dict] = []
    for dataset in datasets:
        all_rows.extend(process_dataset(root, dataset, args.beta))

    write_csv(all_rows, args.output_csv)
    print_rows(all_rows)
    print(f"\nWrote: {args.output_csv}")


if __name__ == "__main__":
    main()
