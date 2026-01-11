#!/usr/bin/env python3
"""Compute EDR diagnostics (raw/scale-adjusted residuals + cosine) for Pi0.5 trajectories.

The script expects an HD5 bundle produced by `openarm/convert_hd5.py`, i.e. a folder that
contains:
  - metadata_summary.json
  - actions.h5
  - diffusion_noise.h5
  - vt_layer_*.h5  (one per action-expert layer)

It will write:
  - target_residual.h5
  - norm_layer_{k}.h5
  - scaled_norm_layer_{k}.h5
  - cosine_layer_{k}.h5
  - scale_layer_{k}.h5   (per-step optimal scaling factor)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


EPS = 1e-8
DEFAULT_DATA_DIR = Path("/work/nvme/bfbo/xzhang42/libero_test/debug_suite_hd5")


def _load_summary(data_dir: Path) -> dict:
    summary_path = data_dir / "metadata_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"metadata_summary.json not found in {data_dir}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_h5(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["data"], dtype=np.float32)


def _write_h5(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=array.reshape(-1), dtype=np.float32)


def _reshape(array: np.ndarray, *, action_dim: int) -> np.ndarray:
    if array.size % action_dim != 0:
        raise ValueError(
            f"Array of length {array.size} cannot be reshaped into (-1, {action_dim})."
        )
    return array.reshape(-1, action_dim)


def _discover_layers(data_dir: Path) -> Iterable[int]:
    indices: list[int] = []
    for path in data_dir.glob("vt_layer_*.h5"):
        suffix = path.stem.split("_")[-1]
        if suffix.isdigit():
            indices.append(int(suffix))
    if not indices:
        raise RuntimeError(f"No vt_layer_*.h5 files found in {data_dir}")
    indices.sort()
    return indices


def _compute_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    predicted = predicted.astype(np.float32, copy=False)
    target = target.astype(np.float32, copy=False)

    dot = np.sum(predicted * target, axis=1)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    cosine = dot / (predicted_norm * target_norm + EPS)

    scale = dot / (np.square(predicted_norm) + EPS)
    scaled = scale[:, None] * predicted
    scaled_residual = scaled - target
    scaled_norm = np.linalg.norm(scaled_residual, axis=1)

    residual = predicted - target
    raw_norm = np.linalg.norm(residual, axis=1)

    return {
        "norm": raw_norm,
        "scaled_norm": scaled_norm,
        "cosine": cosine,
        "scale": scale,
    }


def convert(data_dir: Path) -> None:
    summary = _load_summary(data_dir)
    action_shape = summary["trajectories"][0]["artifact_shapes"]["actions"]
    if not action_shape:
        raise ValueError("Unable to infer action shape from metadata_summary.json")
    action_dim = int(action_shape[-1])

    actions = _reshape(_load_h5(data_dir / "actions.h5"), action_dim=action_dim)
    diffusion = _reshape(_load_h5(data_dir / "diffusion_noise.h5"), action_dim=action_dim)
    target = diffusion - actions
    _write_h5(data_dir / "target_residual.h5", target)

    layer_indices = _discover_layers(data_dir)
    for layer_idx in layer_indices:
        vt_path = data_dir / f"vt_layer_{layer_idx}.h5"
        vt = _reshape(_load_h5(vt_path), action_dim=action_dim)
        metrics = _compute_metrics(vt, target)
        _write_h5(data_dir / f"norm_layer_{layer_idx}.h5", metrics["norm"])
        _write_h5(data_dir / f"scaled_norm_layer_{layer_idx}.h5", metrics["scaled_norm"])
        _write_h5(data_dir / f"cosine_layer_{layer_idx}.h5", metrics["cosine"])
        _write_h5(data_dir / f"scale_layer_{layer_idx}.h5", metrics["scale"])
        print(f"Computed metrics for layer {layer_idx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute EDR metrics for Pi0.5 trajectories.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing actions.h5, diffusion_noise.h5, and vt_layer_*.h5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args.data_dir.resolve())


if __name__ == "__main__":
    main()
