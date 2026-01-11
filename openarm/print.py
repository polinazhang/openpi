#!/usr/bin/env python3
"""Summarize EDR metrics computed by openarm/compute.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import h5py
import numpy as np


DEFAULT_DATA_DIR = Path("/work/nvme/bfbo/xzhang42/libero_test/debug_suite_hd5")
METRIC_NAMES = ("norm", "scaled_norm", "cosine", "scale")


def _load_h5(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["data"], dtype=np.float32)


def _discover_layers(data_dir: Path) -> Iterable[int]:
    indices: list[int] = []
    for path in data_dir.glob("scaled_norm_layer_*.h5"):
        suffix = path.stem.split("_")[-1]
        if suffix.isdigit():
            indices.append(int(suffix))
    if not indices:
        raise RuntimeError(f"No scaled_norm_layer_*.h5 files found in {data_dir}")
    indices.sort()
    return indices


def summarize(data_dir: Path) -> Dict[int, Dict[str, Dict[str, float]]]:
    results: Dict[int, Dict[str, Dict[str, float]]] = {}
    for layer_idx in _discover_layers(data_dir):
        layer_stats: Dict[str, Dict[str, float]] = {}
        for metric in METRIC_NAMES:
            path = data_dir / f"{metric}_layer_{layer_idx}.h5"
            if not path.exists():
                continue
            values = _load_h5(path)
            layer_stats[metric] = {
                "mean": float(np.nanmean(values)),
                "var": float(np.nanvar(values)),
            }
        results[layer_idx] = layer_stats
    return results


def print_summary(stats: Dict[int, Dict[str, Dict[str, float]]]) -> None:
    header_metrics = [m for m in METRIC_NAMES if any(m in stat for stat in stats.values())]
    col_titles = ["Layer"]
    for metric in header_metrics:
        col_titles.extend([f"{metric}_mean", f"{metric}_var"])
    col_widths = [max(len(title), 5) for title in col_titles]

    def _format_row(values):
        return "  ".join(f"{val:>{width}}" for val, width in zip(values, col_widths))

    print(_format_row(col_titles))
    for layer_idx, metric_stats in stats.items():
        row = [str(layer_idx)]
        for metric in header_metrics:
            if metric in metric_stats:
                row.append(f"{metric_stats[metric]['mean']:.6f}")
                row.append(f"{metric_stats[metric]['var']:.6f}")
            else:
                row.extend(["N/A", "N/A"])
        print(_format_row(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print EDR metric statistics per layer.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing *layer_*.h5 metrics produced by openarm/compute.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = summarize(args.data_dir.resolve())
    print_summary(stats)


if __name__ == "__main__":
    main()
