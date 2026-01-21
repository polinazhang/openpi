#!/usr/bin/env python3
"""Convert per-trajectory npy metadata into consolidated HDF5 bundles.

Defaults match the Libero evaluation scripts:
  --source_dir /work/nvme/bfbo/xzhang42/libero_test/debug_suite
  --target_dir /work/nvme/bfbo/xzhang42/libero_test/debug_suite_hd5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from numpy.lib import format as npformat
import multiprocessing as mp


DATASET_NAME = "data"
CHUNK_SIZE = int(os.environ.get("OPENPI_HD5_CHUNK", "1024"))


def _load_metadata(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"metadata.json must be a list, got {type(payload)}")
    return payload


def _synthesize_metadata(source_dir: Path) -> List[Dict]:
    trajectories: List[Dict] = []

    def _sort_key(entry: Path) -> Tuple[int, str]:
        name = entry.name
        if name.isdigit():
            return (0, f"{int(name):08d}")
        return (1, name)

    for traj_dir in sorted(
        (child for child in source_dir.iterdir() if child.is_dir()),
        key=_sort_key,
    ):
        artifacts: Dict[str, str] = {}
        for artifact_path in sorted(traj_dir.glob("*.npy")):
            artifacts[artifact_path.stem] = f"{traj_dir.name}/{artifact_path.name}"

        if not artifacts:
            continue

        trajectories.append(
            {
                "trajectory_rel_dir": traj_dir.name,
                "trajectory_id": traj_dir.name,
                "artifacts": artifacts,
            }
        )

    if not trajectories:
        raise RuntimeError(
            f"Unable to synthesize metadata: no per-trajectory .npy files found under {source_dir}"
        )

    return trajectories


def _sanitize_filename(value: str) -> str:
    return value.replace("/", "__")


def _resolve_artifact_path(source_dir: Path, dataset_name: str, rel_path: str) -> Path:
    candidate = source_dir / rel_path
    if candidate.exists():
        return candidate

    prefix = f"{dataset_name}{os.sep}"
    if rel_path.startswith(prefix):
        trimmed = rel_path[len(prefix) :]
        candidate = source_dir / trimmed
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Missing artifact {rel_path} under {source_dir}")


def _read_npy_header(path: Path) -> Tuple[Tuple[int, ...], np.dtype]:
    with path.open("rb") as handle:
        version = npformat.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = npformat.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _, dtype = npformat.read_array_header_2_0(handle)
        elif version == (3, 0):
            shape, _, dtype = npformat.read_array_header_3_0(handle)
        else:
            raise ValueError(f"Unsupported npy version {version} in {path}")
    return tuple(shape), np.dtype(dtype)


def _inspect_artifact(path: Path) -> Dict:
    _, ext = os.path.splitext(path.name)
    if ext.lower() != ".npy":
        raise ValueError(f"Unsupported artifact type: {path}")

    shape, dtype = _read_npy_header(path)
    length = int(np.prod(shape)) if shape else 1
    return {
        "type": "npy",
        "length": length,
        "raw_shape": shape if shape else (),
        "dtype": dtype.str,
        "has_object": dtype.hasobject,
    }


def _ensure_compatible(info: Dict, inspected: Dict, name: str) -> None:
    if info["dtype"] is None:
        info.update(
            {
                "dtype": inspected["dtype"],
                "type": inspected["type"],
                "has_object": inspected.get("has_object", False),
            }
        )
        return

    if info["dtype"] != inspected["dtype"]:
        raise ValueError(
            f"Inconsistent dtype for {name}: {info['dtype']} vs {inspected['dtype']}"
        )


def _write_dataset(dest: Path, info: Dict, entries: List[Dict]) -> None:
    if info.get("has_object", False):
        raise ValueError(f"Object-dtype artifacts are not supported (saw {dest.name}).")
    dtype = np.dtype(info["dtype"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dest, "w") as handle:
        dataset = handle.create_dataset(
            DATASET_NAME,
            shape=(info["total_length"],),
            dtype=dtype,
            compression="lzf",
            shuffle=True,
        )
        for entry in entries:
            start = entry["offset"]
            length = entry["length"]
            if length == 0:
                continue
            data = np.load(entry["path"], mmap_mode="r")
            reshaped = data.reshape(-1)
            if info["has_object"]:
                reshaped = np.asarray(data, dtype=object).reshape(-1)
            offset = 0
            while offset < length:
                end = min(offset + CHUNK_SIZE, length)
                dataset[start + offset : start + end] = reshaped[offset:end]
                offset = end


def convert_dataset(source_dir: Path, target_dir: Path, workers: int = 0) -> None:
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    dataset_name = source_dir.name
    metadata_path = source_dir / "metadata.json"
    if metadata_path.exists():
        metadata_items = _load_metadata(metadata_path)
    else:
        print(f"No metadata.json found under {source_dir}, synthesizing from directory structure.")
        metadata_items = _synthesize_metadata(source_dir)
    if not metadata_items:
        raise RuntimeError("No trajectories discovered in metadata.")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    artifact_keys = sorted(
        {key for item in metadata_items for key in item.get("artifacts", {})}
    )
    if not artifact_keys:
        raise RuntimeError("No artifacts discovered in metadata.")

    artifact_info: Dict[str, Dict] = {
        key: {
            "file": target_dir / f"{_sanitize_filename(key)}.h5",
            "total_length": 0,
            "dtype": None,
            "has_object": False,
        }
        for key in artifact_keys
    }
    artifact_records: Dict[str, List[Dict]] = {key: [] for key in artifact_keys}

    summary = {
        "dataset_name": dataset_name,
        "source_folder": str(source_dir),
        "artifact_keys": artifact_keys,
        "artifact_files": {
            key: f"{_sanitize_filename(key)}.h5" for key in artifact_keys
        },
        "artifact_dataset": DATASET_NAME,
        "trajectories": [],
    }

    current_offsets = {key: 0 for key in artifact_keys}
    start_time = time.time()

    for item in metadata_items:
        artifact_spans = {}
        artifact_lengths = {}
        artifact_shapes = {}

        for name, rel_path in item.get("artifacts", {}).items():
            abs_path = _resolve_artifact_path(source_dir, dataset_name, rel_path)
            inspected = _inspect_artifact(abs_path)
            info = artifact_info[name]
            _ensure_compatible(info, inspected, name)

            offset = current_offsets[name]
            current_offsets[name] += inspected["length"]
            info["total_length"] += inspected["length"]

            artifact_spans[name] = {"offset": offset, "length": inspected["length"]}
            artifact_lengths[name] = inspected["length"]
            artifact_shapes[name] = list(inspected["raw_shape"])
            artifact_records[name].append(
                {"path": abs_path, "offset": offset, "length": inspected["length"]}
            )

        cleaned = {k: v for k, v in item.items() if k not in {"artifacts", "trajectory_rel_dir"}}
        cleaned["artifact_spans"] = artifact_spans
        cleaned["artifact_lengths"] = artifact_lengths
        cleaned["artifact_shapes"] = artifact_shapes
        summary["trajectories"].append(cleaned)

    summary["artifact_totals"] = {
        key: info["total_length"] for key, info in artifact_info.items()
    }
    summary["artifact_dtypes"] = {key: info["dtype"] for key, info in artifact_info.items()}

    write_args = [
        (info["file"], info, artifact_records[name])
        for name, info in artifact_info.items()
    ]
    worker_target = workers if workers > 0 else (os.cpu_count() or 1)
    worker_target = min(worker_target, len(write_args))
    if worker_target <= 1:
        for args in write_args:
            _write_dataset(*args)
    else:
        with mp.Pool(processes=worker_target) as pool:
            pool.starmap(_write_dataset, write_args)

    with (target_dir / "metadata_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    elapsed = time.time() - start_time
    print(f"Wrote consolidated dataset to {target_dir} (took {elapsed:.2f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw rollout tensors to HD5 bundles.")
    parser.add_argument(
        "--source_dir",
        default="/work/nvme/bfbo/xzhang42/libero_test/debug_suite",
        type=Path,
        help="Directory containing metadata.json and trajectory folders.",
    )
    parser.add_argument(
        "--target_dir",
        default="/work/nvme/bfbo/xzhang42/libero_test/debug_suite_hd5",
        type=Path,
        help="Destination directory for HD5 outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers to use when writing artifacts (0=auto, 1=disable multiprocessing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(
        args.source_dir.resolve(),
        args.target_dir.resolve(),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
