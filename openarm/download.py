#!/usr/bin/env python3
"""Download the private OpenArm dataset (optimal-q/tea_test) in LeRobot format.

This script relies on LeRobot's dataset API instead of the generic Hugging Face
`datasets` loader to make sure the on-disk structure (meta/data/videos) is fully
compatible with the rest of the OpenPI tooling.

Example:
    python openarm/download.py --root /work/nvme/bfbo/xzhang42/datasets
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from huggingface_hub import snapshot_download

INFO_PATH = "meta/info.json"
TASKS_JSONL = "meta/tasks.jsonl"
TASKS_PARQUET = "meta/tasks.parquet"


def _default_cache_dir() -> Path:
    return Path("/work/nvme/bfbo/xzhang42/.cache/huggingface")


def _default_dataset_root() -> Path:
    return Path("/work/nvme/bfbo/xzhang42/lerobot_datasets")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a LeRobot dataset to local storage.")
    parser.add_argument(
        "--repo-id",
        default="optimal-q/tea_test",
        help="HuggingFace dataset repo_id (default: %(default)s)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_dataset_root(),
        help="Directory that will contain the dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=_default_cache_dir(),
        help="Directory for HuggingFace caches (HF_HOME/HF_DATASETS_CACHE). Defaults to a path in fast storage.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of episode indices to download. (Currently not supported; reserved for future use.)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision/tag of the dataset to pull (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="If set, videos under the 'videos/' tree will not be downloaded.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    hf_home = args.hf_home.expanduser()
    hf_home.mkdir(parents=True, exist_ok=True)
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(hf_home)
    if "HF_DATASETS_CACHE" not in os.environ:
        os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    if "TRANSFORMERS_CACHE" not in os.environ:
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")

    dataset_root = args.root.expanduser()
    dataset_root.mkdir(parents=True, exist_ok=True)

    if args.episodes:
        raise NotImplementedError("--episodes filtering is not yet implemented for snapshot downloads.")

    print(f"Downloading {args.repo_id!r} to {dataset_root}")

    ignore_patterns = "videos/" if args.skip_videos else None
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(dataset_root),
        local_dir_use_symlinks=False,
        ignore_patterns=ignore_patterns,
    )

    info_path = dataset_root / INFO_PATH
    if not info_path.exists():
        raise FileNotFoundError(
            f"Expected dataset metadata at {info_path}, but it was not found after download. "
            "Please ensure you have access to this dataset and that it follows the LeRobot layout."
        )

    print(f"Download finished. Metadata available at {info_path}")
    tasks_path = dataset_root / TASKS_JSONL
    if not tasks_path.exists():
        parquet_path = dataset_root / TASKS_PARQUET
        if parquet_path.exists():
            print(
                "Note: tasks.jsonl not found, but tasks.parquet exists. Dataset is likely using LeRobot v3 format.\n"
                "You may need to convert stats/tasks to the latest format with "
                "`python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py --repo-id=<repo>` "
                "before loading it with older tooling."
            )
        else:
            print("Warning: Neither tasks.jsonl nor tasks.parquet were found under meta/.")


if __name__ == "__main__":
    main()
