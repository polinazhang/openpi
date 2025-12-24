#!/usr/bin/env python3
"""Download the OpenArm dataset using the LeRobot v3 loader."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a LeRobot dataset to a local directory.")
    parser.add_argument(
        "--repo-id",
        default="optimal-q/tea_test",
        help="Hugging Face dataset repo_id (default: %(default)s).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/work/nvme/bfbo/xzhang42/datasets/openarm"),
        help="Target directory where the dataset will be stored (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Do not download MP4 videos (saves space, but disables visual inputs).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refreshing the metadata cache before downloading data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_root = args.root.expanduser()
    dataset_root = base_root / Path(args.repo_id)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    # Respect existing cache env vars; only set defaults if unset.
    default_cache = Path("/work/nvme/bfbo/xzhang42/huggingface")
    os.environ.setdefault("HF_HOME", str(default_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(default_cache / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(default_cache / "transformers"))

    print(f"Downloading {args.repo_id!r} to {dataset_root}")
    try:
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=str(dataset_root),
            revision="main",
            download_videos=not args.skip_videos,
            force_cache_sync=args.force_refresh,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the dataset with the installed LeRobot version. "
            "Please ensure you're running lerobot>=0.4 and that the repo supports v3 format."
        ) from exc

    print(
        "Download complete:\n"
        f"- Frames: {dataset.num_frames}\n"
        f"- Episodes: {dataset.num_episodes}\n"
        f"- Cameras: {dataset.meta.camera_keys}\n"
        f"- Dataset root: {dataset.meta.root}"
    )


if __name__ == "__main__":
    main()
