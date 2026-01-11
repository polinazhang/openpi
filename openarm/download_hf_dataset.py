#!/usr/bin/env python3
"""Download arbitrary Hugging Face dataset repos for OpenArm workflows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DATA_ROOT = Path("/work/nvme/bfbo/xzhang42/datasets")
HF_CACHE = Path("/work/nvme/bfbo/xzhang42/huggingface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset repo for OpenArm runs.")
    parser.add_argument(
        "repo_id",
        nargs="?",
        default="qrafty-ai/tea_use_spoon",
        help="Dataset repo on the Hugging Face Hub (default: %(default)s).",
    )
    return parser.parse_args()


def configure_hf_cache() -> None:
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "transformers"))


def main() -> None:
    args = parse_args()
    configure_hf_cache()

    target_dir = (DATA_ROOT / args.repo_id).expanduser()
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        print(f"{target_dir} already exists. Remove it first if you need a fresh download.")
        return

    print(f"Downloading {args.repo_id!r} to {target_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Snapshot ready at {target_dir}")


if __name__ == "__main__":
    main()
