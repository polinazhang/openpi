#!/usr/bin/env python3
"""Convert the OpenArm dataset into Pi-ready LeRobot datasets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="optimal-q/tea_test",
        help="Repo ID for the raw dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/work/nvme/bfbo/xzhang42/datasets/openarm"),
        help="Directory containing the downloaded raw dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/work/nvme/bfbo/xzhang42/datasets/openarm_processed"),
        help="Parent directory for the processed datasets.",
    )
    parser.add_argument(
        "--continuous-repo",
        default="openarm/tea_continuous",
        help="Relative repo ID for the continuous-gripper dataset.",
    )
    parser.add_argument(
        "--discrete-repo",
        default="openarm/tea_discrete",
        help="Relative repo ID for the discrete-gripper dataset.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize gripper values for the discrete variant.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional limit on number of episodes to convert (for debugging).",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Do not copy MP4 videos into the processed datasets (images saved as PNGs).",
    )
    return parser.parse_args()


def make_features(image_shape: tuple[int, int, int], state_names: list[str], action_names: list[str]) -> dict:
    return {
        "left_wrist_image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "right_wrist_image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "prompt": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "actions": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
    }


def prepare_output_dataset(
    repo_id: str,
    output_root: Path,
    fps: int,
    features: dict,
) -> LeRobotDataset:
    dataset_root = output_root / Path(*repo_id.split("/"))
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=str(dataset_root),
        robot_type="openarm",
        fps=fps,
        features=features,
        use_videos=False,
        image_writer_threads=4,
        image_writer_processes=4,
    )


def to_numpy_image(tensor) -> np.ndarray:
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu().numpy()
    if tensor.ndim == 3:  # (C, H, W) -> (H, W, C)
        tensor = np.moveaxis(tensor, 0, -1)
    tensor = np.clip(tensor, 0.0, 1.0)
    return (tensor * 255).astype(np.uint8)


def to_numpy_array(arr, length: int | None = None) -> np.ndarray:
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    np_arr = np.asarray(arr, dtype=np.float32)
    if length is not None:
        np_arr = np_arr[:length]
    return np_arr.astype(np.float32)


def convert_dataset(
    raw_dataset: LeRobotDataset,
    ds_cont: LeRobotDataset,
    ds_disc: LeRobotDataset,
    *,
    max_episodes: int | None,
    gripper_threshold: float,
) -> None:
    current_episode = None
    processed_episodes = 0

    def flush():
        if ds_cont.episode_buffer["size"] > 0:
            ds_cont.save_episode()
        if ds_disc.episode_buffer["size"] > 0:
            ds_disc.save_episode()

    iterator = tqdm(range(len(raw_dataset)), desc="Converting frames")
    for idx in iterator:
        sample = raw_dataset[idx]
        ep_idx = int(sample["episode_index"].item())
        if current_episode is None:
            current_episode = ep_idx
        elif ep_idx != current_episode:
            flush()
            current_episode = ep_idx
            processed_episodes += 1
            if max_episodes is not None and processed_episodes >= max_episodes:
                break

        left_img = to_numpy_image(sample["observation.images.left_cam"])
        right_img = to_numpy_image(sample["observation.images.right_cam"])
        state_vec = to_numpy_array(sample["observation.state"], length=16)
        action_vec = to_numpy_array(sample["action"])
        task_value = sample["task"]
        if isinstance(task_value, bytes):
            task_value = task_value.decode("utf-8")

        cont_state = state_vec.copy()
        cont_action = action_vec.copy()
        disc_state = state_vec.copy()
        disc_action = action_vec.copy()
        disc_state[-2:] = (disc_state[-2:] >= gripper_threshold).astype(np.float32)
        disc_action[-2:] = (disc_action[-2:] >= gripper_threshold).astype(np.float32)

        frame_common = {
            "left_wrist_image": left_img,
            "right_wrist_image": right_img,
            "task": task_value,
            "prompt": task_value,
        }
        ds_cont.add_frame({**frame_common, "state": cont_state, "actions": cont_action})
        ds_disc.add_frame({**frame_common, "state": disc_state, "actions": disc_action})

    flush()
    ds_cont.finalize()
    ds_disc.finalize()


def main() -> None:
    args = parse_args()
    base_root = args.input_root.expanduser()
    dataset_root = base_root / Path(args.repo_id)
    output_root = args.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    raw_dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=str(dataset_root),
        revision="main",
        download_videos=not args.skip_videos,
    )

    image_shape = tuple(raw_dataset.meta.features["observation.images.left_cam"]["shape"])
    state_names = raw_dataset.meta.features["observation.state"]["names"][:16]
    action_names = raw_dataset.meta.features["action"]["names"]
    features = make_features(image_shape, state_names, action_names)

    ds_cont = prepare_output_dataset(args.continuous_repo, output_root, raw_dataset.fps, features)
    ds_disc = prepare_output_dataset(args.discrete_repo, output_root, raw_dataset.fps, features)

    convert_dataset(
        raw_dataset,
        ds_cont,
        ds_disc,
        max_episodes=args.max_episodes,
        gripper_threshold=args.gripper_threshold,
    )

    cont_dir = output_root / Path(*args.continuous_repo.split("/"))
    disc_dir = output_root / Path(*args.discrete_repo.split("/"))
    print("Conversion complete:")
    print(f"  Continuous dataset: {cont_dir}")
    print(f"  Discrete dataset:   {disc_dir}")


if __name__ == "__main__":
    main()
