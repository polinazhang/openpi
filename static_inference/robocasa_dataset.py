from __future__ import annotations

import bisect
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq

DEFAULT_DATASET_BASE = Path("/coc/pskynet4/chuang475/datasets/robocasa")
VIDEO_KEY_PREFIX = "observation.images."

TARGET_SPLITS: dict[str, list[str]] = {
    "atomic_seen": [
        "CloseBlenderLid",
        "CloseFridge",
        "CloseToasterOvenDoor",
        "CoffeeSetupMug",
        "NavigateKitchen",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
        "SlideDishwasherRack",
        "TurnOffStove",
        "TurnOnElectricKettle",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
    ],
    "composite_seen": [
        "DeliverStraw",
        "GetToastedBread",
        "KettleBoiling",
        "LoadDishwasher",
        "PackIdenticalLunches",
        "PreSoakPan",
        "PrepareCoffee",
        "RinseSinkBasin",
        "ScrubCuttingBoard",
        "SearingMeat",
        "SetUpCuttingStation",
        "StackBowlsCabinet",
        "SteamInMicrowave",
        "StirVegetables",
        "StoreLeftoversInBowl",
        "WashLettuce",
    ],
    "composite_unseen": [
        "ArrangeBreadBasket",
        "ArrangeTea",
        "BreadSelection",
        "CategorizeCondiments",
        "CuttingToolSelection",
        "GarnishPancake",
        "GatherTableware",
        "HeatKebabSandwich",
        "MakeIceLemonade",
        "PanTransfer",
        "PortionHotDogs",
        "RecycleBottlesByType",
        "SeparateFreezerRack",
        "WaffleReheat",
        "WashFruitColander",
        "WeighIngredients",
    ],
}

BAD_ROBOCASA_EPISODES = {
    Path("/coc/pskynet4/chuang475/datasets/robocasa/v1.0/target/atomic/CloseBlenderLid/20250822/lerobot"): {0},
}


@dataclass(frozen=True)
class TaskDatasetRoot:
    split: str
    task: str
    dataset_root: Path
    task_family: str
    version_dir: Path


def normalize_split_name(split: str) -> str:
    normalized = split.replace("-", "_").removeprefix("target_")
    if normalized not in TARGET_SPLITS:
        raise KeyError(f"Unknown RoboCasa split {split!r}. Expected one of {sorted(TARGET_SPLITS)}")
    return normalized


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _resolve_dataset_base(dataset_base: Path) -> Path:
    dataset_base = dataset_base.expanduser().resolve()
    if not dataset_base.exists():
        raise FileNotFoundError(f"RoboCasa dataset base does not exist: {dataset_base}")
    target_root = dataset_base / "v1.0" / "target"
    if not target_root.exists():
        raise FileNotFoundError(f"RoboCasa target root does not exist: {target_root}")
    return dataset_base


def _task_dataset_root(dataset_base: Path, split: str, task: str) -> TaskDatasetRoot:
    dataset_base = _resolve_dataset_base(dataset_base)
    normalized_split = normalize_split_name(split)
    if task not in TARGET_SPLITS[normalized_split]:
        raise KeyError(f"Task {task!r} is not in RoboCasa split {normalized_split!r}")

    target_root = dataset_base / "v1.0" / "target"
    candidates: list[tuple[str, Path]] = []
    for family in ("atomic", "composite"):
        task_dir = target_root / family / task
        if task_dir.exists():
            candidates.append((family, task_dir))
    if not candidates:
        raise FileNotFoundError(f"No RoboCasa target dataset directory found for task: {task}")
    if len(candidates) > 1:
        raise RuntimeError(f"Task {task!r} exists in multiple RoboCasa target families: {candidates}")

    family, task_dir = candidates[0]
    lerobot_roots = [
        version_dir / "lerobot"
        for version_dir in sorted(path for path in task_dir.iterdir() if path.is_dir())
        if (version_dir / "lerobot").is_dir()
    ]
    if not lerobot_roots:
        raise FileNotFoundError(f"No lerobot directory found under: {task_dir}")
    if len(lerobot_roots) > 1:
        raise RuntimeError(f"Multiple lerobot roots found for task {task!r}: {lerobot_roots}")

    root = lerobot_roots[0]
    for rel_path in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"):
        required_path = root / rel_path
        if not required_path.exists():
            raise FileNotFoundError(f"Required RoboCasa metadata file missing: {required_path}")

    return TaskDatasetRoot(
        split=normalized_split,
        task=task,
        dataset_root=root,
        task_family=family,
        version_dir=root.parent,
    )


def _normalize_camera_key(camera: str) -> str:
    if camera.startswith(VIDEO_KEY_PREFIX):
        return camera
    return f"{VIDEO_KEY_PREFIX}{camera}"


class _RobocasaTaskDataset:
    def __init__(
        self,
        task_root: TaskDatasetRoot,
        action_horizon: int,
        task_offset: int,
        max_episodes: int | None = None,
    ) -> None:
        self.task_root = task_root
        self.root = task_root.dataset_root.expanduser().resolve()
        self.action_horizon = int(action_horizon)
        self.task_offset = int(task_offset)
        self.info = _read_json(self.root / "meta" / "info.json")
        self.tasks = _read_jsonl(self.root / "meta" / "tasks.jsonl")
        self.task_index_to_text = {int(row["task_index"]): row["task"] for row in self.tasks}
        self.chunks_size = int(self.info["chunks_size"])
        self.video_path_pattern = self.info["video_path"]

        bad_episode_ids = BAD_ROBOCASA_EPISODES.get(self.root, set())
        episode_rows = [
            row
            for row in _read_jsonl(self.root / "meta" / "episodes.jsonl")
            if int(row["episode_index"]) not in bad_episode_ids
        ]
        if bad_episode_ids:
            print(f"Skipping known bad RoboCasa episodes for {self.root}: {sorted(bad_episode_ids)}")

        if max_episodes is not None:
            if max_episodes < 0:
                raise ValueError(f"max_episodes must be non-negative, got {max_episodes}")
            episode_rows = episode_rows[:max_episodes]

        self.episode_indices = [int(row["episode_index"]) for row in episode_rows]
        self.episode_lengths = [int(row["length"]) for row in episode_rows]
        self.cumulative_lengths = np.cumsum(self.episode_lengths).tolist()
        self._cached_episode_index: int | None = None
        self._cached_episode_rows: list[dict[str, Any]] | None = None
        self._cached_video_episode: int | None = None
        self._cached_video_frames: dict[str, list[np.ndarray]] = {}

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _episode_chunk(self, episode_index: int) -> int:
        return episode_index // self.chunks_size

    def _episode_data_path(self, episode_index: int) -> Path:
        return (
            self.root
            / "data"
            / f"chunk-{self._episode_chunk(episode_index):03d}"
            / f"episode_{episode_index:06d}.parquet"
        )

    def _episode_video_path(self, episode_index: int, camera_key: str) -> Path:
        relative_path = self.video_path_pattern.format(
            episode_chunk=self._episode_chunk(episode_index),
            video_key=camera_key,
            episode_index=episode_index,
        )
        return self.root / relative_path

    def _load_episode_rows(self, episode_index: int) -> list[dict[str, Any]]:
        if self._cached_episode_index == episode_index and self._cached_episode_rows is not None:
            return self._cached_episode_rows
        path = self._episode_data_path(episode_index)
        rows = pq.read_table(path).to_pylist()
        self._cached_episode_index = episode_index
        self._cached_episode_rows = rows
        return rows

    def _load_video_frames(self, episode_index: int, camera: str) -> list[np.ndarray]:
        if self._cached_video_episode != episode_index:
            self._cached_video_episode = episode_index
            self._cached_video_frames = {}
        camera_key = _normalize_camera_key(camera)
        if camera_key in self._cached_video_frames:
            return self._cached_video_frames[camera_key]
        path = self._episode_video_path(episode_index, camera_key)
        if not path.exists():
            raise FileNotFoundError(f"Missing RoboCasa video file: {path}")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        self._cached_video_frames[camera_key] = frames
        return frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"RoboCasa task frame index out of range: {index}")
        episode_offset = bisect.bisect_right(self.cumulative_lengths, index)
        episode_index = self.episode_indices[episode_offset]
        episode_start = 0 if episode_offset == 0 else self.cumulative_lengths[episode_offset - 1]
        row_index = index - episode_start
        rows = self._load_episode_rows(episode_index)
        row = rows[row_index]

        action_rows = rows[row_index : row_index + self.action_horizon]
        valid_actions = len(action_rows)
        if valid_actions < self.action_horizon:
            action_rows = action_rows + [rows[-1]] * (self.action_horizon - valid_actions)
        action_is_pad = np.asarray(
            [False] * valid_actions + [True] * (self.action_horizon - valid_actions),
            dtype=bool,
        )

        task_index = int(row["task_index"])
        prompt = row.get("annotation.human.task_description.text") or self.task_index_to_text.get(task_index, "")
        source_episode_index = self.task_offset * 1_000_000 + int(episode_index)

        return {
            "observation/image": self._load_video_frames(episode_index, "robot0_agentview_left")[row_index],
            "observation/wrist_image": self._load_video_frames(episode_index, "robot0_eye_in_hand")[row_index],
            "observation/right_image": self._load_video_frames(episode_index, "robot0_agentview_right")[row_index],
            "observation/state": np.asarray(row["observation.state"], dtype=np.float32),
            "actions": np.asarray([r["action"] for r in action_rows], dtype=np.float32),
            "action_is_pad": action_is_pad,
            "prompt": np.asarray(prompt),
            "task": np.asarray(prompt),
            "episode_index": np.asarray(source_episode_index, dtype=np.int64),
            "frame_index": np.asarray(int(row["frame_index"]), dtype=np.int64),
            "task_index": np.asarray(task_index, dtype=np.int64),
            "robocasa_task": self.task_root.task,
        }


class RobocasaSplitDataset:
    def __init__(
        self,
        *,
        dataset_base: Path,
        split: str,
        action_horizon: int,
        tasks_override: list[str] | None = None,
        max_episodes_per_task: int | None = None,
    ) -> None:
        self.dataset_base = _resolve_dataset_base(dataset_base)
        self.split = normalize_split_name(split)
        self.action_horizon = int(action_horizon)
        if tasks_override is not None:
            if not tasks_override:
                raise ValueError("tasks_override must be a non-empty list when provided.")
            split_tasks = set(TARGET_SPLITS[self.split])
            unknown = [task for task in tasks_override if task not in split_tasks]
            if unknown:
                raise KeyError(
                    f"Tasks {unknown!r} are not part of RoboCasa split {self.split!r}. "
                    f"Valid tasks: {sorted(split_tasks)}"
                )
            self.tasks = list(tasks_override)
        else:
            self.tasks = list(TARGET_SPLITS[self.split])
        self.task_datasets = [
            _RobocasaTaskDataset(
                _task_dataset_root(self.dataset_base, self.split, task),
                action_horizon=self.action_horizon,
                task_offset=task_offset,
                max_episodes=max_episodes_per_task,
            )
            for task_offset, task in enumerate(self.tasks)
        ]
        lengths = [len(dataset) for dataset in self.task_datasets]
        self.cumulative_lengths = np.cumsum(lengths).tolist()
        self.episode_indices = [
            int(sample_episode)
            for task_dataset in self.task_datasets
            for sample_episode in (
                task_dataset.task_offset * 1_000_000 + int(episode_index)
                for episode_index in task_dataset.episode_indices
            )
        ]

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"RoboCasa split frame index out of range: {index}")
        task_offset = bisect.bisect_right(self.cumulative_lengths, index)
        task_start = 0 if task_offset == 0 else self.cumulative_lengths[task_offset - 1]
        return self.task_datasets[task_offset][index - task_start]


def load_robocasa_split_dataset(
    *,
    dataset_base: Path,
    split: str,
    action_horizon: int,
    tasks_override: list[str] | None = None,
    max_episodes_per_task: int | None = None,
) -> RobocasaSplitDataset:
    return RobocasaSplitDataset(
        dataset_base=dataset_base,
        split=split,
        action_horizon=action_horizon,
        tasks_override=tasks_override,
        max_episodes_per_task=max_episodes_per_task,
    )
