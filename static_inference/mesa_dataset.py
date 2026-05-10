from __future__ import annotations

import bisect
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq

DEFAULT_MESA_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation/envs/mesa-env")
DEFAULT_DATASET_ROOT = Path("/coc/testnvme/xzhang3205/lerobot/mesa")
CACHE_FILENAME = "mesa_suite_index.json"
SUITE_NAMES = ("mesa-70", "mesa-instance", "mesa-spatial", "mesa-category", "mesa-composite")


@dataclass(frozen=True)
class SuiteIndex:
    dataset_root: str
    cache_path: str
    suite_to_canonical_task_ids: dict[str, list[str]]
    suite_to_task_strings: dict[str, list[str]]
    suite_to_task_indices: dict[str, list[int]]
    suite_to_episode_indices: dict[str, list[int]]
    task_string_to_task_index: dict[str, int]
    source_metadata: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any]) -> "SuiteIndex":
        return cls(**payload)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_dataset_root(dataset_root: Path) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"MESA dataset root does not exist: {dataset_root}")
    return dataset_root


def _load_suite_task_ids(mesa_root: Path) -> dict[str, list[str]]:
    task_sets_path = mesa_root / "mesa" / "task_suites" / "task_sets.py"
    spec = importlib.util.spec_from_file_location("_mesa_task_sets", task_sets_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load MESA task suite definitions from {task_sets_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing_suites = [suite_name for suite_name in SUITE_NAMES if suite_name not in module.EVAL_SETS]
    if missing_suites:
        raise KeyError(f"Missing MESA suite definitions: {missing_suites}")
    return {suite_name: list(module.EVAL_SETS[suite_name]["tasks"]) for suite_name in SUITE_NAMES}


def _load_task_instruction_mapping(mesa_root: Path) -> dict[str, str]:
    mapping_path = mesa_root / "mesa" / "task_suites" / "task_to_instr_mapping.json"
    return _read_json(mapping_path)


def _build_task_string_index(task_rows: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in task_rows:
        task_string = row["task"].strip().lower()
        task_index = int(row["task_index"])
        if task_string in mapping and mapping[task_string] != task_index:
            raise ValueError(f"Task string maps to multiple task_index values: {task_string}")
        mapping[task_string] = task_index
    return mapping


def build_suite_index(dataset_root: Path, cache_path: Path, mesa_root: Path = DEFAULT_MESA_ROOT) -> SuiteIndex:
    dataset_root = _resolve_dataset_root(dataset_root)
    cache_path = cache_path.expanduser().resolve()
    mesa_root = mesa_root.expanduser().resolve()
    suite_to_canonical_task_ids = _load_suite_task_ids(mesa_root)
    task_id_to_instruction = _load_task_instruction_mapping(mesa_root)
    task_rows = _read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    episode_rows = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    task_string_to_task_index = _build_task_string_index(task_rows)

    suite_to_task_strings = {
        suite_name: [task_id_to_instruction[task_id].strip().lower() for task_id in task_ids]
        for suite_name, task_ids in suite_to_canonical_task_ids.items()
    }
    suite_to_task_indices = {
        suite_name: [
            task_string_to_task_index[task_string]
            for task_string in task_strings
            if task_string in task_string_to_task_index
        ]
        for suite_name, task_strings in suite_to_task_strings.items()
    }
    missing_suite_task_strings = {
        suite_name: [task_string for task_string in task_strings if task_string not in task_string_to_task_index]
        for suite_name, task_strings in suite_to_task_strings.items()
    }
    suite_task_sets = {suite_name: set(task_strings) for suite_name, task_strings in suite_to_task_strings.items()}
    suite_to_episode_indices = {suite_name: [] for suite_name in suite_to_task_strings}
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        episode_tasks = {task.strip().lower() for task in row["tasks"]}
        for suite_name, suite_tasks in suite_task_sets.items():
            if episode_tasks & suite_tasks:
                suite_to_episode_indices[suite_name].append(episode_index)

    info = _read_json(dataset_root / "meta" / "info.json")
    suite_index = SuiteIndex(
        dataset_root=str(dataset_root),
        cache_path=str(cache_path),
        suite_to_canonical_task_ids=suite_to_canonical_task_ids,
        suite_to_task_strings=suite_to_task_strings,
        suite_to_task_indices=suite_to_task_indices,
        suite_to_episode_indices=suite_to_episode_indices,
        task_string_to_task_index=task_string_to_task_index,
        source_metadata={
            "dataset_root": str(dataset_root),
            "codebase_version": info.get("codebase_version"),
            "total_tasks": info.get("total_tasks"),
            "total_episodes": info.get("total_episodes"),
            "missing_suite_task_strings": missing_suite_task_strings,
        },
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(suite_index.to_jsonable(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return suite_index


def load_or_build_suite_index(
    dataset_root: Path,
    cache_path: Path,
    mesa_root: Path = DEFAULT_MESA_ROOT,
) -> SuiteIndex:
    cache_path = cache_path.expanduser().resolve()
    if cache_path.exists():
        return SuiteIndex.from_jsonable(_read_json(cache_path))
    return build_suite_index(dataset_root, cache_path, mesa_root)


class MesaSuiteDataset:
    def __init__(
        self,
        *,
        dataset_root: Path,
        suite_index: SuiteIndex,
        suite_name: str,
        action_horizon: int,
    ) -> None:
        if suite_name not in suite_index.suite_to_episode_indices:
            raise KeyError(f"Unknown MESA suite {suite_name!r}. Expected one of {sorted(SUITE_NAMES)}")
        self.root = _resolve_dataset_root(dataset_root)
        self.suite_name = suite_name
        self.action_horizon = int(action_horizon)
        self.episode_indices = list(suite_index.suite_to_episode_indices[suite_name])
        self.task_index_to_string = {
            int(task_index): task_string
            for task_string, task_index in suite_index.task_string_to_task_index.items()
        }
        info = _read_json(self.root / "meta" / "info.json")
        self.chunks_size = int(info["chunks_size"])
        self._episode_lengths = self._read_episode_lengths()
        self._cumulative_lengths = np.cumsum(self._episode_lengths).tolist()
        self._cached_episode_index: int | None = None
        self._cached_episode_rows: list[dict[str, Any]] | None = None
        self._cached_video_episode: int | None = None
        self._cached_video_frames: dict[str, list[np.ndarray]] = {}

    def _read_episode_lengths(self) -> list[int]:
        length_by_episode = {
            int(row["episode_index"]): int(row["length"])
            for row in _read_jsonl(self.root / "meta" / "episodes.jsonl")
        }
        return [length_by_episode[episode_index] for episode_index in self.episode_indices]

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
        return (
            self.root
            / "videos"
            / f"chunk-{self._episode_chunk(episode_index):03d}"
            / camera_key
            / f"episode_{episode_index:06d}.mp4"
        )

    def _load_episode_rows(self, episode_index: int) -> list[dict[str, Any]]:
        if self._cached_episode_index == episode_index and self._cached_episode_rows is not None:
            return self._cached_episode_rows
        rows = pq.read_table(self._episode_data_path(episode_index)).to_pylist()
        self._cached_episode_index = episode_index
        self._cached_episode_rows = rows
        return rows

    def _load_video_frames(self, episode_index: int, camera_key: str) -> list[np.ndarray]:
        if self._cached_video_episode != episode_index:
            self._cached_video_episode = episode_index
            self._cached_video_frames = {}
        if camera_key in self._cached_video_frames:
            return self._cached_video_frames[camera_key]
        path = self._episode_video_path(episode_index, camera_key)
        if not path.exists():
            raise FileNotFoundError(f"Missing MESA video file: {path}")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        self._cached_video_frames[camera_key] = frames
        return frames

    def __len__(self) -> int:
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"MESA frame index out of range: {index}")
        episode_offset = bisect.bisect_right(self._cumulative_lengths, index)
        episode_index = self.episode_indices[episode_offset]
        episode_start = 0 if episode_offset == 0 else self._cumulative_lengths[episode_offset - 1]
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

        joint_pos = np.asarray(row["robot0_joint_pos"], dtype=np.float32)
        gripper_width = np.asarray([row["robot0_gripper_jaw_width"]], dtype=np.float32)
        task_index = int(row["task_index"])
        prompt = self.task_index_to_string.get(task_index, "")
        return {
            "leftshoulder_image": self._load_video_frames(episode_index, "leftshoulder_image")[row_index],
            "robot0_eye_in_hand_image": self._load_video_frames(episode_index, "robot0_eye_in_hand_image")[row_index],
            "robot0_joint_pos+robot0_gripper_jaw_width": np.concatenate([joint_pos, gripper_width], axis=0),
            "actions_joint_pos": np.asarray([r["actions_joint_pos"] for r in action_rows], dtype=np.float32),
            "action_is_pad": action_is_pad,
            "prompt": np.asarray(prompt),
            "task": np.asarray(prompt),
            "episode_index": np.asarray(episode_index, dtype=np.int64),
            "frame_index": np.asarray(int(row["frame_index"]), dtype=np.int64),
            "task_index": np.asarray(task_index, dtype=np.int64),
        }


def load_mesa_suite_dataset(
    *,
    dataset_root: Path,
    suite_name: str,
    action_horizon: int,
    mesa_root: Path = DEFAULT_MESA_ROOT,
) -> MesaSuiteDataset:
    dataset_root = _resolve_dataset_root(dataset_root)
    cache_path = dataset_root / "meta" / CACHE_FILENAME
    suite_index = load_or_build_suite_index(dataset_root, cache_path, mesa_root)
    return MesaSuiteDataset(
        dataset_root=dataset_root,
        suite_index=suite_index,
        suite_name=suite_name,
        action_horizon=action_horizon,
    )
