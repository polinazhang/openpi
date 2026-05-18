#!/usr/bin/env python3
"""Run static inference on one source episode without scanning unrelated frames."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

import static_inference as _static
import mesa_dataset as _mesa_dataset
import robocasa_dataset as _robocasa_dataset

from openpi.shared import download
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


@dataclass(frozen=True)
class EpisodeSlice:
    source_episode_index: int
    local_episode_index: int
    start: int
    stop: int
    label: str
    benchmark: str
    dataset_name: str
    dataset_root: Path
    fps: int
    camera_names: tuple[str, ...]
    action_key: str

    @property
    def num_frames(self) -> int:
        return self.stop - self.start

    def limited(self, max_frames: int | None) -> "EpisodeSlice":
        if max_frames is None or max_frames <= 0 or self.num_frames <= max_frames:
            return self
        return dataclasses.replace(self, stop=self.start + max_frames)


class EpisodeDatasetView:
    def __init__(self, dataset, episode_slice: EpisodeSlice) -> None:
        self._dataset = dataset
        self.episode_slice = episode_slice

    def __len__(self) -> int:
        return self.episode_slice.num_frames

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(f"Episode frame index out of range: {index}")
        return self._dataset[self.episode_slice.start + index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run static inference on exactly one source episode.")
    parser.add_argument("--dataset", choices=sorted(_static.DATASETS.keys()), required=True)
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help=(
            "Source episode id to run. For RoboCasa, use the synthetic id already saved by the "
            "loader: task_offset * 1000000 + local_episode_index."
        ),
    )
    parser.add_argument(
        "--use-first-episode",
        type=_static.parse_bool,
        default=False,
        help="If true, use the first episode exposed by the selected dataset/suite/split.",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "gradient", "perturbance", "perturbance-noise"],
        default="perturbance-noise",
    )
    parser.add_argument("--condition", choices=["training", "inference"], default="inference")
    parser.add_argument("--save_meta", type=_static.parse_bool, default=True)
    parser.add_argument("--save_cosine", type=_static.parse_bool, default=True)
    parser.add_argument("--save_displacement_trace", type=_static.parse_bool, default=True)
    parser.add_argument("--perturbance_step_num", type=int, default=0)
    parser.add_argument("--perturbance_step_size", type=float, default=1e-2)
    parser.add_argument(
        "--embedding_type",
        nargs="+",
        default=["vision"],
        help="Embedding types for perturbance modes; accepts space/comma/plus separated values.",
    )
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--skip-frame", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=_static.DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-root-final",
        type=_static.parse_bool,
        default=False,
        help="If true, write directly into --output-root instead of appending metric/dataset components.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(_static.BASE_CHECKPOINT_URI))
    parser.add_argument("--norm-stats-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--video-backend",
        choices=["pyav", "torchcodec", "video_reader"],
        default="pyav",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--max-steps-per-trajectory",
        type=int,
        default=0,
        help="Keep 0 for one output trajectory per selected source episode.",
    )
    parser.add_argument("--data.default_prompt", dest="data_default_prompt", default=None)
    parser.add_argument("--mesa-root", type=Path, default=_mesa_dataset.DEFAULT_MESA_ROOT)
    parser.add_argument("--robocasa-base", type=Path, default=_robocasa_dataset.DEFAULT_DATASET_BASE)
    return parser.parse_args()


def _as_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(np.asarray(value).item())


def _require_episode_choice(args: argparse.Namespace) -> None:
    if args.episode_index is None and not args.use_first_episode:
        raise SystemExit("ERROR: pass --episode-index or --use-first-episode=True.")
    if args.episode_index is not None and args.use_first_episode:
        raise SystemExit("ERROR: pass only one of --episode-index or --use-first-episode=True.")


def benchmark_name(dataset_cfg: dict[str, str]) -> str:
    loader = dataset_cfg.get("loader")
    if loader in {"robocasa", "mesa"}:
        return loader
    if str(dataset_cfg["repo"]).startswith("franka") or str(dataset_cfg["path"]).find("/franka") >= 0:
        return "franka"
    return "openarm"


def _read_info(root: Path) -> dict[str, Any]:
    return _static.json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def _resolve_robocasa_episode(dataset_cfg: dict[str, str], dataset, episode_index: int | None) -> EpisodeSlice:
    for task_offset, task_dataset in enumerate(dataset.task_datasets):
        task_start = 0 if task_offset == 0 else int(dataset.cumulative_lengths[task_offset - 1])
        episode_indices = [int(ep) for ep in task_dataset.episode_indices]
        cumulative_lengths = [int(length) for length in task_dataset.cumulative_lengths]
        if not episode_indices:
            continue

        for local_offset, local_episode_index in enumerate(episode_indices):
            source_episode_index = int(task_dataset.task_offset) * 1_000_000 + local_episode_index
            if episode_index is not None and source_episode_index != episode_index:
                continue
            local_start = 0 if local_offset == 0 else cumulative_lengths[local_offset - 1]
            local_stop = cumulative_lengths[local_offset]
            return EpisodeSlice(
                source_episode_index=source_episode_index,
                local_episode_index=local_episode_index,
                start=task_start + local_start,
                stop=task_start + local_stop,
                label=f"robocasa_task={task_dataset.task_root.task}",
                benchmark="robocasa",
                dataset_name=str(dataset_cfg["repo"]),
                dataset_root=task_dataset.root,
                fps=int(task_dataset.info.get("fps", 20)),
                camera_names=(
                    "robot0_agentview_left",
                    "robot0_eye_in_hand",
                    "robot0_agentview_right",
                ),
                action_key="action",
            )

    raise SystemExit(f"ERROR: RoboCasa source episode not found: {episode_index}")


def _resolve_indexed_episode(
    *,
    dataset_cfg: dict[str, str],
    dataset_root: Path,
    benchmark: str,
    episode_indices: list[int],
    cumulative_lengths: list[int],
    episode_index: int | None,
    label: str,
    fps: int,
    camera_names: tuple[str, ...],
    action_key: str,
) -> EpisodeSlice:
    if not episode_indices:
        raise SystemExit(f"ERROR: {label} dataset has no episodes.")
    selected_episode = episode_indices[0] if episode_index is None else episode_index
    try:
        episode_offset = episode_indices.index(selected_episode)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {label} source episode not found: {selected_episode}") from exc
    start = 0 if episode_offset == 0 else cumulative_lengths[episode_offset - 1]
    stop = cumulative_lengths[episode_offset]
    return EpisodeSlice(
        source_episode_index=selected_episode,
        local_episode_index=selected_episode,
        start=start,
        stop=stop,
        label=label,
        benchmark=benchmark,
        dataset_name=str(dataset_cfg["repo"]),
        dataset_root=dataset_root,
        fps=fps,
        camera_names=camera_names,
        action_key=action_key,
    )


def _resolve_mesa_episode(dataset_cfg: dict[str, str], dataset, episode_index: int | None) -> EpisodeSlice:
    root = Path(dataset.root)
    info = _read_info(root)
    return _resolve_indexed_episode(
        dataset_cfg=dataset_cfg,
        dataset_root=root,
        benchmark="mesa",
        episode_indices=[int(ep) for ep in dataset.episode_indices],
        cumulative_lengths=[int(length) for length in dataset._cumulative_lengths],
        episode_index=episode_index,
        label=f"mesa_suite={dataset.suite_name}",
        fps=int(info.get("fps", 15)),
        camera_names=("leftshoulder_image", "robot0_eye_in_hand_image"),
        action_key="actions_joint_pos",
    )


def _unwrap_dataset(dataset):
    current = dataset
    while hasattr(current, "_dataset"):
        current = current._dataset
    return current


def _resolve_lerobot_episode(dataset_cfg: dict[str, str], dataset, episode_index: int | None) -> EpisodeSlice:
    base_dataset = _unwrap_dataset(dataset)
    dataset_root = Path(dataset_cfg["path"]).expanduser().resolve()
    info = _read_info(dataset_root)
    selected_episode = 0 if episode_index is None else episode_index
    start = None
    stop = None

    meta_episodes = getattr(getattr(base_dataset, "meta", None), "episodes", None)
    if meta_episodes is not None:
        num_episodes = len(meta_episodes)
        if selected_episode < 0 or selected_episode >= num_episodes:
            raise SystemExit(
                f"ERROR: LeRobot episode index {selected_episode} outside available range [0, {num_episodes})."
            )
        row = meta_episodes[selected_episode]
        start = _as_int(row["dataset_from_index"])
        stop = _as_int(row["dataset_to_index"])
    else:
        episode_data_index = getattr(base_dataset, "episode_data_index", None)
        if not isinstance(episode_data_index, dict):
            raise SystemExit(
                "ERROR: generic single-episode resolution requires dataset.meta.episodes or dataset.episode_data_index."
            )
        starts = episode_data_index.get("from")
        stops = episode_data_index.get("to")
        if starts is None or stops is None:
            raise SystemExit("ERROR: dataset.episode_data_index must contain 'from' and 'to'.")
        num_episodes = len(starts)
        if selected_episode < 0 or selected_episode >= num_episodes:
            raise SystemExit(
                f"ERROR: LeRobot episode index {selected_episode} outside available range [0, {num_episodes})."
            )
        start = _as_int(starts[selected_episode])
        stop = _as_int(stops[selected_episode])

    selected_episode = 0 if episode_index is None else episode_index
    return EpisodeSlice(
        source_episode_index=selected_episode,
        local_episode_index=selected_episode,
        start=start,
        stop=stop,
        label="lerobot_episode",
        benchmark=benchmark_name(dataset_cfg),
        dataset_name=str(dataset_cfg["repo"]),
        dataset_root=dataset_root,
        fps=int(info.get("fps", 20)),
        camera_names=tuple(
            key
            for key, spec in info.get("features", {}).items()
            if isinstance(spec, dict) and spec.get("dtype") == "video"
        ),
        action_key="action",
    )


def resolve_episode_slice(dataset_cfg: dict[str, str], dataset, episode_index: int | None) -> EpisodeSlice:
    if dataset_cfg.get("loader") == "robocasa":
        return _resolve_robocasa_episode(dataset_cfg, dataset, episode_index)
    if dataset_cfg.get("loader") == "mesa":
        return _resolve_mesa_episode(dataset_cfg, dataset, episode_index)
    return _resolve_lerobot_episode(dataset_cfg, dataset, episode_index)


def episode_data_path(episode: EpisodeSlice) -> Path:
    info = _read_info(episode.dataset_root)
    chunks_size = int(info.get("chunks_size", 1000))
    if episode.benchmark == "franka":
        data_path = info["data_path"].format(
            chunk_index=episode.local_episode_index // chunks_size,
            file_index=episode.local_episode_index // chunks_size,
        )
        return episode.dataset_root / data_path
    data_path = info["data_path"].format(
        episode_chunk=episode.local_episode_index // chunks_size,
        episode_index=episode.local_episode_index,
    )
    return episode.dataset_root / data_path


def episode_video_path(episode: EpisodeSlice, camera_name: str) -> Path:
    info = _read_info(episode.dataset_root)
    chunks_size = int(info.get("chunks_size", 1000))
    pattern = info["video_path"]
    if episode.benchmark == "franka":
        return episode.dataset_root / pattern.format(
            video_key=camera_name,
            chunk_index=episode.local_episode_index // chunks_size,
            file_index=episode.local_episode_index // chunks_size,
        )
    camera_key = camera_name
    if episode.benchmark == "robocasa" and not camera_key.startswith(_robocasa_dataset.VIDEO_KEY_PREFIX):
        camera_key = f"{_robocasa_dataset.VIDEO_KEY_PREFIX}{camera_key}"
    return episode.dataset_root / pattern.format(
        episode_chunk=episode.local_episode_index // chunks_size,
        video_key=camera_key,
        episode_index=episode.local_episode_index,
    )


def load_episode_actions(episode: EpisodeSlice) -> np.ndarray:
    table = pq.read_table(episode_data_path(episode))
    rows = table.to_pylist()
    if episode.benchmark == "franka":
        rows = [row for row in rows if int(row["episode_index"]) == episode.local_episode_index]
    actions = np.asarray([row[episode.action_key] for row in rows], dtype=np.float32)
    expected = episode.num_frames
    if actions.shape[0] < expected:
        raise SystemExit(
            f"ERROR: {episode.action_key} has {actions.shape[0]} rows but episode expects {expected} frames."
        )
    return actions[:expected]


def load_dataset_and_transform(args: argparse.Namespace):
    dataset_cfg = _static.DATASETS[args.dataset]
    train_config = _config.get_config(dataset_cfg["config"])
    if dataset_cfg.get("loader") == "robocasa":
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, dataset_soup_keys=None, data_dirs=None),
        )
    if args.data_default_prompt is not None:
        if not hasattr(train_config.data, "default_prompt"):
            raise ValueError(f"Config {train_config.name} does not support default_prompt override")
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, default_prompt=args.data_default_prompt),
        )

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    assets_config = getattr(train_config.data, "assets", None)
    config_assets_dir = getattr(assets_config, "assets_dir", None) or train_config.assets_dirs
    config_norm_stats_path = Path(config_assets_dir) / data_config.asset_id if data_config.asset_id else None
    final_norm_stats_source = config_norm_stats_path
    if args.norm_stats_dir is not None:
        override_norm_stats_dir = args.norm_stats_dir.expanduser()
        override_norm_stats = _normalize.load(override_norm_stats_dir)
        if dataset_cfg.get("loader") == "robocasa":
            override_norm_stats = _static.pad_robocasa_override_norm_stats_if_needed(
                override_norm_stats,
                target_dim=train_config.model.action_dim,
                source=override_norm_stats_dir,
            )
        data_config = dataclasses.replace(data_config, norm_stats=override_norm_stats)
        final_norm_stats_source = override_norm_stats_dir
    print(f"Final norm stats source: {final_norm_stats_source}")
    if data_config.norm_stats is None:
        raise SystemExit(f"ERROR: norm stats path not found at {final_norm_stats_source}")

    if dataset_cfg.get("loader") == "mesa":
        dataset = _mesa_dataset.load_mesa_suite_dataset(
            dataset_root=Path(dataset_cfg["path"]),
            suite_name=dataset_cfg["suite"],
            action_horizon=train_config.model.action_horizon,
            mesa_root=args.mesa_root,
        )
        print(
            "Loaded MESA suite "
            f"{dataset_cfg['suite']} with {len(dataset.episode_indices)} episodes and {len(dataset)} frames."
        )
    elif dataset_cfg.get("loader") == "robocasa":
        robocasa_base = args.robocasa_base if args.robocasa_base is not None else Path(dataset_cfg["path"])
        dataset = _robocasa_dataset.load_robocasa_split_dataset(
            dataset_base=robocasa_base,
            split=dataset_cfg["split"],
            action_horizon=train_config.model.action_horizon,
        )
        print(
            "Loaded RoboCasa split "
            f"{dataset_cfg['split']} with {len(dataset.task_datasets)} tasks, "
            f"{len(dataset.episode_indices)} episodes, and {len(dataset)} frames."
        )
    else:
        from openpi.training import data_loader as _data_loader

        dataset = _data_loader.create_torch_dataset(
            data_config,
            train_config.model.action_horizon,
            train_config.model,
            repo_root_override=dataset_cfg["path"],
        )
        if hasattr(dataset, "video_backend"):
            dataset.video_backend = args.video_backend
            print(f"Using dataset video backend: {dataset.video_backend}")

    return dataset_cfg, train_config, dataset, _static.build_transform(data_config)


def dispatch_metric(args: argparse.Namespace, model, dataset, transform, output_dir: Path) -> None:
    if args.metric == "gradient":
        _static.run_gradient_mode(args, model, dataset, transform, output_dir)
        return
    if args.metric == "cosine":
        _static.run_cosine_mode(args, model, dataset, transform, output_dir)
        return
    if args.metric == "perturbance":
        _static.run_perturbance_mode(args, model, dataset, transform, output_dir)
        return
    if args.metric == "perturbance-noise":
        _static.run_perturbance_noise_mode(args, model, dataset, transform, output_dir)
        return
    raise ValueError(f"Unsupported metric mode: {args.metric}")


def main() -> None:
    args = parse_args()
    _require_episode_choice(args)
    args.embedding_type = _static.parse_embedding_types(args.embedding_type)
    if args.metric == "perturbance-noise":
        invalid = sorted(set(args.embedding_type) - _static.PERTURBANCE_NOISE_EMBEDDING_TYPES)
        if invalid:
            raise ValueError(
                "metric=perturbance-noise only supports embedding_type in "
                f"{sorted(_static.PERTURBANCE_NOISE_EMBEDDING_TYPES)}; got invalid={invalid}"
            )

    output_root = args.output_root.expanduser()
    if args.output_root_final:
        output_dir = output_root
    elif args.metric == "perturbance-noise" and output_root.name != "perturbance-all":
        output_root = output_root.parent / "perturbance-all"
        output_dir = output_root / args.dataset
    else:
        output_dir = output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg, train_config, dataset, transform = load_dataset_and_transform(args)
    episode_slice = resolve_episode_slice(dataset_cfg, dataset, args.episode_index).limited(args.max_frames)
    if episode_slice.num_frames <= 0:
        raise SystemExit(f"ERROR: selected episode has no frames: {episode_slice}")
    print(
        "Selected single episode "
        f"source_episode_index={episode_slice.source_episode_index} "
        f"frames=[{episode_slice.start}, {episode_slice.stop}) "
        f"num_frames={episode_slice.num_frames} {episode_slice.label}"
    )
    dataset = EpisodeDatasetView(dataset, episode_slice)

    checkpoint_path = Path(download.maybe_download(str(args.checkpoint_dir)))
    print(f"Final checkpoint path: {checkpoint_path}")
    model = _static.load_model(train_config, checkpoint_path, args.device)
    dispatch_metric(args, model, dataset, transform, output_dir)


if __name__ == "__main__":
    main()
