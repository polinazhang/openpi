#!/usr/bin/env python3
"""Compute static EDR/cosine metrics using ground-truth dataset actions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from tqdm import tqdm

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.shared import download

DATASETS: dict[str, dict[str, str]] = {
    "pick_cup": {
        "repo": "qrafty-ai/tea_pick_cup",
        "path": "/work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_pick_cup",
        "config": "pi05_tea_pick_cup",
    },
    "pour_ice": {
        "repo": "qrafty-ai/tea_pour_ice",
        "path": "/work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_pour_ice",
        "config": "pi05_tea_pour_ice",
    },
    "use_spoon": {
        "repo": "qrafty-ai/tea_use_spoon_openpi",
        "path": "/work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_use_spoon_openpi",
        "config": "pi05_tea_use_spoon",
    },
    "use_steel_spoon": {
        "repo": "qrafty-ai/tea_use_steel_spoon",
        "path": "/work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_use_steel_spoon",
        "config": "pi05_tea_use_steel_spoon",
    },
}

DEFAULT_OUTPUT_ROOT = Path("/work/nvme/bfbo/xzhang42/static")
BASE_CHECKPOINT_URI = "/work/nvme/bfbo/xzhang42/openpi/checkpoints/base"
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run static EDR evaluation on LeRobot datasets.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        default="pick_cup",
        help="Dataset to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-frame",
        type=int,
        default=50,
        help="Only evaluate frames where global_index %% skip_frame == 0 (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where metadata and artifacts will be written.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on (default: auto-detect).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on processed frames (for debugging).",
    )
    parser.add_argument(
        "--max-steps-per-trajectory",
        type=int,
        default=2048,
        help=(
            "Maximum processed steps to pack into a single trajectory artifact. "
            "Use 0 to disable chunking (falls back to per-episode flush)."
        ),
    )
    return parser.parse_args()


def build_transform(data_config: _config.DataConfig) -> _transforms.DataTransformFn:
    steps: list[_transforms.DataTransformFn] = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    return _transforms.compose(steps)


def load_model(train_config: _config.TrainConfig, checkpoint_dir: Path, device: str):
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing PyTorch weights at {weight_path}")
    model = train_config.model.load_pytorch(train_config, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()
    return model.to(device)


def _to_torch_tensor(array, device, dtype=None):
    tensor = torch.as_tensor(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device)


def build_observation(sample: dict, device: str) -> _model.Observation:
    batch_image = {}
    for name, image in sample["image"].items():
        dtype = torch.uint8 if getattr(image, "dtype", None) == np.uint8 else torch.float32
        tensor = _to_torch_tensor(image, device, dtype=dtype).unsqueeze(0)
        batch_image[name] = tensor

    batch_mask = {}
    for name, mask in sample["image_mask"].items():
        tensor = _to_torch_tensor(np.asarray(mask), device, dtype=torch.bool).unsqueeze(0)
        batch_mask[name] = tensor

    obs_dict = {
        "image": batch_image,
        "image_mask": batch_mask,
        "state": _to_torch_tensor(sample["state"], device, dtype=torch.float32).unsqueeze(0),
    }

    if "tokenized_prompt" in sample:
        obs_dict["tokenized_prompt"] = (
            _to_torch_tensor(sample["tokenized_prompt"], device, dtype=torch.long).unsqueeze(0)
        )
        obs_dict["tokenized_prompt_mask"] = (
            _to_torch_tensor(sample["tokenized_prompt_mask"], device, dtype=torch.bool).unsqueeze(0)
        )

    return _model.Observation.from_dict(obs_dict)


def compute_metrics(vt: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vt = vt.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    dot = torch.sum(vt * target, dim=-1)
    vt_norm = torch.linalg.norm(vt, dim=-1)
    target_norm = torch.linalg.norm(target, dim=-1)
    cosine = dot / (vt_norm * target_norm + EPS)

    scale = dot / (vt_norm.square() + EPS)
    scaled = scale.unsqueeze(-1) * vt
    scaled_residual = scaled - target
    scaled_norm = torch.linalg.norm(scaled_residual, dim=-1)
    return scaled_norm, cosine


@dataclass
class EpisodeBuffer:
    episode_index: int
    start_offset: int
    edr: dict[int, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    cosine: dict[int, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    final_loss: list[np.ndarray] = field(default_factory=list)

    def add_step(self, layer_metrics: dict[int, tuple[np.ndarray, np.ndarray]], final_loss: np.ndarray) -> None:
        for layer_idx, (edr_values, cosine_values) in layer_metrics.items():
            self.edr[layer_idx].append(edr_values)
            self.cosine[layer_idx].append(cosine_values)
        self.final_loss.append(final_loss)

    def has_data(self) -> bool:
        return bool(self.final_loss)

    def num_steps(self) -> int:
        return len(self.final_loss)


def finalize_episode(
    buffer: EpisodeBuffer,
    trajectory_id: int,
    layer_indices: Iterable[int],
    out_root: Path,
    offsets: Dict[str, int],
    metadata_entries: list,
) -> int:
    rel_prefix = f"{trajectory_id:06d}/npy-metadata"
    traj_dir = out_root / f"{trajectory_id:06d}" / "npy-metadata"
    traj_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    artifact_shapes: dict[str, list[int]] = {}
    artifact_lengths: dict[str, int] = {}
    artifact_spans: dict[str, dict[str, int]] = {}

    def _record_artifact(name: str, array: np.ndarray):
        nonlocal offsets, artifacts, artifact_shapes, artifact_lengths, artifact_spans
        out_path = traj_dir / f"{name}.npy"
        np.save(out_path, array.astype(np.float16))
        artifacts[name] = f"{rel_prefix}/{name}.npy"
        artifact_shapes[name] = list(array.shape)
        length = int(array.size)
        artifact_lengths[name] = length
        start = offsets.get(name, 0)
        artifact_spans[name] = {"offset": start, "length": length}
        offsets[name] = start + length

    for layer_idx in layer_indices:
        if layer_idx not in buffer.edr:
            continue
        edr_array = np.stack(buffer.edr[layer_idx], axis=0)
        cosine_array = np.stack(buffer.cosine[layer_idx], axis=0)
        _record_artifact(f"static_edr_layer_{layer_idx:02d}", edr_array)
        _record_artifact(f"static_cosine_layer_{layer_idx:02d}", cosine_array)

    final_loss_array = np.stack(buffer.final_loss, axis=0)
    _record_artifact("static_final_loss", final_loss_array)

    metadata_entries.append(
        {
            "trajectory_id": trajectory_id,
            "source_episode_index": buffer.episode_index,
            "episode_step_offset": buffer.start_offset,
            "trajectory_rel_dir": rel_prefix,
            "num_steps": buffer.num_steps(),
            "artifacts": artifacts,
            "artifact_shapes": artifact_shapes,
            "artifact_lengths": artifact_lengths,
            "artifact_spans": artifact_spans,
        }
    )
    return trajectory_id + 1


def main() -> None:
    args = parse_args()
    dataset_cfg = DATASETS[args.dataset]
    output_dir = args.output_root.expanduser() / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = _config.get_config(dataset_cfg["config"])
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    checkpoint_path = Path(download.maybe_download(BASE_CHECKPOINT_URI))
    model = load_model(train_config, checkpoint_path, args.device)

    dataset = _data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
        repo_root_override=dataset_cfg["path"],
    )
    transform = build_transform(data_config)

    metadata_entries: list = []
    offsets: dict[str, int] = {}
    trajectory_counter = 0
    current_buffer: EpisodeBuffer | None = None
    current_episode = None
    processed = 0
    layer_indices: Iterable[int] | None = None
    current_episode_step_offset = 0
    step_chunk_limit = args.max_steps_per_trajectory if args.max_steps_per_trajectory > 0 else None

    total_frames = len(dataset)
    limit = total_frames if args.max_frames is None else min(args.max_frames, total_frames)
    iterator = range(limit)
    eval_total = (limit + max(args.skip_frame, 1) - 1) // max(args.skip_frame, 1)
    eval_pbar = tqdm(total=eval_total, desc="Evaluated frames")

    for global_idx in tqdm(iterator, desc="Processing frames"):
        sample = dataset[global_idx]
        episode_idx = int(sample["episode_index"].item())

        if current_episode is None or episode_idx != current_episode:
            if current_buffer and current_buffer.has_data():
                trajectory_counter = finalize_episode(
                    current_buffer,
                    trajectory_counter,
                    layer_indices or [],
                    output_dir,
                    offsets,
                    metadata_entries,
                )
            current_episode_step_offset = 0
            current_episode = episode_idx
            current_buffer = EpisodeBuffer(episode_index=episode_idx, start_offset=current_episode_step_offset)

        if args.skip_frame > 1 and global_idx % args.skip_frame != 0:
            continue

        transformed = transform(sample)
        observation = build_observation(transformed, args.device)
        actions = _to_torch_tensor(transformed["actions"], args.device, dtype=torch.float32).unsqueeze(0)

        result = model.compute_static_targets(observation, actions)
        vt_layers = result["vt_layers"]
        target = result["target"]
        final_prediction = result["final_prediction"]

        if not vt_layers:
            continue

        if layer_indices is None:
            layer_indices = sorted(vt_layers.keys())

        metrics_per_layer: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for layer_idx, vt in vt_layers.items():
            scaled_norm, cosine = compute_metrics(vt, target)
            metrics_per_layer[layer_idx] = (
                scaled_norm.squeeze(0).cpu().numpy(),
                cosine.squeeze(0).cpu().numpy(),
            )

        final_loss = torch.linalg.norm(final_prediction - target, dim=-1)
        final_loss_np = final_loss.squeeze(0).cpu().numpy()

        if current_buffer is not None:
            current_buffer.add_step(metrics_per_layer, final_loss_np)
            if step_chunk_limit and current_buffer.num_steps() >= step_chunk_limit:
                next_offset = current_episode_step_offset + current_buffer.num_steps()
                trajectory_counter = finalize_episode(
                    current_buffer,
                    trajectory_counter,
                    layer_indices or [],
                    output_dir,
                    offsets,
                    metadata_entries,
                )
                current_episode_step_offset = next_offset
                current_buffer = EpisodeBuffer(
                    episode_index=episode_idx,
                    start_offset=current_episode_step_offset,
                )
        processed += 1
        eval_pbar.update(1)

    if current_buffer and current_buffer.has_data():
        trajectory_counter = finalize_episode(
            current_buffer,
            trajectory_counter,
            layer_indices or [],
            output_dir,
            offsets,
            metadata_entries,
        )

    eval_pbar.close()

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_entries, handle, indent=2)

    print(f"Wrote metadata for {trajectory_counter} trajectories to {metadata_path}")
    print(f"Recorded {processed} frames (skip_frame={args.skip_frame}).")


if __name__ == "__main__":
    main()
