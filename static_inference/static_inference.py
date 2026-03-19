#!/usr/bin/env python3
"""Static inference metrics runner (gradient first, cosine reserved)."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

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
    "franka_object": {
        "repo": "franka_object",
        "path": "/coc/testnvme/xzhang3205/lerobot/franka_object",
        "config": "pi05_franka_object",
    },
    "franka_object_plus": {
        "repo": "franka_object",
        "path": "/coc/testnvme/xzhang3205/lerobot/franka_object_plus",
        "config": "pi05_franka_object",
    },
    "franka_object_plus_2": {
        "repo": "franka_object",
        "path": "/coc/testnvme/xzhang3205/lerobot/franka_object_plus_2",
        "config": "pi05_franka_object",
    },
    "franka_object_two": {
        "repo": "franka_object_two",
        "path": "/coc/testnvme/xzhang3205/lerobot/franka_object_two",
        "config": "pi05_franka_object",
    },
    "franka_on_top": {
        "repo": "franka_on_top",
        "path": "/coc/testnvme/xzhang3205/lerobot/franka_on_top",
        "config": "pi05_franka_object",
    },
}

DEFAULT_OUTPUT_ROOT = Path("/coc/testnvme/xzhang3205/static")
BASE_CHECKPOINT_URI = "/coc/testnvme/xzhang3205/openpi/checkpoints/torch_30000"
EPS = 1e-8


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run static inference metrics on LeRobot datasets.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        default="franka_object",
        help="Dataset to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--metric",
        choices=["", "cosine", "gradient"],
        default="gradient",
        help="Metric mode. Empty maps to gradient for backward-compatibility.",
    )
    parser.add_argument(
        "--condition",
        choices=["training", "inference"],
        default="training",
        help="Condition for gradient mode. Ignored by cosine mode.",
    )
    parser.add_argument(
        "--save_meta",
        type=parse_bool,
        default=False,
        help="Cosine-only flag. Ignored for gradient mode.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of inference denoising steps for condition-inference (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-frame",
        type=int,
        default=10,
        help="Only evaluate frames where global_index %% skip_frame == 0 (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where metadata and artifacts will be written.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(BASE_CHECKPOINT_URI),
        help="Checkpoint directory containing model.safetensors.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on (default: auto-detect).",
    )
    parser.add_argument(
        "--video-backend",
        choices=["pyav", "torchcodec", "video_reader"],
        default="pyav",
        help="LeRobot video decoding backend (default: %(default)s).",
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
            "Maximum processed frames to pack into a single trajectory artifact. "
            "Use 0 to disable chunking (falls back to per-episode flush)."
        ),
    )
    parser.add_argument(
        "--data.default_prompt",
        dest="data_default_prompt",
        default=None,
        help=(
            "Optional prompt override (train.py-style flag). "
            "When set, replaces sample task/prompt strings before tokenization."
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


def has_valid_action_chunk(raw_sample: dict) -> bool:
    """Tail-discard rule for gradient mode: skip padded action chunks."""
    pad_mask = raw_sample.get("action_is_pad")
    if pad_mask is None:
        return True
    return not bool(np.asarray(pad_mask, dtype=bool).any())


@dataclass
class GradientEpisodeBuffer:
    episode_index: int
    start_offset: int
    gradients: dict[int, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    taus: dict[int, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    final_loss: list[np.ndarray] = field(default_factory=list)

    def add_frame(self, gradient_steps: list[np.ndarray], tau_steps: list[np.ndarray], final_loss: np.ndarray) -> None:
        for step_idx, grad in enumerate(gradient_steps):
            self.gradients[step_idx].append(grad)
        for step_idx, tau in enumerate(tau_steps):
            self.taus[step_idx].append(np.asarray(tau))
        self.final_loss.append(final_loss)

    def has_data(self) -> bool:
        return bool(self.final_loss)

    def num_frames(self) -> int:
        return len(self.final_loss)


def finalize_gradient_episode(
    buffer: GradientEpisodeBuffer,
    trajectory_id: int,
    out_root: Path,
    offsets: Dict[str, int],
    metadata_entries: list[dict],
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

    for step_idx in sorted(buffer.gradients.keys()):
        grad_array = np.stack(buffer.gradients[step_idx], axis=0)
        tau_array = np.stack(buffer.taus[step_idx], axis=0)
        _record_artifact(f"gradient_step_{step_idx}", grad_array)
        _record_artifact(f"tau_step_{step_idx}", tau_array)

    final_loss_array = np.stack(buffer.final_loss, axis=0)
    _record_artifact("final_layer_loss", final_loss_array)

    metadata_entries.append(
        {
            "trajectory_id": trajectory_id,
            "source_episode_index": buffer.episode_index,
            "episode_step_offset": buffer.start_offset,
            "trajectory_rel_dir": rel_prefix,
            "num_steps": buffer.num_frames(),
            "artifacts": artifacts,
            "artifact_shapes": artifact_shapes,
            "artifact_lengths": artifact_lengths,
            "artifact_spans": artifact_spans,
        }
    )
    return trajectory_id + 1


def run_gradient_mode(
    args: argparse.Namespace,
    model,
    dataset,
    transform,
    output_dir: Path,
) -> None:
    metadata_entries: list[dict] = []
    offsets: dict[str, int] = {}
    trajectory_counter = 0
    current_buffer: GradientEpisodeBuffer | None = None
    current_episode = None
    current_episode_step_offset = 0
    step_chunk_limit = args.max_steps_per_trajectory if args.max_steps_per_trajectory > 0 else None
    processed = 0

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
                trajectory_counter = finalize_gradient_episode(
                    current_buffer,
                    trajectory_counter,
                    output_dir,
                    offsets,
                    metadata_entries,
                )
            current_episode_step_offset = 0
            current_episode = episode_idx
            current_buffer = GradientEpisodeBuffer(episode_index=episode_idx, start_offset=current_episode_step_offset)

        if args.skip_frame > 1 and global_idx % args.skip_frame != 0:
            continue
        if not has_valid_action_chunk(sample):
            continue

        sample_for_transform = sample
        if args.data_default_prompt is not None:
            sample_for_transform = dict(sample)
            sample_for_transform["task"] = np.asarray(args.data_default_prompt)
            sample_for_transform["prompt"] = np.asarray(args.data_default_prompt)

        transformed = transform(sample_for_transform)
        observation = build_observation(transformed, args.device)
        actions = _to_torch_tensor(transformed["actions"], args.device, dtype=torch.float32).unsqueeze(0)

        if args.condition == "training":
            result = model.compute_static_gradient_guidance_training(observation, actions)
        else:
            result = model.compute_static_gradient_guidance_inference(observation, actions, num_steps=args.num_steps)

        gradient_steps = [step.squeeze(0).detach().cpu().numpy() for step in result["gradient_steps"]]
        tau_steps = [step.squeeze(0).detach().cpu().numpy() for step in result["tau_steps"]]
        final_loss = result["final_layer_loss"].squeeze(0).detach().cpu().numpy()

        if current_buffer is not None:
            current_buffer.add_frame(gradient_steps, tau_steps, final_loss)
            if step_chunk_limit and current_buffer.num_frames() >= step_chunk_limit:
                next_offset = current_episode_step_offset + current_buffer.num_frames()
                trajectory_counter = finalize_gradient_episode(
                    current_buffer,
                    trajectory_counter,
                    output_dir,
                    offsets,
                    metadata_entries,
                )
                current_episode_step_offset = next_offset
                current_buffer = GradientEpisodeBuffer(
                    episode_index=episode_idx,
                    start_offset=current_episode_step_offset,
                )
        processed += 1
        eval_pbar.update(1)

    if current_buffer and current_buffer.has_data():
        trajectory_counter = finalize_gradient_episode(
            current_buffer,
            trajectory_counter,
            output_dir,
            offsets,
            metadata_entries,
        )

    eval_pbar.close()
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_entries, handle, indent=2)

    print(f"Wrote metadata for {trajectory_counter} trajectories to {metadata_path}")
    print(
        f"Recorded {processed} frames (skip_frame={args.skip_frame}, metric=gradient, condition={args.condition})."
    )


def main() -> None:
    args = parse_args()
    if args.metric == "":
        args.metric = "gradient"
    if args.metric != "cosine" and args.save_meta:
        print("Ignoring --save_meta because it only applies to metric=cosine.")

    dataset_cfg = DATASETS[args.dataset]
    output_dir = args.output_root.expanduser() / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = _config.get_config(dataset_cfg["config"])
    if args.data_default_prompt is not None:
        if not hasattr(train_config.data, "default_prompt"):
            raise ValueError(f"Config {train_config.name} does not support default_prompt override")
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, default_prompt=args.data_default_prompt),
        )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    checkpoint_path = Path(download.maybe_download(str(args.checkpoint_dir)))
    model = load_model(train_config, checkpoint_path, args.device)

    dataset = _data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
        repo_root_override=dataset_cfg["path"],
    )
    if hasattr(dataset, "video_backend"):
        dataset.video_backend = args.video_backend
        print(f"Using dataset video backend: {dataset.video_backend}")
    transform = build_transform(data_config)

    if args.metric == "gradient":
        run_gradient_mode(args, model, dataset, transform, output_dir)
        return

    if args.metric == "cosine":
        # EDR calculation retained for future re-enable:
        # scale = dot(v_t, u_t) / (||v_t||^2 + eps)
        # edr = ||scale * v_t - u_t||_2
        raise NotImplementedError("Cosine mode will be implemented in a later milestone.")

    raise ValueError(f"Unsupported metric mode: {args.metric}")


if __name__ == "__main__":
    main()
