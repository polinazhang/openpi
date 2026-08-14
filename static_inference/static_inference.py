#!/usr/bin/env python3
"""Final-output static inference for pi05 on RoboCasa demonstrations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import einops
import numpy as np
import remap
import robocasa_dataset
import torch
from tqdm import tqdm

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.shared import normalize as _normalize

DEFAULT_OUTPUT_ROOT = Path("/coc/testnvme/xzhang3205/static")


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("atomic-seen",), default="atomic-seen")
    parser.add_argument("--robocasa-base", type=Path, required=True)
    parser.add_argument("--robocasa-task", required=True)
    parser.add_argument("--robocasa-max-episodes", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--dim-remap", choices=tuple(remap.SPECS), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--skip-frame", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-steps-per-trajectory", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_meta", type=parse_bool, default=True)

    # Accepted to preserve the previous launcher interface. The replacement
    # pipeline has one unambiguous metric mode and always stores final cosine.
    parser.add_argument("--metric", default="perturbance-noise")
    parser.add_argument("--condition", default="inference")
    parser.add_argument("--save_cosine", type=parse_bool, default=True)
    parser.add_argument("--save_displacement_trace", type=parse_bool, default=False)
    parser.add_argument("--embedding_type", default="vision")
    parser.add_argument("--perturbance_step_num", type=int, default=0)
    parser.add_argument("--perturbance_step_size", type=float, default=1e-2)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--data.assets.assets-dir", dest="assets_dir", default=None)
    return parser.parse_args()


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs:
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "state": np.asarray(data["observation/state"]),
            "actions": np.asarray(data["actions"]),
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "prompt": data["prompt"],
        }


def build_transform(model_config: pi0_config.Pi0Config, norm_stats):
    return remap.compose(
        [
            RobocasaInputs(),
            _transforms.Normalize(norm_stats),
            _transforms.ResizeImages(224, 224),
            _transforms.TokenizePrompt(
                _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=model_config.discrete_state_input,
            ),
            _transforms.PadStatesAndActions(model_config.action_dim),
        ]
    )


def _tensor(value, device, dtype=None):
    result = torch.as_tensor(value, device=device)
    return result.to(dtype=dtype) if dtype is not None else result


def build_observation(sample: dict[str, Any], device: str) -> _model.Observation:
    batch_images = {}
    for key, value in sample["image"].items():
        dtype = torch.uint8 if getattr(value, "dtype", None) == np.uint8 else torch.float32
        batch_images[key] = _tensor(value, device, dtype).unsqueeze(0)

    return _model.Observation.from_dict(
        {
            "image": batch_images,
            "image_mask": {
                key: _tensor(np.asarray(value), device, torch.bool).unsqueeze(0)
                for key, value in sample["image_mask"].items()
            },
            "state": _tensor(sample["state"], device, torch.float32).unsqueeze(0),
            "tokenized_prompt": _tensor(sample["tokenized_prompt"], device, torch.long).unsqueeze(0),
            "tokenized_prompt_mask": _tensor(sample["tokenized_prompt_mask"], device, torch.bool).unsqueeze(0),
        }
    )


def load_model(checkpoint_dir: Path, device: str):
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Checkpoint lacks model.safetensors: {checkpoint_dir}")
    config = pi0_config.Pi0Config(pi05=True, pytorch_compile_mode=None)
    train_config = SimpleNamespace(model=config)
    model = config.load_pytorch(train_config, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()
    return model.to(device), config


@dataclasses.dataclass
class EpisodeBuffer:
    episode_index: int
    start_offset: int
    cosine: list[np.ndarray] = dataclasses.field(default_factory=list)
    losses: dict[int, list[np.ndarray]] = dataclasses.field(default_factory=lambda: defaultdict(list))
    gradnorms: dict[int, list[np.ndarray]] = dataclasses.field(default_factory=lambda: defaultdict(list))
    targets: list[np.ndarray] = dataclasses.field(default_factory=list)
    velocities: list[np.ndarray] = dataclasses.field(default_factory=list)

    def add(self, result: dict[str, Any], *, save_meta: bool) -> None:
        self.cosine.append(np.stack([x.squeeze(0).cpu().numpy() for x in result["cosine_steps"]], axis=0))
        for step, value in enumerate(result["loss_steps"]):
            self.losses[step].append(value.squeeze(0).to(torch.float32).cpu().numpy())
        for step, value in enumerate(result["vision_gradnorm_steps"]):
            self.gradnorms[step].append(value.squeeze(0).to(torch.float32).cpu().numpy())
        if save_meta:
            self.targets.append(result["target"].squeeze(0).cpu().numpy())
            self.velocities.append(np.stack([x.squeeze(0).cpu().numpy() for x in result["velocity_steps"]], axis=0))

    def __len__(self) -> int:
        return len(self.cosine)


def flush_episode(
    buffer: EpisodeBuffer,
    trajectory_id: int,
    output_dir: Path,
    offsets: dict[str, int],
    metadata: list[dict[str, Any]],
    *,
    save_meta: bool,
) -> int:
    rel_prefix = f"{trajectory_id:06d}/npy-metadata"
    destination = output_dir / rel_prefix
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}
    lengths: dict[str, int] = {}
    spans: dict[str, dict[str, int]] = {}

    def save(name: str, array: np.ndarray, dtype) -> None:
        path = destination / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(array).astype(dtype)
        np.save(path, array)
        artifacts[name] = f"{rel_prefix}/{name}.npy"
        shapes[name] = list(array.shape)
        lengths[name] = int(array.size)
        start = offsets.get(name, 0)
        spans[name] = {"offset": start, "length": int(array.size)}
        offsets[name] = start + int(array.size)

    # Legacy-compatible name; it now always means the actual final output.
    save("cinference-cosine_17", np.stack(buffer.cosine), np.float16)
    for step, values in sorted(buffer.losses.items()):
        save(f"final_layer_loss_vision_step_{step}", np.stack(values), np.float32)
    for step, values in sorted(buffer.gradnorms.items()):
        save(f"gradnorm_vision_step_{step}", np.stack(values), np.float16)
    if save_meta:
        save("meta/u", np.stack(buffer.targets), np.float16)
        save("meta/cinference-v_17", np.stack(buffer.velocities), np.float16)

    metadata.append(
        {
            "trajectory_id": trajectory_id,
            "source_episode_index": buffer.episode_index,
            "episode_step_offset": buffer.start_offset,
            "trajectory_rel_dir": rel_prefix,
            "num_steps": len(buffer),
            "artifacts": artifacts,
            "artifact_shapes": shapes,
            "artifact_lengths": lengths,
            "artifact_spans": spans,
        }
    )
    return trajectory_id + 1


def main() -> None:
    args = parse_args()
    if args.metric != "perturbance-noise" or args.condition != "inference":
        raise ValueError("The replacement pipeline supports only perturbance-noise/inference")
    if args.embedding_type != "vision":
        raise ValueError("The replacement pipeline computes only combined vision gradient norm")
    if args.perturbance_step_num != 0:
        raise ValueError("Perturbation optimization is not part of the replacement metric contract")
    if not args.save_cosine:
        raise ValueError("Final-output cosine is a required metric")

    spec = remap.SPECS[args.dim_remap]
    model, model_config = load_model(args.checkpoint_dir, args.device)
    norm_stats = _normalize.load(args.norm_stats_dir)
    norm_stats = remap.remap_action_norm_stats(norm_stats, spec)
    transform = build_transform(model_config, norm_stats)

    raw_dataset = robocasa_dataset.load_robocasa_split_dataset(
        dataset_base=args.robocasa_base,
        split="atomic_seen",
        action_horizon=model_config.action_horizon,
        tasks_override=[args.robocasa_task],
        max_episodes_per_task=args.robocasa_max_episodes,
    )
    dataset = remap.RemappedRobocasaDataset(raw_dataset, spec)

    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    offsets: dict[str, int] = {}
    trajectory_id = 0
    current_episode = None
    current_offset = 0
    buffer: EpisodeBuffer | None = None
    processed = 0
    limit = len(dataset) if args.max_frames is None else min(len(dataset), args.max_frames)
    chunk_limit = args.max_steps_per_trajectory if args.max_steps_per_trajectory > 0 else None

    for global_index in tqdm(range(limit), desc="Static inference"):
        raw = dataset[global_index]
        episode = int(raw["episode_index"])
        if episode != current_episode:
            if buffer is not None and len(buffer):
                trajectory_id = flush_episode(
                    buffer, trajectory_id, output_dir, offsets, metadata, save_meta=args.save_meta
                )
            current_episode = episode
            current_offset = 0
            buffer = EpisodeBuffer(episode, current_offset)

        if args.skip_frame > 1 and global_index % args.skip_frame:
            continue
        if np.asarray(raw["action_is_pad"], dtype=bool).any():
            continue

        transformed = transform(raw)
        observation = build_observation(transformed, args.device)
        actions = _tensor(transformed["actions"], args.device, torch.float32).unsqueeze(0)
        result = model.compute_static_inference_metrics(
            observation,
            actions,
            num_steps=args.num_steps,
            metric_dim_mask=spec.metric_mask,
            aligned_to_native_perm=spec.aligned_to_native_perm,
        )
        assert buffer is not None
        buffer.add(result, save_meta=args.save_meta)
        processed += 1

        if chunk_limit and len(buffer) >= chunk_limit:
            trajectory_id = flush_episode(
                buffer, trajectory_id, output_dir, offsets, metadata, save_meta=args.save_meta
            )
            current_offset += len(buffer)
            buffer = EpisodeBuffer(episode, current_offset)

    if buffer is not None and len(buffer):
        flush_episode(buffer, trajectory_id, output_dir, offsets, metadata, save_meta=args.save_meta)

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Recorded {processed} frames in {len(metadata)} trajectory artifacts at {output_dir}")


if __name__ == "__main__":
    main()
