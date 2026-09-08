#!/usr/bin/env python3
"""Run OpenPI static inference on one selected DROID trajectory."""

from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation")
STATIC_ROOT = REPO_ROOT / "static-inference"
sys.path[:0] = [str(STATIC_ROOT), str(REPO_ROOT)]

from droid.archive import DroidTrajectory, manifest_entry
from droid.contracts import (
    droid_action,
    droid_cartesian_action,
    droid_state,
    pi05_robocasa_state,
    robocasa_model_action,
    robocasa_model_action_metric_mask,
)
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies.droid_policy import DroidInputs
from openpi.shared import normalize as _normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi05-base", "pi05-robocasa"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-meta", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaCheckpointInputs:
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "state": np.asarray(data["observation/state"]),
            "actions": np.asarray(data["actions"]),
            "image": {
                "base_0_rgb": parse_image(data["observation/exterior_image_1_left"]),
                "left_wrist_0_rgb": parse_image(data["observation/wrist_image_left"]),
                "right_wrist_0_rgb": parse_image(data["observation/exterior_image_2_left"]),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
            "prompt": data["prompt"],
        }


def build_transform(model_name: str, config: pi0_config.Pi0Config, norm_stats):
    inputs = DroidInputs(model_type=_model.ModelType.PI05) if model_name == "pi05-base" else RobocasaCheckpointInputs()
    return _transforms.compose(
        [
            inputs,
            _transforms.Normalize(norm_stats),
            _transforms.ResizeImages(224, 224),
            _transforms.TokenizePrompt(
                _tokenizer.PaligemmaTokenizer(config.max_token_len),
                discrete_state_input=config.discrete_state_input,
            ),
            _transforms.PadStatesAndActions(config.action_dim),
        ]
    )


def tensor(value, device, dtype=None):
    result = torch.as_tensor(value, device=device)
    return result.to(dtype=dtype) if dtype is not None else result


def observation(sample: dict[str, Any], device: str) -> _model.Observation:
    images = {
        key: tensor(value, device, torch.uint8 if value.dtype == np.uint8 else torch.float32).unsqueeze(0)
        for key, value in sample["image"].items()
    }
    return _model.Observation.from_dict({
        "image": images,
        "image_mask": {key: tensor(np.asarray(value), device, torch.bool).unsqueeze(0) for key, value in sample["image_mask"].items()},
        "state": tensor(sample["state"], device, torch.float32).unsqueeze(0),
        "tokenized_prompt": tensor(sample["tokenized_prompt"], device, torch.long).unsqueeze(0),
        "tokenized_prompt_mask": tensor(sample["tokenized_prompt_mask"], device, torch.bool).unsqueeze(0),
    })


def load_model(checkpoint_dir: Path, device: str):
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    config = pi0_config.Pi0Config(pi05=True, action_horizon=15, pytorch_compile_mode=None)
    model = config.load_pytorch(SimpleNamespace(model=config), str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    return model.eval().to(device), config


def main() -> None:
    args = parse_args()
    entry = manifest_entry(args.manifest, args.trajectory_index)
    model, config = load_model(args.checkpoint_dir, args.device)
    norm_stats = _normalize.load(args.norm_stats_dir)
    transform = build_transform(args.model, config, norm_stats)
    metric_mask = np.zeros(config.action_dim, dtype=np.float32)
    metric_mask[:8] = 1.0
    if args.model == "pi05-robocasa":
        metric_mask = robocasa_model_action_metric_mask(config.action_dim)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosines = []
    losses: dict[int, list[np.ndarray]] = defaultdict(list)
    gradnorms: dict[int, list[np.ndarray]] = defaultdict(list)
    targets = []
    velocities = []

    with DroidTrajectory(entry["archive_path"]) as trajectory:
        horizon = config.action_horizon
        frame_count = max(0, len(trajectory) - horizon + 1)
        if args.max_frames is not None:
            frame_count = min(frame_count, args.max_frames)
        for frame in range(frame_count):
            state = droid_state(trajectory.arrays["observation_joint_position"][frame], trajectory.arrays["observation_gripper_position"][frame])
            if args.model == "pi05-base":
                actions = droid_action(
                    trajectory.arrays["action_joint_velocity"][frame : frame + horizon],
                    trajectory.arrays["action_gripper_position"][frame : frame + horizon],
                )
            else:
                actions = droid_cartesian_action(
                    trajectory.arrays["action_cartesian_velocity"][frame : frame + horizon],
                    trajectory.arrays["action_gripper_position"][frame : frame + horizon],
                )
            sample = {
                "observation/exterior_image_1_left": trajectory.image("exterior_image_1_left", frame),
                "observation/exterior_image_2_left": trajectory.image("exterior_image_2_left", frame),
                "observation/wrist_image_left": trajectory.image("wrist_image_left", frame),
                "observation/joint_position": state[:7],
                "observation/gripper_position": state[7:],
                "observation/state": pi05_robocasa_state(state) if args.model == "pi05-robocasa" else state,
                "actions": robocasa_model_action(actions) if args.model == "pi05-robocasa" else actions,
                "prompt": trajectory.prompt,
            }
            if args.model == "pi05-robocasa":
                padded = np.zeros((horizon, config.action_dim), dtype=np.float32)
                padded[:, :12] = sample["actions"]
                sample["actions"] = padded
            transformed = transform(sample)
            result = model.compute_static_inference_metrics(
                observation(transformed, args.device),
                tensor(transformed["actions"], args.device, torch.float32).unsqueeze(0),
                num_steps=args.num_steps,
                metric_dim_mask=metric_mask,
            )
            cosines.append(np.stack([value.squeeze(0).cpu().numpy() for value in result["cosine_steps"]]))
            for step, value in enumerate(result["loss_steps"]):
                losses[step].append(value.squeeze(0).float().cpu().numpy())
            for step, value in enumerate(result["vision_gradnorm_steps"]):
                gradnorms[step].append(value.squeeze(0).float().cpu().numpy())
            if args.save_meta:
                targets.append(result["target"].squeeze(0).cpu().numpy())
                velocities.append(np.stack([value.squeeze(0).cpu().numpy() for value in result["velocity_steps"]]))

    np.save(args.output_dir / "cosine.npy", np.stack(cosines).astype(np.float16))
    for step, values in sorted(losses.items()):
        np.save(args.output_dir / f"final_loss_{step}.npy", np.stack(values).astype(np.float32))
    for step, values in sorted(gradnorms.items()):
        np.save(args.output_dir / f"gradnorm_vision_step_{step}.npy", np.stack(values).astype(np.float16))
    if args.save_meta:
        meta = args.output_dir / "meta"
        meta.mkdir(exist_ok=True)
        np.save(meta / "u.npy", np.stack(targets).astype(np.float16))
        np.save(meta / "v.npy", np.stack(velocities).astype(np.float16))
    (args.output_dir / "trajectory_meta.json").write_text(json.dumps({
        "model": args.model,
        "trajectory_index": args.trajectory_index,
        "trajectory_id": entry["trajectory_id"],
        "source_length": entry["length"],
        "action_horizon": config.action_horizon,
        "num_frames_used": frame_count,
        "discarded_tail": min(config.action_horizon - 1, entry["length"]),
        "metric_dims": np.flatnonzero(metric_mask).tolist(),
        "action_source": ("joint_velocity + gripper_position" if args.model == "pi05-base" else "direct checkpoint-mapped cartesian_velocity + gripper_position"),
        "language_field": "language_instruction",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
