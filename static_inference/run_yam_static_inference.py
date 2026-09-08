#!/usr/bin/env python3
"""Run OpenPI static inference on one Bimanual YAM episode."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation")
STATIC_ROOT = REPO_ROOT / "static-inference"
YAM_ROOT = STATIC_ROOT / "molmoact-yam"
sys.path[:0] = [str(YAM_ROOT), str(STATIC_ROOT), str(REPO_ROOT)]

from contracts import metric_mask, right_arm_gripper, seven_arm_model_mapping
from data import PartialYAMDataset
from run_droid_static_inference import build_transform, observation, tensor
from openpi.models import pi0_config
from openpi.shared import normalize as _normalize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi05-base", "pi05-robocasa"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-meta", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_checkpoint_model(checkpoint_dir: Path, device: str):
    saved = json.loads((checkpoint_dir / "config.json").read_text())
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=int(saved["action_dim"]),
        action_horizon=int(saved["action_horizon"]),
        paligemma_variant=saved["paligemma_variant"],
        action_expert_variant=saved["action_expert_variant"],
        pytorch_compile_mode=None,
    )
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    model = config.load_pytorch(SimpleNamespace(model=config), str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    return model.eval().to(device), config


def main():
    args = parse_args()
    dataset = PartialYAMDataset(args.dataset_root)
    trajectory = dataset[args.trajectory_index]
    model, config = load_checkpoint_model(args.checkpoint_dir, args.device)
    norm_stats = _normalize.load(args.norm_stats_dir)
    transform = build_transform(args.model, config, norm_stats)

    if args.model == "pi05-base":
        dimensions = (0, 1, 2, 3, 4, 5, 7)
        state_neutral = float(norm_stats["state"].mean[6])
        action_neutral = float(norm_stats["actions"].mean[6])
    else:
        dimensions = (0, 1, 2, 3, 4, 5, 6)
    mask = metric_mask(config.action_dim, dimensions)

    horizon = config.action_horizon
    frame_count = max(0, len(trajectory) - horizon + 1)
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosines, targets, velocities = [], [], []
    losses, gradnorms = defaultdict(list), defaultdict(list)

    for frame in range(frame_count):
        selected_state = right_arm_gripper(trajectory.state[frame])
        selected_actions = right_arm_gripper(trajectory.actions[frame : frame + horizon])
        if args.model == "pi05-base":
            state = seven_arm_model_mapping(trajectory.state[frame], state_neutral)
            actions = seven_arm_model_mapping(
                trajectory.actions[frame : frame + horizon], action_neutral
            )
            mapped_state = state
        else:
            mapped_state = np.asarray(norm_stats["state"].mean[:16], dtype=np.float32).copy()
            mapped_state[:6] = selected_state[:6]
            mapped_state[14] = selected_state[6]
            actions = np.broadcast_to(
                np.asarray(norm_stats["actions"].mean, dtype=np.float32),
                (horizon, len(norm_stats["actions"].mean)),
            ).copy()
            actions[:, :6] = selected_actions[:, :6]
            actions[:, 6] = selected_actions[:, 6]
            state = selected_state
        sample = {
            "observation/exterior_image_1_left": trajectory.image("observation.images.top", frame),
            "observation/exterior_image_2_left": trajectory.image("observation.images.left", frame),
            "observation/wrist_image_left": trajectory.image("observation.images.right", frame),
            "observation/joint_position": state[:7],
            "observation/gripper_position": state[7:],
            "observation/state": mapped_state,
            "actions": actions,
            "prompt": trajectory.instruction,
        }
        transformed = transform(sample)
        result = model.compute_static_inference_metrics(
            observation(transformed, args.device),
            tensor(transformed["actions"], args.device, torch.float32).unsqueeze(0),
            num_steps=args.num_steps,
            metric_dim_mask=mask,
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
    (args.output_dir / "trajectory_meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "episode_index": trajectory.episode_index,
                "source_length": len(trajectory),
                "action_horizon": horizon,
                "num_frames_used": frame_count,
                "side": "right",
                "metric_dims": list(dimensions),
                "unused_model_arm_slot": 6 if args.model == "pi05-base" else None,
                "instruction_source": "meta/tasks_annotated.parquet",
                "camera_mapping": {"base": "top", "primary_wrist": "right", "secondary_wrist": "left"},
            },
            indent=2,
        )
        + "\n"
    )
    dataset.clear_video_cache()


if __name__ == "__main__":
    main()
