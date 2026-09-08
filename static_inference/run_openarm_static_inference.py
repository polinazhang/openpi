#!/usr/bin/env python3
"""Run OpenPI static inference on one selected right-side OpenArm episode."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation")
sys.path[:0] = [str(REPO_ROOT / "static-inference"), str(REPO_ROOT)]

from openarm.contracts import metric_mask, right_action, right_state, six_arm_gripper_overflow
from openarm.data import OpenArmTrajectory, manifest_entry
from run_droid_static_inference import build_transform, load_model, observation, tensor
from openpi.shared import normalize as _normalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi05-base", "pi05-robocasa"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-sequence", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-meta", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    entry = manifest_entry(args.manifest, args.episode_sequence)
    model, config = load_model(args.checkpoint_dir, args.device)
    transform = build_transform(args.model, config, _normalize.load(args.norm_stats_dir))
    mask = metric_mask(config.action_dim, 12) if args.model == "pi05-robocasa" else metric_mask(config.action_dim)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosines, targets, velocities = [], [], []
    losses, gradnorms = defaultdict(list), defaultdict(list)

    trajectory = OpenArmTrajectory(entry["dataset_dir"], entry["episode_index"])
    horizon = config.action_horizon
    frame_count = max(0, len(trajectory) - horizon + 1)
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    for frame in range(frame_count):
        state = right_state(trajectory.state[frame])
        raw_actions = trajectory.actions[frame : frame + horizon]
        actions = right_action(raw_actions)
        if args.model == "pi05-robocasa":
            mapped_state = np.zeros(16, dtype=np.float32)
            mapped_state[:7] = state[:7]
            mapped_state[14] = state[7]
            actions = six_arm_gripper_overflow(raw_actions, config.action_dim, 12)
        else:
            mapped_state = state
        sample = {
            "observation/exterior_image_1_left": trajectory.image("head_image", frame),
            "observation/exterior_image_2_left": trajectory.image("left_wrist_image", frame),
            "observation/wrist_image_left": trajectory.image("right_wrist_image", frame),
            "observation/joint_position": state[:7],
            "observation/gripper_position": state[7:],
            "observation/state": mapped_state,
            "actions": actions,
            "prompt": trajectory.prompt,
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
        meta = args.output_dir / "meta"; meta.mkdir(exist_ok=True)
        np.save(meta / "u.npy", np.stack(targets).astype(np.float16))
        np.save(meta / "v.npy", np.stack(velocities).astype(np.float16))
    (args.output_dir / "trajectory_meta.json").write_text(json.dumps({
        "model": args.model, "dataset_name": entry["dataset_name"], "episode_index": entry["episode_index"],
        "episode_sequence": args.episode_sequence, "source_length": len(trajectory), "action_horizon": horizon,
        "num_frames_used": frame_count, "discarded_tail": min(horizon - 1, len(trajectory)),
        "side": "right", "metric_dims": np.flatnonzero(mask).tolist(),
        "overflow_identity_slot": 12 if args.model == "pi05-robocasa" else None,
        "camera_mapping": {"base": "head_image", "primary_wrist": "right_wrist_image", "secondary": "left_wrist_image"},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
