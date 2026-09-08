#!/usr/bin/env python3
"""Run OpenPI static inference over every episode in one Franka dataset."""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

REPO_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation")
STATIC_ROOT = REPO_ROOT / "static-inference"
FRANKA_ROOT = STATIC_ROOT / "franka"
sys.path[:0] = [str(FRANKA_ROOT), str(STATIC_ROOT), str(REPO_ROOT)]
from franka.contracts import metric_mask, robocasa_state, six_arm_gripper_overflow
from franka.data import FrankaDataset
from run_droid_static_inference import build_transform, observation, tensor
from openpi.models import pi0_config
from openpi.shared import normalize as _normalize
from openpi import transforms as _transforms


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cartesian", action="store_true", help="Use stored Cartesian actions unchanged (legacy behavior).")
    mode.add_argument("--joint", action="store_true", help="Convert absolute joint targets to chunk-start-relative deltas; keep gripper absolute.")
    p.add_argument("--model", choices=("pi05-base", "pi05-robocasa"), required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--norm-stats-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--save-meta", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(checkpoint, device):
    saved = json.loads((checkpoint / "config.json").read_text())
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=int(saved["action_dim"]),
        action_horizon=int(saved["action_horizon"]),
        paligemma_variant=saved["paligemma_variant"],
        action_expert_variant=saved["action_expert_variant"],
        pytorch_compile_mode=None,
    )
    model = config.load_pytorch(SimpleNamespace(model=config), str(checkpoint / "model.safetensors"))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    return model.eval().to(device), config


def save_array(path, values, dtype, empty_shape):
    np.save(path, np.stack(values).astype(dtype) if values else np.empty(empty_shape, dtype=dtype))


def main():
    a = parse_args()
    dataset = FrankaDataset(a.dataset_dir)
    model, config = load_model(a.checkpoint_dir, a.device)
    stats = _normalize.load(a.norm_stats_dir)
    transform = build_transform(a.model, config, stats)
    dims = tuple(range(8)) if a.model == "pi05-base" else (0, 1, 2, 3, 4, 5, 6, 12)
    mask = metric_mask(config.action_dim, dims)
    episodes = list(dataset)[: a.max_episodes]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for episode in episodes:
        horizon = config.action_horizon
        count = max(0, len(episode) - horizon + 1)
        count = min(count, a.max_frames) if a.max_frames is not None else count
        losses, grads = defaultdict(list), defaultdict(list)
        cosines = []
        targets = []
        velocities = []
        for frame in range(count):
            state = episode.state[frame]
            actions = episode.actions[frame : frame + horizon]
            if a.joint:
                # DeltaActions mutates its input: never modify the cached episode slice.
                actions = _transforms.DeltaActions(_transforms.make_bool_mask(7, -1))(
                    {"state": state, "actions": actions.copy()}
                )["actions"]
            if a.model == "pi05-robocasa":
                state = robocasa_state(state, np.asarray(stats["state"].mean, dtype=np.float32))
                actions = six_arm_gripper_overflow(actions, config.action_dim, 12)
                # Preserve the checkpoint-neutral values of unrelated modalities.
                actions[:, 7:12] = np.asarray(stats["actions"].mean[7:12], dtype=np.float32)
            sample = {
                "observation/exterior_image_1_left": episode.image("observation.images.camera_front", frame),
                "observation/exterior_image_2_left": episode.image("observation.images.camera_side", frame),
                "observation/wrist_image_left": episode.image("observation.images.camera_wrist", frame),
                "observation/joint_position": episode.state[frame, :7],
                "observation/gripper_position": episode.state[frame, 7:],
                "observation/state": state,
                "actions": actions,
                "prompt": episode.instruction,
            }
            transformed = transform(sample)
            result = model.compute_static_inference_metrics(
                observation(transformed, a.device),
                tensor(transformed["actions"], a.device, torch.float32).unsqueeze(0),
                num_steps=a.num_steps,
                metric_dim_mask=mask,
            )
            cosines.append(np.stack([x.squeeze(0).cpu().numpy() for x in result["cosine_steps"]]))
            for step, x in enumerate(result["loss_steps"]):
                losses[step].append(x.squeeze(0).float().cpu().numpy())
            for step, x in enumerate(result["vision_gradnorm_steps"]):
                grads[step].append(x.squeeze(0).float().cpu().numpy())
            if a.save_meta:
                targets.append(result["target"].squeeze(0).cpu().numpy())
                velocities.append(np.stack([x.squeeze(0).cpu().numpy() for x in result["velocity_steps"]]))
        out = a.output_dir / f"episode_{episode.episode_index:03d}"
        out.mkdir(parents=True, exist_ok=True)
        save_array(out / "cosine.npy", cosines, np.float16, (0, a.num_steps, horizon))
        for step in range(a.num_steps):
            save_array(out / f"final_loss_{step}.npy", losses[step], np.float32, (0, horizon))
            save_array(out / f"gradnorm_vision_step_{step}.npy", grads[step], np.float16, (0,))
        if a.save_meta:
            meta = out / "meta"
            meta.mkdir(exist_ok=True)
            save_array(meta / "u.npy", targets, np.float16, (0, horizon, config.action_dim))
            save_array(meta / "v.npy", velocities, np.float16, (0, a.num_steps, horizon, config.action_dim))
        (out / "trajectory_meta.json").write_text(
            json.dumps(
                {
                    "model": a.model,
                    "dataset_name": a.dataset_dir.name,
                    "episode_index": episode.episode_index,
                    "source_length": len(episode),
                    "action_horizon": horizon,
                    "num_frames_used": count,
                    "discarded_tail": min(horizon - 1, len(episode)),
                    "metric_dims": list(dims),
                    "action_source": (
                        "absolute joint targets converted to chunk-start-relative joint deltas; absolute gripper; semantic slot mapping"
                        if a.joint else
                        "absolute Cartesian xyz + quaternion xyzw + binary gripper; direct semantic slot mapping"
                    ),
                    "instruction": episode.instruction,
                    "camera_mapping": {"base": "camera_front", "secondary": "camera_side", "wrist": "camera_wrist"},
                },
                indent=2,
            )
            + "\n"
        )
        total += count
        dataset.clear_video_cache()
    (a.output_dir / "summary.json").write_text(
        json.dumps(
            {"model": a.model, "dataset_name": a.dataset_dir.name, "episodes": len(episodes), "frames": total}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
