#!/usr/bin/env python3
"""Export videos and raw actions for one resolved prototype episode."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

import static_inference_single as _single


PROTOTYPE_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation/prototype")
MAX_VIDEO_BYTES = 99 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render/copy one selected dataset episode.")
    parser.add_argument("--dataset", choices=sorted(_single._static.DATASETS.keys()), required=True)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--use-first-episode", type=_single._static.parse_bool, default=True)
    parser.add_argument("--output-root", type=Path, default=PROTOTYPE_ROOT / "visuals")
    parser.add_argument(
        "--output-benchmark",
        default=None,
        help="Output benchmark directory name. Defaults to the resolved source benchmark.",
    )
    parser.add_argument("--static-output-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(_single._static.BASE_CHECKPOINT_URI))
    parser.add_argument("--norm-stats-dir", type=Path, default=None)
    parser.add_argument("--video-backend", choices=["pyav", "torchcodec", "video_reader"], default="pyav")
    parser.add_argument("--data.default_prompt", dest="data_default_prompt", default=None)
    parser.add_argument("--mesa-root", type=Path, default=_single._mesa_dataset.DEFAULT_MESA_ROOT)
    parser.add_argument("--robocasa-base", type=Path, default=_single._robocasa_dataset.DEFAULT_DATASET_BASE)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _camera_filename(camera_name: str) -> str:
    return camera_name.replace("/", "_") + ".mp4"


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _copy_or_segment_video(
    *,
    src: Path,
    dst: Path,
    start_sec: float,
    duration_sec: float,
    copy_full_file: bool,
    dry_run: bool,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_full_file:
        print(f"copy {src} -> {dst}")
        if not dry_run:
            shutil.copy2(src, dst)
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.6f}",
            "-i",
            str(src),
            "-t",
            f"{duration_sec:.6f}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            str(dst),
        ]
        _run(cmd, dry_run=dry_run)

    if not dry_run and dst.stat().st_size > MAX_VIDEO_BYTES:
        compressed = dst.with_suffix(".compressed.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(dst),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "30",
            str(compressed),
        ]
        _run(cmd, dry_run=False)
        compressed.replace(dst)


def main() -> None:
    args = parse_args()
    _single._require_episode_choice(args)
    dataset_cfg, _, dataset, _ = _single.load_dataset_and_transform(args)
    episode = _single.resolve_episode_slice(dataset_cfg, dataset, args.episode_index).limited(args.max_frames)
    actions = _single.load_episode_actions(episode)

    output_benchmark = args.output_benchmark or episode.benchmark
    output_dir = args.output_root / output_benchmark / args.dataset / "episode_000000"
    videos_dir = output_dir / "videos"
    if args.dry_run:
        print(f"Would write render output to {output_dir}")
    else:
        videos_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "actions.npy", actions.astype(np.float32))

    duration_sec = episode.num_frames / float(episode.fps)
    source_video_start_sec = 0.0 if episode.benchmark in {"robocasa", "mesa"} else episode.start / float(episode.fps)
    copy_full_file = args.max_frames is None and episode.benchmark in {"robocasa", "mesa"}
    video_outputs: dict[str, str] = {}
    for camera_name in episode.camera_names:
        src = _single.episode_video_path(episode, camera_name)
        dst = videos_dir / _camera_filename(camera_name)
        if not src.exists():
            raise SystemExit(f"ERROR: missing source video for {camera_name}: {src}")
        _copy_or_segment_video(
            src=src,
            dst=dst,
            start_sec=source_video_start_sec,
            duration_sec=duration_sec,
            copy_full_file=copy_full_file,
            dry_run=args.dry_run,
        )
        video_outputs[camera_name] = str(dst.relative_to(output_dir))

    frames = [
        {
            "frame_index": int(i),
            "timestamp_sec": float(i / episode.fps),
            "action_index": int(i),
        }
        for i in range(actions.shape[0])
    ]
    metadata = {
        "benchmark": episode.benchmark,
        "output_benchmark": output_benchmark,
        "dataset": args.dataset,
        "source_episode_index": episode.source_episode_index,
        "local_episode_index": episode.local_episode_index,
        "label": episode.label,
        "dataset_root": str(episode.dataset_root),
        "fps": episode.fps,
        "camera_names": list(episode.camera_names),
        "videos": video_outputs,
        "action_key": episode.action_key,
        "action_shape": list(actions.shape),
        "num_frames": int(actions.shape[0]),
        "static_output_root": str(args.static_output_root) if args.static_output_root else None,
        "alignment": "video frame i aligns with actions.npy[i] and frames.json[i]",
    }

    if not args.dry_run:
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        with (output_dir / "frames.json").open("w", encoding="utf-8") as handle:
            json.dump(frames, handle, indent=2)
            handle.write("\n")
        with (output_dir / "actions.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "action_key": episode.action_key,
                    "action_shape": list(actions.shape),
                    "actions_npy": "actions.npy",
                },
                handle,
                indent=2,
            )
            handle.write("\n")
    print(f"Rendered episode {episode.source_episode_index} to {output_dir}")


if __name__ == "__main__":
    main()
