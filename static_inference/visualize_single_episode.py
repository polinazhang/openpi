#!/usr/bin/env python3
"""Visualize one prototype static-inference trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROTOTYPE_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation/prototype")
NUM_LAYERS = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one prototype static inference result.")
    parser.add_argument("--benchmark", required=True, choices=["robocasa", "mesa", "mesa-unfinetuned", "franka"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--static-root",
        type=Path,
        default=None,
        help="Defaults to /prototype/static/<benchmark>/<dataset>/perturbance-all.",
    )
    parser.add_argument(
        "--visual-root",
        type=Path,
        default=None,
        help="Defaults to /prototype/visuals/<benchmark>/<dataset>/episode_000000.",
    )
    parser.add_argument("--output-root", type=Path, default=PROTOTYPE_ROOT / "results")
    return parser.parse_args()


def _load_npy_if_exists(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def _load_cosine(npy_dir: Path) -> dict[str, np.ndarray] | None:
    layers = []
    for layer_idx in range(NUM_LAYERS):
        arr = _load_npy_if_exists(npy_dir / f"cinference-cosine_{layer_idx:02d}.npy")
        if arr is None:
            break
        layers.append(arr.mean(axis=-1))
    if not layers:
        return None
    stacked = np.stack(layers, axis=-1)
    return {
        "final_layer": stacked[..., -1],
        "all_layers": stacked.mean(axis=-1),
        "per_layer": stacked,
    }


def _load_gradnorm(npy_dir: Path) -> np.ndarray | None:
    steps = []
    for step_idx in range(100):
        arr = _load_npy_if_exists(npy_dir / f"gradnorm_vision_step_{step_idx}.npy")
        if arr is None:
            break
        steps.append(arr)
    if not steps:
        return None
    return np.stack(steps, axis=-1)


def _plot_steps(arr: np.ndarray | None, title: str, ylabel: str, out_path: Path) -> None:
    if arr is None:
        return
    x = np.arange(1, arr.shape[-1] + 1)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, mean)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_cosine_layers(cosine: dict[str, np.ndarray] | None, out_path: Path) -> None:
    if cosine is None:
        return
    per_layer = cosine["per_layer"]
    first_step = per_layer[:, 0, :] if per_layer.ndim == 3 else per_layer
    mean = first_step.mean(axis=0)
    std = first_step.std(axis=0)
    x = np.arange(first_step.shape[-1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, mean)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_title("Cosine by Layer, First Diffusion Step")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summary_stats(arr: np.ndarray | None) -> dict | None:
    if arr is None:
        return None
    return {
        "shape": list(arr.shape),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    args = parse_args()
    static_root = args.static_root or (PROTOTYPE_ROOT / "static" / args.benchmark / args.dataset / "perturbance-all")
    visual_root = args.visual_root or (PROTOTYPE_ROOT / "visuals" / args.benchmark / args.dataset / "episode_000000")
    npy_dir = static_root / "000000" / "npy-metadata"
    if not npy_dir.exists():
        raise SystemExit(f"ERROR: missing trajectory npy metadata dir: {npy_dir}")

    output_dir = args.output_root / args.benchmark / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    cosine = _load_cosine(npy_dir)
    gradnorm = _load_gradnorm(npy_dir)
    displacement = _load_npy_if_exists(npy_dir / "displacement_norm.npy")

    _plot_steps(cosine["final_layer"] if cosine else None, "Final-Layer Cosine", "Cosine", output_dir / "cosine_final.png")
    _plot_steps(cosine["all_layers"] if cosine else None, "All-Layer Cosine", "Cosine", output_dir / "cosine_all.png")
    _plot_cosine_layers(cosine, output_dir / "cosine_layers_first_step.png")
    _plot_steps(gradnorm, "Vision Gradnorm", "Norm", output_dir / "gradnorm_vision.png")
    _plot_steps(displacement, "Displacement Norm", "Norm", output_dir / "displacement_norm.png")

    visual_metadata = None
    visual_metadata_path = visual_root / "metadata.json"
    if visual_metadata_path.exists():
        visual_metadata = json.loads(visual_metadata_path.read_text(encoding="utf-8"))

    summary = {
        "benchmark": args.benchmark,
        "dataset": args.dataset,
        "static_root": str(static_root),
        "visual_root": str(visual_root),
        "visual_metadata": visual_metadata,
        "cosine_final": _summary_stats(cosine["final_layer"] if cosine else None),
        "cosine_all": _summary_stats(cosine["all_layers"] if cosine else None),
        "gradnorm_vision": _summary_stats(gradnorm),
        "displacement_norm": _summary_stats(displacement),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Wrote visualization outputs to {output_dir}")


if __name__ == "__main__":
    main()
