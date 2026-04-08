#!/usr/bin/env python3
"""Post-process franka_full Mode 3 outputs into tables and plots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

# Hard-coded paths as requested.
FRANKA_FULL_ROOT = Path("/coc/testnvme/xzhang3205/static/franka_full")
OUTPUT_ROOT = Path("/coc/testnvme/xzhang3205/openpi/static_results/mode3")
TABLE_PATH = OUTPUT_ROOT / "table.txt"

SANITY_DATASETS = ["franka_object", "franka_object_action_ood", "franka_object_vision_ood"]
DOMAIN_DATASETS = ["franka_object", "franka_object_plus", "franka_object_two", "franka_on_top"]

# For these two datasets only keep source_episode_index in [0, 9].
EPISODE_FILTER_DATASETS = {"franka_object", "franka_object_plus"}
EPISODE_MAX = 9

DIFFUSION_STEPS = list(range(10))  # Saved as step_0..step_9, displayed as 1..10.
LAYER_IDS = [f"{i:02d}" for i in range(18)]


@dataclass
class ScalarStats:
    mean: float
    std: float


@dataclass
class DatasetMode3:
    perturb_step_mean: np.ndarray  # [S]
    perturb_step_std: np.ndarray  # [S]
    cosine_all_step_mean: np.ndarray  # [S]
    cosine_all_step_std: np.ndarray  # [S]
    cosine_last_step_mean: np.ndarray  # [S]
    cosine_last_step_std: np.ndarray  # [S]
    perturb_first: ScalarStats
    cosine_all_first: ScalarStats
    cosine_last_first: ScalarStats
    perturb_over_steps: ScalarStats
    cosine_all_over_steps: ScalarStats
    cosine_last_over_steps: ScalarStats


def _safe_stats(vals: Iterable[float]) -> ScalarStats:
    arr = np.asarray(list(vals), dtype=np.float64)
    if arr.size == 0:
        return ScalarStats(mean=float("nan"), std=float("nan"))
    return ScalarStats(mean=float(np.nanmean(arr)), std=float(np.nanstd(arr)))


def _load_metadata(folder: Path) -> list[dict]:
    with (folder / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _trajectory_id_from_relpath(relpath: str) -> str:
    # e.g. "000007/npy-metadata" -> "000007"
    return Path(relpath).parts[0]


def _filter_entries(dataset: str, entries: list[dict]) -> list[dict]:
    if dataset not in EPISODE_FILTER_DATASETS:
        return entries
    out: list[dict] = []
    for entry in entries:
        epi = entry.get("source_episode_index")
        if epi is None:
            continue
        if 0 <= int(epi) <= EPISODE_MAX:
            out.append(entry)
    return out


def _load_array(base_folder: Path, relpath: str) -> np.ndarray:
    return np.load(base_folder / relpath).astype(np.float64, copy=False)


def _mean_over_all_except_step(arr: np.ndarray) -> np.ndarray:
    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2 dims for step-aware tensor, got shape={arr.shape}")
    axes = tuple(i for i in range(arr.ndim) if i != 1)
    return np.mean(arr, axis=axes)


def _load_perturb_noise_step(base: Path, dataset: str, traj_id: str, step: int) -> np.ndarray:
    candidates = [
        base / "perturbance-noise" / dataset / traj_id / "npy-metadata" / f"gradnorm_vision_step_{step}.npy",
        base / "perturbance-noise" / traj_id / "npy-metadata" / f"gradnorm_vision_step_{step}.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path).astype(np.float64, copy=False)
    raise FileNotFoundError(f"Missing perturbance-noise gradnorm file for dataset={dataset} traj={traj_id} step={step}")


def compute_mode3(dataset: str) -> DatasetMode3:
    base = FRANKA_FULL_ROOT / dataset
    cosine_folder = base / "cosine"
    cosine_entries = _filter_entries(dataset, _load_metadata(cosine_folder))

    traj_ids = [_trajectory_id_from_relpath(entry["trajectory_rel_dir"]) for entry in cosine_entries]
    if len(traj_ids) == 0:
        raise ValueError(f"No trajectories available for dataset={dataset}")

    perturb_rows: list[np.ndarray] = []
    cosine_all_rows: list[np.ndarray] = []
    cosine_last_rows: list[np.ndarray] = []

    for entry, traj_id in zip(cosine_entries, traj_ids):
        perturb_vals = []
        for step in DIFFUSION_STEPS:
            arr = _load_perturb_noise_step(base, dataset, traj_id, step)
            perturb_vals.append(float(np.mean(arr)))
        perturb_rows.append(np.asarray(perturb_vals, dtype=np.float64))

        per_layer_steps: list[np.ndarray] = []
        for lid in LAYER_IDS:
            key = f"cinference-cosine_{lid}"
            arr = _load_array(cosine_folder, entry["artifacts"][key])
            step_vals = _mean_over_all_except_step(arr)
            per_layer_steps.append(step_vals[: len(DIFFUSION_STEPS)])
        layer_step = np.stack(per_layer_steps, axis=0)  # [L, S]
        cosine_all_rows.append(np.mean(layer_step, axis=0))
        cosine_last_rows.append(layer_step[-1, :])

    perturb_matrix = np.stack(perturb_rows, axis=0)  # [T, S]
    cosine_all_matrix = np.stack(cosine_all_rows, axis=0)  # [T, S]
    cosine_last_matrix = np.stack(cosine_last_rows, axis=0)  # [T, S]

    return DatasetMode3(
        perturb_step_mean=np.nanmean(perturb_matrix, axis=0),
        perturb_step_std=np.nanstd(perturb_matrix, axis=0),
        cosine_all_step_mean=np.nanmean(cosine_all_matrix, axis=0),
        cosine_all_step_std=np.nanstd(cosine_all_matrix, axis=0),
        cosine_last_step_mean=np.nanmean(cosine_last_matrix, axis=0),
        cosine_last_step_std=np.nanstd(cosine_last_matrix, axis=0),
        perturb_first=_safe_stats(perturb_matrix[:, 0]),
        cosine_all_first=_safe_stats(cosine_all_matrix[:, 0]),
        cosine_last_first=_safe_stats(cosine_last_matrix[:, 0]),
        perturb_over_steps=_safe_stats(np.nanmean(perturb_matrix, axis=1)),
        cosine_all_over_steps=_safe_stats(np.nanmean(cosine_all_matrix, axis=1)),
        cosine_last_over_steps=_safe_stats(np.nanmean(cosine_last_matrix, axis=1)),
    )


def _format_stat(s: ScalarStats) -> str:
    return f"{s.mean:.6f} $\\pm$ {s.std:.6f}"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("$", "\\$")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _write_latex_table(
    title: str,
    datasets: list[str],
    results: dict[str, DatasetMode3],
    use_first_step: bool,
) -> str:
    safe_title = _latex_escape(title)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Dataset & Perturbance norm (vision) & Cosine(all layers) & Cosine(last layer) \\",
        r"\hline",
    ]
    for ds in datasets:
        row = results[ds]
        if use_first_step:
            p, ca, cl = row.perturb_first, row.cosine_all_first, row.cosine_last_first
        else:
            p, ca, cl = row.perturb_over_steps, row.cosine_all_over_steps, row.cosine_last_over_steps
        lines.append(
            "{} & {} & {} & {} \\\\".format(
                _latex_escape(ds),
                _format_stat(p),
                _format_stat(ca),
                _format_stat(cl),
            )
        )
    lines.extend([r"\hline", r"\end{tabular}", rf"\caption{{{safe_title}}}", r"\end{table}"])
    return "\n".join(lines)


def _plot_metric(
    datasets: list[str],
    results: dict[str, DatasetMode3],
    mean_attr: str,
    std_attr: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    x = np.arange(1, len(DIFFUSION_STEPS) + 1)
    plt.figure(figsize=(9, 5))
    for ds in datasets:
        y = getattr(results[ds], mean_attr)
        s = getattr(results[ds], std_attr)
        plt.plot(x, y, label=ds)
        plt.fill_between(x, y - s, y + s, alpha=0.2)
    plt.xlabel("Diffusion step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    results: dict[str, DatasetMode3] = {}
    all_datasets = sorted(set(SANITY_DATASETS + DOMAIN_DATASETS))
    for ds in all_datasets:
        results[ds] = compute_mode3(ds)

    sanity_dir = OUTPUT_ROOT / "sanity"
    domain_dir = OUTPUT_ROOT / "domain_gap"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    domain_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric(
        SANITY_DATASETS,
        results,
        "perturb_step_mean",
        "perturb_step_std",
        "Perturbance norm vision",
        "Sanity check, Mode 3: perturbance-noise vision",
        sanity_dir / "perturbance_vision_per_step.png",
    )
    _plot_metric(
        SANITY_DATASETS,
        results,
        "cosine_all_step_mean",
        "cosine_all_step_std",
        "Cosine(all layers)",
        "Sanity check, Mode 3: condition-inference cosine(all layers)",
        sanity_dir / "cosine_all_layers_per_step.png",
    )
    _plot_metric(
        SANITY_DATASETS,
        results,
        "cosine_last_step_mean",
        "cosine_last_step_std",
        "Cosine(last layer)",
        "Sanity check, Mode 3: condition-inference cosine(last layer)",
        sanity_dir / "cosine_last_layer_per_step.png",
    )

    _plot_metric(
        DOMAIN_DATASETS,
        results,
        "perturb_step_mean",
        "perturb_step_std",
        "Perturbance norm vision",
        "Domain gap, Mode 3: perturbance-noise vision",
        domain_dir / "perturbance_vision_per_step.png",
    )
    _plot_metric(
        DOMAIN_DATASETS,
        results,
        "cosine_all_step_mean",
        "cosine_all_step_std",
        "Cosine(all layers)",
        "Domain gap, Mode 3: condition-inference cosine(all layers)",
        domain_dir / "cosine_all_layers_per_step.png",
    )
    _plot_metric(
        DOMAIN_DATASETS,
        results,
        "cosine_last_step_mean",
        "cosine_last_step_std",
        "Cosine(last layer)",
        "Domain gap, Mode 3: condition-inference cosine(last layer)",
        domain_dir / "cosine_last_layer_per_step.png",
    )

    tables = [
        _write_latex_table(
            "Sanity check, Mode 3 (franka_object / action_ood / vision_ood), first diffusion step",
            SANITY_DATASETS,
            results,
            use_first_step=True,
        ),
        _write_latex_table(
            "Sanity check, Mode 3 (franka_object / action_ood / vision_ood), average over all diffusion steps",
            SANITY_DATASETS,
            results,
            use_first_step=False,
        ),
        _write_latex_table(
            "Domain gap, Mode 3 (franka_object / plus / two / on_top), first diffusion step",
            DOMAIN_DATASETS,
            results,
            use_first_step=True,
        ),
        _write_latex_table(
            "Domain gap, Mode 3 (franka_object / plus / two / on_top), average over all diffusion steps",
            DOMAIN_DATASETS,
            results,
            use_first_step=False,
        ),
    ]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n\n".join(tables) + "\n", encoding="utf-8")

    print(f"Wrote tables: {TABLE_PATH}")
    print(f"Wrote plots under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
