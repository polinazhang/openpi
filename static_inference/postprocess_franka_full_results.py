#!/usr/bin/env python3
"""Post-process franka_full static inference results into tables and plots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

# Hard-coded paths as requested.
FRANKA_FULL_ROOT = Path("/coc/testnvme/xzhang3205/static/franka_full")
OUTPUT_ROOT = Path("/coc/testnvme/xzhang3205/openpi/static_results")
TABLE_PATH = OUTPUT_ROOT / "table.txt"

SANITY_DATASETS = ["franka_object", "franka_object_action_ood", "franka_object_vision_ood"]
DOMAIN_DATASETS = ["franka_object", "franka_object_plus", "franka_object_two", "franka_on_top"]

# For these two datasets only keep source_episode_index in [0, 9].
EPISODE_FILTER_DATASETS = {"franka_object", "franka_object_plus"}
EPISODE_MAX = 9

# Layer ids present in cosine output.
LAYER_IDS = [f"{i:02d}" for i in range(18)]


@dataclass
class ScalarStats:
    mean: float
    std: float


@dataclass
class DatasetMode1:
    perturbance: dict[str, ScalarStats]
    cosine_all_layers: ScalarStats
    cosine_last_layer: ScalarStats
    cosine_layer_means: np.ndarray
    cosine_layer_stds: np.ndarray


@dataclass
class DatasetMode2:
    norm_l2: dict[str, ScalarStats]
    norm_l2_sq: dict[str, ScalarStats]
    cosine: dict[str, ScalarStats]
    cosine_layers: dict[str, tuple[np.ndarray, np.ndarray]]


def _load_metadata(folder: Path) -> list[dict]:
    with (folder / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _safe_stats(vals: Iterable[float]) -> ScalarStats:
    arr = np.asarray(list(vals), dtype=np.float64)
    return ScalarStats(mean=float(np.mean(arr)), std=float(np.std(arr)))


def _load_array(base_folder: Path, relpath: str) -> np.ndarray:
    return np.load(base_folder / relpath).astype(np.float64, copy=False)


def _load_gradnorm(base_folder: Path, entry: dict, embedding: str) -> np.ndarray:
    # Read from physical path directly to avoid metadata-key mismatch across reruns.
    traj_rel = Path(entry["trajectory_rel_dir"])
    if traj_rel.name == "npy-metadata":
        path = base_folder / traj_rel / f"gradnorm_{embedding}_step_0.npy"
    else:
        path = base_folder / traj_rel / "npy-metadata" / f"gradnorm_{embedding}_step_0.npy"
    if not path.exists():
        print(f"[WARN] Missing perturbance file: {path}; returning NaN placeholder.")
        return np.asarray([np.nan], dtype=np.float64)
    return np.load(path).astype(np.float64, copy=False)


def _mean_of_array(a: np.ndarray) -> float:
    return float(np.mean(a))


def _layer_cosine_from_collapsed(cosine_folder: Path, entry: dict, prefix: str) -> np.ndarray:
    layer_vals: list[float] = []
    for lid in LAYER_IDS:
        arr = _load_array(cosine_folder, entry["artifacts"][f"{prefix}-cosine_{lid}"])
        layer_vals.append(float(np.mean(arr)))
    return np.asarray(layer_vals, dtype=np.float64)


def _cosine_from_vectors(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    # v and u: [..., H, D], cosine over last 2 dims.
    eps = 1e-12
    dot = np.sum(v * u, axis=(-1, -2))
    v_norm = np.sqrt(np.sum(np.square(v), axis=(-1, -2)))
    u_norm = np.sqrt(np.sum(np.square(u), axis=(-1, -2)))
    return dot / (v_norm * u_norm + eps)


def compute_mode1(dataset: str) -> DatasetMode1:
    base = FRANKA_FULL_ROOT / dataset
    perturb_folder = base / "perturbance"
    cosine_folder = base / "cosine"

    perturb_entries = _filter_entries(dataset, _load_metadata(perturb_folder))
    cosine_entries = _filter_entries(dataset, _load_metadata(cosine_folder))

    perturb_vals: dict[str, list[float]] = {"vision": [], "action": [], "time": []}
    for entry in perturb_entries:
        for emb in ("vision", "action", "time"):
            arr = _load_gradnorm(perturb_folder, entry, emb)
            perturb_vals[emb].append(_mean_of_array(arr))

    cosine_layer_per_traj: list[np.ndarray] = []
    for entry in cosine_entries:
        cosine_layer_per_traj.append(_layer_cosine_from_collapsed(cosine_folder, entry, "ctraining"))

    layer_matrix = np.stack(cosine_layer_per_traj, axis=0)  # [T, L]
    per_traj_all_layers = np.mean(layer_matrix, axis=1)
    per_traj_last_layer = layer_matrix[:, -1]

    return DatasetMode1(
        perturbance={k: _safe_stats(v) for k, v in perturb_vals.items()},
        cosine_all_layers=_safe_stats(per_traj_all_layers),
        cosine_last_layer=_safe_stats(per_traj_last_layer),
        cosine_layer_means=np.mean(layer_matrix, axis=0),
        cosine_layer_stds=np.std(layer_matrix, axis=0),
    )


def compute_mode2(dataset: str) -> DatasetMode2:
    base = FRANKA_FULL_ROOT / dataset
    gt_folder = base / "gradient-training"
    gi_folder = base / "gradient-inference"
    cosine_folder = base / "cosine"

    gt_entries = _filter_entries(dataset, _load_metadata(gt_folder))
    gi_entries = _filter_entries(dataset, _load_metadata(gi_folder))
    cos_entries = _filter_entries(dataset, _load_metadata(cosine_folder))

    if not (len(gt_entries) == len(gi_entries) == len(cos_entries)):
        raise ValueError(
            f"Entry mismatch for {dataset}: gt={len(gt_entries)} gi={len(gi_entries)} cos={len(cos_entries)}"
        )

    l2_vals: dict[str, list[float]] = {"action": [], "vision": [], "all": []}
    l2_sq_vals: dict[str, list[float]] = {"action": [], "vision": [], "all": []}
    cosine_vals: dict[str, list[float]] = {"action": [], "vision": [], "all": []}
    cosine_layers_accum: dict[str, list[np.ndarray]] = {"action": [], "vision": [], "all": []}

    for gt_entry, gi_entry, cos_entry in zip(gt_entries, gi_entries, cos_entries, strict=True):
        v_action = _load_array(gt_folder, gt_entry["artifacts"]["gradient_step_0"])  # [F,H,D]

        step_ids = sorted(
            int(k.split("_")[-1])
            for k in gi_entry["artifacts"]
            if k.startswith("gradient_step_")
        )
        v_all_steps = [
            _load_array(gi_folder, gi_entry["artifacts"][f"gradient_step_{sid}"])
            for sid in step_ids
        ]
        v_all = np.stack(v_all_steps, axis=1)  # [F,S,H,D]

        v_action_4d = np.repeat(v_action[:, None, :, :], v_all.shape[1], axis=1)
        v_vision = v_all - v_action_4d

        for name, vec in (("action", v_action_4d), ("vision", v_vision), ("all", v_all)):
            l2_sq = np.sum(np.square(vec), axis=(-1, -2))  # [F,S]
            l2 = np.sqrt(l2_sq)
            l2_vals[name].append(float(np.mean(l2)))
            l2_sq_vals[name].append(float(np.mean(l2_sq)))

        # Cosine standard 2 derived from raw vectors (not collapsed cosine arrays).
        layer_action: list[float] = []
        layer_vision: list[float] = []
        layer_all: list[float] = []
        for lid in LAYER_IDS:
            va = _load_array(cosine_folder, cos_entry["artifacts"][f"meta/ctraining-v_{lid}"])  # [F,H,D]
            vall = _load_array(cosine_folder, cos_entry["artifacts"][f"meta/cinference-v_{lid}"])  # [F,S,H,D]
            u = _load_array(cosine_folder, cos_entry["artifacts"]["meta/u"])  # [F,H,D]

            va4 = np.repeat(va[:, None, :, :], vall.shape[1], axis=1)
            u4 = np.repeat(u[:, None, :, :], vall.shape[1], axis=1)
            vv = vall - va4

            ca = _cosine_from_vectors(va4, u4)
            cv = _cosine_from_vectors(vv, u4)
            call = _cosine_from_vectors(vall, u4)

            layer_action.append(float(np.mean(ca)))
            layer_vision.append(float(np.mean(cv)))
            layer_all.append(float(np.mean(call)))

        la = np.asarray(layer_action, dtype=np.float64)
        lv = np.asarray(layer_vision, dtype=np.float64)
        ll = np.asarray(layer_all, dtype=np.float64)

        cosine_layers_accum["action"].append(la)
        cosine_layers_accum["vision"].append(lv)
        cosine_layers_accum["all"].append(ll)

        cosine_vals["action"].append(float(np.mean(la)))
        cosine_vals["vision"].append(float(np.mean(lv)))
        cosine_vals["all"].append(float(np.mean(ll)))

    cosine_layer_stats = {
        name: (
            np.mean(np.stack(vals, axis=0), axis=0),
            np.std(np.stack(vals, axis=0), axis=0),
        )
        for name, vals in cosine_layers_accum.items()
    }

    return DatasetMode2(
        norm_l2={k: _safe_stats(v) for k, v in l2_vals.items()},
        norm_l2_sq={k: _safe_stats(v) for k, v in l2_sq_vals.items()},
        cosine={k: _safe_stats(v) for k, v in cosine_vals.items()},
        cosine_layers=cosine_layer_stats,
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


def _write_latex_table_mode1(title: str, datasets: list[str], mode1: dict[str, DatasetMode1]) -> str:
    safe_title = _latex_escape(title)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Dataset & Perturb-vision & Perturb-action & Perturb-time & Cosine(all layers) & Cosine(last layer) \\",
        r"\hline",
    ]
    for ds in datasets:
        row = mode1[ds]
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                _latex_escape(ds),
                _format_stat(row.perturbance["vision"]),
                _format_stat(row.perturbance["action"]),
                _format_stat(row.perturbance["time"]),
                _format_stat(row.cosine_all_layers),
                _format_stat(row.cosine_last_layer),
            )
        )
    lines.extend([r"\hline", r"\end{tabular}", rf"\caption{{{safe_title}}}", r"\end{table}"])
    return "\n".join(lines)


def _write_latex_table_mode2(title: str, datasets: list[str], mode2: dict[str, DatasetMode2]) -> str:
    safe_title = _latex_escape(title)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lccccccccc}",
        r"\hline",
        r"Dataset & $||v_a||_2$ & $||v_v||_2$ & $||v_{all}||_2$ & $||v_a||_2^2$ & $||v_v||_2^2$ & $||v_{all}||_2^2$ & Cos-a & Cos-v & Cos-all \\",
        r"\hline",
    ]
    for ds in datasets:
        row = mode2[ds]
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _latex_escape(ds),
                _format_stat(row.norm_l2["action"]),
                _format_stat(row.norm_l2["vision"]),
                _format_stat(row.norm_l2["all"]),
                _format_stat(row.norm_l2_sq["action"]),
                _format_stat(row.norm_l2_sq["vision"]),
                _format_stat(row.norm_l2_sq["all"]),
                _format_stat(row.cosine["action"]),
                _format_stat(row.cosine["vision"]),
                _format_stat(row.cosine["all"]),
            )
        )
    lines.extend([r"\hline", r"\end{tabular}", rf"\caption{{{safe_title}}}", r"\end{table}"])
    return "\n".join(lines)


def _plot_mode1(group_name: str, datasets: list[str], mode1: dict[str, DatasetMode1], out_dir: Path) -> None:
    x = np.arange(len(LAYER_IDS))
    plt.figure(figsize=(9, 5))
    for ds in datasets:
        y = mode1[ds].cosine_layer_means
        s = mode1[ds].cosine_layer_stds
        plt.plot(x, y, label=ds)
        plt.fill_between(x, y - s, y + s, alpha=0.2)
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity")
    plt.title(f"{group_name} Mode 1: condition-training cosine per layer")
    plt.xticks(x, LAYER_IDS)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_per_layer.png", dpi=180)
    plt.close()


def _plot_mode2(group_name: str, datasets: list[str], mode2: dict[str, DatasetMode2], out_dir: Path) -> None:
    x = np.arange(len(LAYER_IDS))
    for kind, fname in (("action", "cosine_action_per_layer.png"), ("vision", "cosine_vision_per_layer.png"), ("all", "cosine_all_per_layer.png")):
        plt.figure(figsize=(9, 5))
        for ds in datasets:
            y, s = mode2[ds].cosine_layers[kind]
            plt.plot(x, y, label=ds)
            plt.fill_between(x, y - s, y + s, alpha=0.2)
        plt.xlabel("Layer")
        plt.ylabel("Cosine similarity")
        plt.title(f"{group_name} Mode 2: cosine-{kind} per layer")
        plt.xticks(x, LAYER_IDS)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=180)
        plt.close()


def main() -> None:
    mode1: dict[str, DatasetMode1] = {}
    mode2: dict[str, DatasetMode2] = {}

    all_datasets = sorted(set(SANITY_DATASETS + DOMAIN_DATASETS))
    for ds in all_datasets:
        mode1[ds] = compute_mode1(ds)
        mode2[ds] = compute_mode2(ds)

    sanity_mode1_dir = OUTPUT_ROOT / "sanity" / "mode1"
    sanity_mode2_dir = OUTPUT_ROOT / "sanity" / "mode2"
    domain_mode1_dir = OUTPUT_ROOT / "domain_gap" / "mode1"
    domain_mode2_dir = OUTPUT_ROOT / "domain_gap" / "mode2"
    for d in [sanity_mode1_dir, sanity_mode2_dir, domain_mode1_dir, domain_mode2_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _plot_mode1("Sanity", SANITY_DATASETS, mode1, sanity_mode1_dir)
    _plot_mode2("Sanity", SANITY_DATASETS, mode2, sanity_mode2_dir)
    _plot_mode1("Domain Gap", DOMAIN_DATASETS, mode1, domain_mode1_dir)
    _plot_mode2("Domain Gap", DOMAIN_DATASETS, mode2, domain_mode2_dir)

    tables = [
        _write_latex_table_mode1("Sanity check, Mode 1 (franka_object / action_ood / vision_ood)", SANITY_DATASETS, mode1),
        _write_latex_table_mode2("Sanity check, Mode 2 (franka_object / action_ood / vision_ood)", SANITY_DATASETS, mode2),
        _write_latex_table_mode1("Domain gap, Mode 1 (franka_object / plus / two / on_top)", DOMAIN_DATASETS, mode1),
        _write_latex_table_mode2("Domain gap, Mode 2 (franka_object / plus / two / on_top)", DOMAIN_DATASETS, mode2),
    ]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n\n".join(tables) + "\n", encoding="utf-8")

    print(f"Wrote tables: {TABLE_PATH}")
    print(f"Wrote plots under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
