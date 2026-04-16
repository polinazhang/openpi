#!/usr/bin/env python3
"""
Post-processing analysis for static inference results.
Generates tables and plots for franka, openarm, and ood datasets.

Outputs:
    static_results/franka/   -> franka.txt, plots
    static_results/openarm/  -> openarm.txt, plots
    static_results/ood/      -> ood.txt, plots
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# ============================================================
# Configuration - hardcoded paths
# ============================================================

FRANKA_ROOT  = "/coc/testnvme/xzhang3205/static/franka_corrected"
OPENARM_ROOT = "/coc/testnvme/xzhang3205/static/openarm_corrected"
OOD_ROOT     = "/coc/testnvme/xzhang3205/static/ood_corrected"
OUTPUT_ROOT  = "/coc/testnvme/xzhang3205/openpi/static_results"

FRANKA_DATASETS  = ["franka_object", "franka_object_plus", "franka_object_two", "franka_on_top"]
OPENARM_DATASETS = ["pick_cup", "pour_ice", "use_spoon", "use_steel_spoon"]
OOD_DATASETS     = [
    "franka_object_action_ood",
    "franka_object_single",
    "franka_object_vision_ood_addition",
    "franka_object_vision_ood_replace",
]

NUM_LAYERS      = 18
NUM_DIFF_STEPS  = 10
EPISODES_0_9    = [f"{i:06d}" for i in range(10)]  # 000000 .. 000009

GIF_INTERVAL_MS = 500  # milliseconds between GIF frames

# ============================================================
# Directory helpers
# ============================================================

def _find_episode_base(metric_dir: str) -> str | None:
    """
    Return the directory that directly contains episode subdirs (000000, ...).
    Handles up to one level of extra nesting (e.g. metric_dir/<dataset>/000000/).
    Returns None if the directory does not exist or has no episode subdirs.
    """
    if not os.path.isdir(metric_dir):
        return None
    entries = sorted(
        e for e in os.listdir(metric_dir)
        if os.path.isdir(os.path.join(metric_dir, e))
    )
    if not entries:
        return None
    # Episode dirs look like 6-digit zero-padded numbers
    if entries[0].isdigit() and len(entries[0]) == 6:
        return metric_dir
    # One level deeper
    nested = os.path.join(metric_dir, entries[0])
    nested_entries = sorted(
        e for e in os.listdir(nested)
        if os.path.isdir(os.path.join(nested, e))
    )
    if nested_entries and nested_entries[0].isdigit() and len(nested_entries[0]) == 6:
        return nested
    return None


def _npy_dir(episode_base: str, episode_id: str) -> str:
    return os.path.join(episode_base, episode_id, "npy-metadata")


def _warn_episodes(dataset: str, metric: str, available: list, requested: list):
    """Warn if available episodes are fewer than requested."""
    available_set = set(available)
    missing = [e for e in requested if e not in available_set]
    if missing:
        warnings.warn(
            f"[{dataset}/{metric}] Only {len(available_set & set(requested))}/{len(requested)} "
            f"episodes available (missing: {missing[:5]}{'...' if len(missing) > 5 else ''})",
            RuntimeWarning,
            stacklevel=3,
        )


def _available_episodes(episode_base: str) -> list:
    if not episode_base or not os.path.isdir(episode_base):
        return []
    return sorted(
        e for e in os.listdir(episode_base)
        if os.path.isdir(os.path.join(episode_base, e))
        and e.isdigit() and len(e) == 6
    )

# ============================================================
# Data loaders
# ============================================================

def load_cosine(
    dataset_root: str,
    episodes: list,
    cosine_subdir: str = "cosine",
    warn_tag: str = "",
) -> dict | None:
    """
    Load cinference cosine similarity.

    Files: cinference-cosine_{layer:02d}.npy  shape (N_frames, 10, action_horizon)

    Returns dict:
        'final_layer'  : (total_frames, 10)   cosine for layer 17, mean over action_horizon
        'all_layers'   : (total_frames, 10)   cosine averaged over all 18 layers
        'per_layer'    : (total_frames, 10, 18) cosine per layer (mean over action_horizon)
    """
    metric_dir = os.path.join(dataset_root, cosine_subdir)
    ep_base = _find_episode_base(metric_dir)
    if ep_base is None:
        return None

    avail = _available_episodes(ep_base)
    if warn_tag:
        _warn_episodes(warn_tag, cosine_subdir, avail, episodes)

    final_layer_list, all_layers_list, per_layer_list = [], [], []

    for ep_id in episodes:
        npy_d = _npy_dir(ep_base, ep_id)
        if not os.path.isdir(npy_d):
            continue

        # Load all layers
        layer_frames = []
        for l in range(NUM_LAYERS):
            f = os.path.join(npy_d, f"cinference-cosine_{l:02d}.npy")
            if not os.path.exists(f):
                break
            arr = np.load(f).astype(np.float32)  # (N, 10, action_horizon)
            layer_frames.append(arr.mean(axis=-1))  # (N, 10)
        if len(layer_frames) != NUM_LAYERS:
            continue

        stacked = np.stack(layer_frames, axis=-1)  # (N, 10, 18)
        per_layer_list.append(stacked)
        final_layer_list.append(stacked[:, :, -1])          # (N, 10) - layer 17
        all_layers_list.append(stacked.mean(axis=-1))       # (N, 10)

    if not final_layer_list:
        return None

    return {
        'final_layer': np.concatenate(final_layer_list, axis=0),    # (T, 10)
        'all_layers' : np.concatenate(all_layers_list, axis=0),     # (T, 10)
        'per_layer'  : np.concatenate(per_layer_list, axis=0),      # (T, 10, 18)
    }


def load_cosine_ctraining(
    dataset_root: str,
    episodes: list,
    cosine_subdir: str = "cosine",
    warn_tag: str = "",
) -> dict | None:
    """
    Load ctraining cosine similarity.

    Files: ctraining-cosine_{layer:02d}.npy  shape (N_frames, action_horizon)
    (condition-training runs a single step, so there is no diffusion-step axis)

    Returns dict:
        'final_layer'  : (total_frames,)    cosine for layer 17, mean over action_horizon
        'all_layers'   : (total_frames,)    cosine averaged over all 18 layers
        'per_layer'    : (total_frames, 18) cosine per layer (mean over action_horizon)
    """
    metric_dir = os.path.join(dataset_root, cosine_subdir)
    ep_base = _find_episode_base(metric_dir)
    if ep_base is None:
        return None

    avail = _available_episodes(ep_base)
    if warn_tag:
        _warn_episodes(warn_tag, cosine_subdir + " (ctraining)", avail, episodes)

    final_layer_list, all_layers_list, per_layer_list = [], [], []

    for ep_id in episodes:
        npy_d = _npy_dir(ep_base, ep_id)
        if not os.path.isdir(npy_d):
            continue

        layer_frames = []
        for l in range(NUM_LAYERS):
            f = os.path.join(npy_d, f"ctraining-cosine_{l:02d}.npy")
            if not os.path.exists(f):
                break
            arr = np.load(f).astype(np.float32)  # (N, action_horizon)
            layer_frames.append(arr.mean(axis=-1))  # (N,)
        if len(layer_frames) != NUM_LAYERS:
            continue

        stacked = np.stack(layer_frames, axis=-1)  # (N, 18)
        per_layer_list.append(stacked)
        final_layer_list.append(stacked[:, -1])          # (N,) - layer 17
        all_layers_list.append(stacked.mean(axis=-1))    # (N,)

    if not final_layer_list:
        return None

    return {
        'final_layer': np.concatenate(final_layer_list, axis=0),    # (T,)
        'all_layers' : np.concatenate(all_layers_list, axis=0),     # (T,)
        'per_layer'  : np.concatenate(per_layer_list, axis=0),      # (T, 18)
    }


def load_gradnorm(
    dataset_root: str,
    episodes: list,
    perturbance_subdir: str,
    warn_tag: str = "",
) -> np.ndarray | None:
    """
    Load vision perturbance gradient norm.

    Files: gradnorm_vision_step_{k}.npy  shape (N_frames,)

    Returns (total_frames, 10).
    """
    metric_dir = os.path.join(dataset_root, perturbance_subdir)
    ep_base = _find_episode_base(metric_dir)
    if ep_base is None:
        return None

    avail = _available_episodes(ep_base)
    if warn_tag:
        _warn_episodes(warn_tag, perturbance_subdir, avail, episodes)

    all_data = []
    for ep_id in episodes:
        npy_d = _npy_dir(ep_base, ep_id)
        if not os.path.isdir(npy_d):
            continue
        steps = []
        for k in range(NUM_DIFF_STEPS):
            f = os.path.join(npy_d, f"gradnorm_vision_step_{k}.npy")
            if not os.path.exists(f):
                break
            steps.append(np.load(f).astype(np.float32))  # (N,)
        if len(steps) == NUM_DIFF_STEPS:
            all_data.append(np.stack(steps, axis=-1))  # (N, 10)

    if not all_data:
        return None
    return np.concatenate(all_data, axis=0)  # (T, 10)


def load_gradient(
    dataset_root: str,
    episodes: list,
    gradient_subdir: str = "gradient-inference",
    warn_tag: str = "",
) -> np.ndarray | None:
    """
    Load gradient guidance vector norms.

    Files: gradient_step_{k}.npy  shape (N_frames, action_horizon, action_dim)

    Returns (total_frames, 10) of L2 norms (over full action space per frame).
    """
    metric_dir = os.path.join(dataset_root, gradient_subdir)
    ep_base = _find_episode_base(metric_dir)
    if ep_base is None:
        return None

    avail = _available_episodes(ep_base)
    if warn_tag:
        _warn_episodes(warn_tag, gradient_subdir, avail, episodes)

    all_data = []
    for ep_id in episodes:
        npy_d = _npy_dir(ep_base, ep_id)
        if not os.path.isdir(npy_d):
            continue
        steps = []
        for k in range(NUM_DIFF_STEPS):
            f = os.path.join(npy_d, f"gradient_step_{k}.npy")
            if not os.path.exists(f):
                break
            arr = np.load(f).astype(np.float32)  # (N, action_h, action_dim)
            norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=-1)  # (N,)
            steps.append(norms)
        if len(steps) == NUM_DIFF_STEPS:
            all_data.append(np.stack(steps, axis=-1))  # (N, 10)

    if not all_data:
        return None
    return np.concatenate(all_data, axis=0)  # (T, 10)


def load_displacement(
    dataset_root: str,
    episodes: list,
    perturbance_subdir: str,
    warn_tag: str = "",
) -> np.ndarray | None:
    """
    Load action displacement vector norms.

    File: displacement_norm.npy  shape (N_frames, 10)

    Returns (total_frames, 10).
    """
    metric_dir = os.path.join(dataset_root, perturbance_subdir)
    ep_base = _find_episode_base(metric_dir)
    if ep_base is None:
        return None

    avail = _available_episodes(ep_base)
    if warn_tag:
        _warn_episodes(warn_tag, perturbance_subdir, avail, episodes)

    all_data = []
    for ep_id in episodes:
        npy_d = _npy_dir(ep_base, ep_id)
        if not os.path.isdir(npy_d):
            continue
        f = os.path.join(npy_d, "displacement_norm.npy")
        if os.path.exists(f):
            all_data.append(np.load(f).astype(np.float32))  # (N, 10)

    if not all_data:
        return None
    return np.concatenate(all_data, axis=0)  # (T, 10)

# ============================================================
# Per-dataset data assembly
# ============================================================

def collect_franka(dataset: str) -> dict:
    root = os.path.join(FRANKA_ROOT, dataset)
    eps  = EPISODES_0_9
    return {
        'cosine'      : load_cosine(root, eps, "cosine"),
        'gradnorm'    : load_gradnorm(root, eps, "perturbance-noise-displacement"),
        'gradient'    : load_gradient(root, eps, "gradient-inference"),
        'displacement': load_displacement(root, eps, "perturbance-noise-displacement"),
    }


def collect_openarm(dataset: str) -> dict:
    root = os.path.join(OPENARM_ROOT, dataset)
    eps  = EPISODES_0_9
    tag  = dataset  # for warnings
    return {
        'cosine'      : load_cosine(root, eps, "cosine", warn_tag=tag),
        'gradnorm'    : load_gradnorm(root, eps, "perturbance-noise", warn_tag=tag),
        'gradient'    : load_gradient(root, eps, "gradient-inference", warn_tag=tag),
        'displacement': None,  # not saved for openarm
    }


def collect_ood(dataset: str) -> dict:
    root = os.path.join(OOD_ROOT, dataset)
    # Use all available episodes for ood
    metric_dir = os.path.join(root, "perturbance-all")
    ep_base    = _find_episode_base(metric_dir)
    eps        = _available_episodes(ep_base) if ep_base else []
    return {
        'cosine'      : load_cosine(root, eps, "perturbance-all"),
        'gradnorm'    : load_gradnorm(root, eps, "perturbance-all"),
        'gradient'    : None,   # not saved for ood
        'displacement': load_displacement(root, eps, "perturbance-all"),
    }

def collect_franka_ctraining(dataset: str) -> dict:
    root = os.path.join(FRANKA_ROOT, dataset)
    return {'cosine': load_cosine_ctraining(root, EPISODES_0_9, "cosine")}


def collect_openarm_ctraining(dataset: str) -> dict:
    root = os.path.join(OPENARM_ROOT, dataset)
    return {'cosine': load_cosine_ctraining(root, EPISODES_0_9, "cosine", warn_tag=dataset)}


def collect_ood_ctraining(dataset: str) -> dict:
    root = os.path.join(OOD_ROOT, dataset)
    # ood perturbance-all has no ctraining cosine
    return {'cosine': None}


# ============================================================
# Statistics helpers
# ============================================================

def step_stats(arr: np.ndarray | None):
    """
    Given (total_frames, 10), return (mean_per_step, std_per_step).
    Shape of each: (10,).
    If arr is None, return (None, None).
    """
    if arr is None:
        return None, None
    return arr.mean(axis=0), arr.std(axis=0)


def fmt(mean_val, std_val, prec=2) -> str:
    if mean_val is None:
        return "--"
    # Use scientific notation when both mean and std are very small
    if abs(mean_val) < 0.01 and abs(std_val) < 0.01:
        return f"{mean_val:.2e} $\\pm$ {std_val:.2e}"
    return f"{mean_val:.{prec}f} $\\pm$ {std_val:.{prec}f}"

# ============================================================
# Table generation
# ============================================================

COLUMN_NAMES = ["cosine final", "cosine all", "vision norm", "velocity norm", "displacement norm"]
COL_HEADER   = " & ".join([r"\textbf{Dataset}", r"\textbf{cosine final}", r"\textbf{cosine all}",
                            r"\textbf{vision norm}", r"\textbf{velocity norm}", r"\textbf{displacement norm}"])


def _escape(s: str) -> str:
    return s.replace("_", r"\_")


def build_table(
    datasets: list,
    all_data: dict,
    step_idx: int | None,
    title: str,
) -> str:
    """
    Build a LaTeX table.

    step_idx: int -> use that diffusion step (0-indexed); None -> average over all 10 steps.
    """
    rows = []
    for ds in datasets:
        d = all_data[ds]

        def get_val(arr, step):
            if arr is None:
                return None, None
            if step is not None:
                col = arr[:, step]         # (T,)
            else:
                col = arr.mean(axis=1)     # (T,) - mean over steps per frame, then stats
            return col.mean(), col.std()

        cosine_final_m, cosine_final_s = get_val(
            d['cosine']['final_layer'] if d['cosine'] else None, step_idx)
        cosine_all_m, cosine_all_s = get_val(
            d['cosine']['all_layers'] if d['cosine'] else None, step_idx)
        gradnorm_m, gradnorm_s = get_val(d['gradnorm'], step_idx)
        gradient_m, gradient_s = get_val(d['gradient'], step_idx)
        displace_m, displace_s = get_val(d['displacement'], step_idx)

        row = " & ".join([
            _escape(ds),
            fmt(cosine_final_m, cosine_final_s),
            fmt(cosine_all_m, cosine_all_s),
            fmt(gradnorm_m, gradnorm_s),
            fmt(gradient_m, gradient_s),
            fmt(displace_m, displace_s),
        ]) + r" \\"
        rows.append(row)

    body = "\n".join(rows)
    lines = [
        f"% {title}",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + title + "}",
        r"\begin{tabular}{llllll}",
        r"\hline",
        COL_HEADER + r" \\",
        r"\hline",
        body,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def write_tables(datasets: list, all_data: dict, out_path: str, set_name: str):
    tables = []
    tables.append(build_table(
        datasets, all_data, step_idx=0,
        title=f"{set_name}: Metrics at First Diffusion Step"))
    tables.append(build_table(
        datasets, all_data, step_idx=None,
        title=f"{set_name}: Metrics Averaged Over All 10 Diffusion Steps"))

    with open(out_path, "w") as fh:
        fh.write("\n\n".join(tables) + "\n")
    print(f"  Tables written to {out_path}")

# ============================================================
# Plot generation
# ============================================================

DIFF_STEPS = np.arange(1, NUM_DIFF_STEPS + 1)   # x ticks 1..10
LAYER_IDS  = np.arange(NUM_LAYERS)               # 0..17
COLORS     = plt.cm.tab10.colors


def _plot_metric(
    datasets: list,
    all_data: dict,
    key: str,
    subkey: str | None,
    ylabel: str,
    title: str,
    out_path: str,
):
    """
    Plot metric vs diffusion step for all datasets.
    key: top-level key in data dict ('cosine', 'gradnorm', 'gradient', 'displacement')
    subkey: sub-key if key='cosine' ('final_layer' or 'all_layers'); None otherwise.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, ds in enumerate(datasets):
        d = all_data[ds]
        arr = d.get(key)
        if arr is None:
            continue
        if subkey is not None:
            arr = arr[subkey]   # (T, 10)
        mean = arr.mean(axis=0)  # (10,)
        std  = arr.std(axis=0)
        ax.plot(DIFF_STEPS, mean, color=COLORS[i % 10], label=ds)
        ax.fill_between(DIFF_STEPS, mean - std, mean + std,
                        color=COLORS[i % 10], alpha=0.2)

    ax.set_xlabel("Diffusion Step")
    ax.set_xticks(DIFF_STEPS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def _cosine_per_layer_plot(
    datasets: list,
    all_data: dict,
    diff_step_idx: int,
    title: str,
    out_path: str,
) -> plt.Figure:
    """
    Plot cosine vs layer id for a specific diffusion step.
    Returns the figure (for GIF assembly).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, ds in enumerate(datasets):
        d = all_data[ds]
        if d['cosine'] is None:
            continue
        per_layer = d['cosine']['per_layer']  # (T, 10, 18)
        arr = per_layer[:, diff_step_idx, :]  # (T, 18)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        ax.plot(LAYER_IDS, mean, color=COLORS[i % 10], label=ds)
        ax.fill_between(LAYER_IDS, mean - std, mean + std,
                        color=COLORS[i % 10], alpha=0.2)

    ax.set_xlabel("Layer ID")
    ax.set_xticks(LAYER_IDS)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(title)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")
    return fig


def make_cosine_gif(frame_paths: list, gif_path: str, interval_ms: int = GIF_INTERVAL_MS):
    """Stitch per-step PNG files into a GIF using PIL."""
    try:
        from PIL import Image
        imgs = [Image.open(p) for p in frame_paths]
        imgs[0].save(
            gif_path,
            save_all=True,
            append_images=imgs[1:],
            loop=0,
            duration=interval_ms,
        )
        print(f"  GIF saved: {gif_path}")
    except Exception as e:
        print(f"  WARNING: could not create GIF: {e}", file=sys.stderr)


def generate_plots(datasets: list, all_data: dict, out_dir: str, set_name: str):
    os.makedirs(out_dir, exist_ok=True)

    # --- 5 main plots ---
    _plot_metric(datasets, all_data, 'cosine', 'final_layer',
                 "Cosine Similarity (Final Layer)",
                 f"{set_name}: Cosine Final Layer vs Diffusion Step",
                 os.path.join(out_dir, "cosine_final_layer.png"))

    _plot_metric(datasets, all_data, 'cosine', 'all_layers',
                 "Cosine Similarity (All Layers)",
                 f"{set_name}: Cosine All Layers vs Diffusion Step",
                 os.path.join(out_dir, "cosine_all_layers.png"))

    _plot_metric(datasets, all_data, 'gradnorm', None,
                 "Vision Perturbance Norm",
                 f"{set_name}: Vision Norm vs Diffusion Step",
                 os.path.join(out_dir, "vision_norm.png"))

    _plot_metric(datasets, all_data, 'gradient', None,
                 "Gradient Guidance Vector Norm",
                 f"{set_name}: Velocity Norm vs Diffusion Step",
                 os.path.join(out_dir, "velocity_norm.png"))

    _plot_metric(datasets, all_data, 'displacement', None,
                 "Action Displacement Norm",
                 f"{set_name}: Displacement Norm vs Diffusion Step",
                 os.path.join(out_dir, "displacement_norm.png"))

    # --- cosine per-layer plots + GIF ---
    frame_paths = []
    for k in range(NUM_DIFF_STEPS):
        step_label = k + 1
        title      = f"{set_name}: Cosine per Layer — Diffusion Step {step_label}"
        path       = os.path.join(out_dir, f"cosine_per_layer_step_{step_label:02d}.png")
        _cosine_per_layer_plot(datasets, all_data, diff_step_idx=k, title=title, out_path=path)
        frame_paths.append(path)

    gif_path = os.path.join(out_dir, "cosine_per_layer.gif")
    make_cosine_gif(frame_paths, gif_path)

# ============================================================
# ctraining-specific table and plot functions
# ============================================================

COL_HEADER_CT = " & ".join([
    r"\textbf{Dataset}",
    r"\textbf{cosine final}",
    r"\textbf{cosine all}",
])


def build_table_ctraining(datasets: list, all_data: dict, title: str) -> str:
    """LaTeX table for ctraining cosine (single step, no step_idx needed)."""
    rows = []
    for ds in datasets:
        d = all_data[ds]
        cosine = d.get('cosine')
        if cosine is not None:
            final_m, final_s = cosine['final_layer'].mean(), cosine['final_layer'].std()
            all_m,   all_s   = cosine['all_layers'].mean(),  cosine['all_layers'].std()
        else:
            final_m = final_s = all_m = all_s = None
        row = " & ".join([
            _escape(ds),
            fmt(final_m, final_s),
            fmt(all_m, all_s),
        ]) + r" \\"
        rows.append(row)

    body = "\n".join(rows)
    lines = [
        f"% {title}",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + title + "}",
        r"\begin{tabular}{lll}",
        r"\hline",
        COL_HEADER_CT + r" \\",
        r"\hline",
        body,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def write_tables_ctraining(datasets: list, all_data: dict, out_path: str, set_name: str):
    table = build_table_ctraining(
        datasets, all_data,
        title=f"{set_name} (ctraining): Cosine Similarity")
    with open(out_path, "w") as fh:
        fh.write(table + "\n")
    print(f"  Tables written to {out_path}")


def _plot_cosine_bar(datasets: list, all_data: dict, subkey: str, ylabel: str,
                     title: str, out_path: str):
    """Bar plot: one bar per dataset, y = ctraining cosine (mean ± std)."""
    means, stds, labels = [], [], []
    for ds in datasets:
        cosine = all_data[ds].get('cosine')
        if cosine is None:
            continue
        arr = cosine[subkey]   # (T,)
        means.append(arr.mean())
        stds.append(arr.std())
        labels.append(ds)

    if not means:
        return

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=[COLORS[i % 10] for i in range(len(labels))],
                  alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_escape(l).replace(r"\_", "\n") for l in labels],
                       fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def _cosine_per_layer_plot_ctraining(datasets: list, all_data: dict,
                                     title: str, out_path: str):
    """Cosine vs layer id for ctraining (single step — no step index needed)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, ds in enumerate(datasets):
        cosine = all_data[ds].get('cosine')
        if cosine is None:
            continue
        per_layer = cosine['per_layer']   # (T, 18)
        mean = per_layer.mean(axis=0)
        std  = per_layer.std(axis=0)
        ax.plot(LAYER_IDS, mean, color=COLORS[i % 10], label=ds)
        ax.fill_between(LAYER_IDS, mean - std, mean + std,
                        color=COLORS[i % 10], alpha=0.2)

    ax.set_xlabel("Layer ID")
    ax.set_xticks(LAYER_IDS)
    ax.set_ylabel("Cosine Similarity (ctraining)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def generate_plots_ctraining(datasets: list, all_data: dict, out_dir: str, set_name: str):
    os.makedirs(out_dir, exist_ok=True)

    _plot_cosine_bar(
        datasets, all_data, 'final_layer',
        "Cosine Similarity (Final Layer, ctraining)",
        f"{set_name} (ctraining): Cosine Final Layer",
        os.path.join(out_dir, "cosine_final_layer.png"))

    _plot_cosine_bar(
        datasets, all_data, 'all_layers',
        "Cosine Similarity (All Layers, ctraining)",
        f"{set_name} (ctraining): Cosine All Layers",
        os.path.join(out_dir, "cosine_all_layers.png"))

    _cosine_per_layer_plot_ctraining(
        datasets, all_data,
        title=f"{set_name} (ctraining): Cosine per Layer",
        out_path=os.path.join(out_dir, "cosine_per_layer.png"))


def run_set_ctraining(set_name: str, datasets: list, collect_fn, out_subdir: str):
    print(f"\n{'='*60}")
    print(f"  Processing ctraining: {set_name}  ({len(datasets)} datasets)")
    print(f"{'='*60}")

    out_dir = os.path.join(OUTPUT_ROOT, out_subdir, "ctraining")
    os.makedirs(out_dir, exist_ok=True)

    all_data = {}
    for ds in datasets:
        print(f"  Loading {ds} ...")
        all_data[ds] = collect_fn(ds)

    write_tables_ctraining(datasets, all_data,
                           os.path.join(out_dir, "ctraining.txt"),
                           set_name)
    generate_plots_ctraining(datasets, all_data, out_dir, set_name)


# ============================================================
# Main
# ============================================================

def run_set(set_name: str, datasets: list, collect_fn, out_subdir: str):
    print(f"\n{'='*60}")
    print(f"  Processing: {set_name}  ({len(datasets)} datasets)")
    print(f"{'='*60}")

    out_dir  = os.path.join(OUTPUT_ROOT, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    all_data = {}
    for ds in datasets:
        print(f"  Loading {ds} ...")
        all_data[ds] = collect_fn(ds)

    # Tables
    write_tables(datasets, all_data,
                 os.path.join(out_dir, f"{out_subdir}.txt"),
                 set_name)

    # Plots
    generate_plots(datasets, all_data, out_dir, set_name)


def main():
    warnings.simplefilter("always", RuntimeWarning)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    run_set("Franka Full",  FRANKA_DATASETS,  collect_franka,  "franka")
    run_set("OpenArm Full", OPENARM_DATASETS, collect_openarm, "openarm")
    run_set("OOD Full",     OOD_DATASETS,     collect_ood,     "ood")

    run_set_ctraining("Franka Full",  FRANKA_DATASETS,  collect_franka_ctraining,  "franka")
    run_set_ctraining("OpenArm Full", OPENARM_DATASETS, collect_openarm_ctraining, "openarm")
    run_set_ctraining("OOD Full",     OOD_DATASETS,     collect_ood_ctraining,     "ood")

    print("\nDone.")


if __name__ == "__main__":
    main()
