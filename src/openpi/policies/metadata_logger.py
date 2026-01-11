from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


class MetadataLogger:
    """Persists per-inference tensors and packages them into trajectory-level artifacts."""

    def __init__(self, data_dir: str | Path, evaluation_suite: str) -> None:
        root = Path(data_dir).expanduser()
        self._suite_dir = root / evaluation_suite
        self._suite_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self._suite_dir / "metadata.json"
        if self._metadata_path.exists():
            with self._metadata_path.open("r", encoding="utf-8") as handle:
                self._metadata_entries = json.load(handle)
        else:
            self._metadata_entries = []
        self._current_trajectory_id = len(self._metadata_entries)
        self._current_step_idx = 0

    def record_step(self, artifacts: Dict[str, np.ndarray]) -> None:
        """Persist tensors from a single inference call."""
        if not artifacts:
            return
        metadata_dir = self._metadata_dir()
        step_dir = metadata_dir / "steps" / f"{self._current_step_idx:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for name, array in artifacts.items():
            array_path = step_dir / f"{name}.npy"
            np.save(array_path, array.astype(np.float16))
        self._current_step_idx += 1

    def end_trajectory(self) -> None:
        """Combine recorded steps into trajectory-level artifacts and advance the counter."""
        steps_dir = self._steps_dir()
        if not steps_dir.exists() or not any(steps_dir.iterdir()):
            # No data for this trajectory; just advance the counter.
            self._cleanup_current_dirs(remove_steps=True)
            self._current_step_idx = 0
            self._current_trajectory_id += 1
            return

        artifact_names = self._discover_artifact_names(steps_dir)
        if not artifact_names:
            self._cleanup_current_dirs(remove_steps=True)
            self._current_step_idx = 0
            self._current_trajectory_id += 1
            return

        metadata_dir = self._metadata_dir()
        metadata_dir.mkdir(parents=True, exist_ok=True)
        rel_prefix = f"{self._current_trajectory_id:06d}/npy-metadata"
        artifact_entries: Dict[str, str] = {}
        step_paths = sorted(steps_dir.iterdir())

        for name in sorted(artifact_names):
            arrays = [np.load(step / f"{name}.npy") for step in step_paths]
            stacked = np.stack(arrays, axis=0)
            out_path = metadata_dir / f"{name}.npy"
            np.save(out_path, stacked.astype(np.float16))
            artifact_entries[name] = f"{rel_prefix}/{name}.npy"

        entry = {
            "trajectory_id": self._current_trajectory_id,
            "trajectory_rel_dir": rel_prefix,
            "num_steps": len(step_paths),
            "artifacts": artifact_entries,
        }
        self._metadata_entries.append(entry)
        with self._metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(self._metadata_entries, handle, indent=2)

        shutil.rmtree(steps_dir, ignore_errors=True)
        self._current_step_idx = 0
        self._current_trajectory_id += 1

    def _metadata_dir(self) -> Path:
        return self._suite_dir / f"{self._current_trajectory_id:06d}" / "npy-metadata"

    def _steps_dir(self) -> Path:
        return self._metadata_dir() / "steps"

    def _discover_artifact_names(self, steps_dir: Path) -> Iterable[str]:
        names: set[str] = set()
        for step in steps_dir.iterdir():
            for artifact_path in step.glob("*.npy"):
                names.add(artifact_path.stem)
        return names

    def _cleanup_current_dirs(self, *, remove_steps: bool) -> None:
        metadata_dir = self._metadata_dir()
        if remove_steps:
            shutil.rmtree(self._steps_dir(), ignore_errors=True)
        # Remove empty metadata dir if it only held steps.
        if metadata_dir.exists() and not any(metadata_dir.iterdir()):
            shutil.rmtree(metadata_dir, ignore_errors=True)
