"""RoboCasa dataset-to-pi05 alignment at the dataset boundary."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any, Protocol

import numpy as np

from openpi.shared import normalize as _normalize

ACTION_DIM = 32
STATE_PERM = np.asarray([7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15])


@dataclasses.dataclass(frozen=True)
class RemapSpec:
    name: str
    # Pairs are (aligned destination slot, raw RoboCasa source slot).
    action_layout: tuple[tuple[int, int], ...]
    metric_dims: tuple[int, ...]
    # aligned_action[..., aligned_to_native_perm] is the native model action.
    aligned_to_native_perm: tuple[int, ...]
    norm_stats_swap: tuple[int, int] | None = None

    @property
    def metric_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        mask[list(self.metric_dims)] = 1.0
        return mask


def _swap_perm(a: int, b: int) -> tuple[int, ...]:
    perm = np.arange(ACTION_DIM)
    perm[a], perm[b] = perm[b], perm[a]
    return tuple(int(x) for x in perm)


def _base_spec(name: str, parked: int, *, base: bool, grip_control: bool) -> RemapSpec:
    layout = [(dst, src) for dst, src in enumerate((5, 6, 7, 8, 9, 10))]
    dims = list(range(6))
    if grip_control:
        layout.append((7, 11))
        dims.append(7)
    if base:
        layout.extend((8 + i, i) for i in range(3))
        dims.extend((8, 9, 10))
    if grip_control:
        layout.append((11, 4))
        dims.append(11)
    # There is no seventh arm coordinate in RoboCasa's Cartesian action.
    # Its aligned raw value remains zero at the required parked slot.
    dims.append(parked)
    return RemapSpec(
        name=name,
        action_layout=tuple(layout),
        metric_dims=tuple(sorted(dims)),
        aligned_to_native_perm=_swap_perm(6, parked),
        norm_stats_swap=(6, parked),
    )


SPECS = {
    "robocasa": RemapSpec(
        name="robocasa",
        action_layout=tuple(enumerate((5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4))),
        metric_dims=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
        aligned_to_native_perm=tuple(range(ACTION_DIM)),
    ),
    "base-arm": _base_spec("base-arm", 8, base=False, grip_control=False),
    "base-arm-base": _base_spec("base-arm-base", 11, base=True, grip_control=False),
    "base-arm-base-grip": _base_spec("base-arm-base-grip", 12, base=True, grip_control=True),
}


class IndexableDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict[str, Any]: ...


class RemappedRobocasaDataset:
    """View a raw RoboCasa dataset in the selected pi05-aligned layout."""

    def __init__(self, dataset: IndexableDataset, spec: RemapSpec):
        self.dataset = dataset
        self.spec = spec

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.dataset[index]
        state = np.asarray(raw["observation/state"])
        actions = np.asarray(raw["actions"])
        if state.shape[-1] != 16:
            raise ValueError(f"Expected a 16-D RoboCasa state, got shape={state.shape}")
        if actions.shape[-1] != 12:
            raise ValueError(f"Expected 12-D RoboCasa actions, got shape={actions.shape}")

        aligned = np.zeros((*actions.shape[:-1], ACTION_DIM), dtype=actions.dtype)
        for dst, src in self.spec.action_layout:
            aligned[..., dst] = actions[..., src]

        sample = dict(raw)
        sample["observation/state"] = state[..., STATE_PERM].copy()
        sample["actions"] = aligned
        return sample


def remap_action_norm_stats(
    norm_stats: dict[str, _normalize.NormStats], spec: RemapSpec
) -> dict[str, _normalize.NormStats]:
    """Return checkpoint-original stats viewed in the aligned action layout."""
    if spec.norm_stats_swap is None or "actions" not in norm_stats:
        return norm_stats
    a, b = spec.norm_stats_swap
    stats = norm_stats["actions"]

    def swap(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        result = np.asarray(values).copy()
        result[..., [a, b]] = result[..., [b, a]]
        return result

    result = dict(norm_stats)
    result["actions"] = _normalize.NormStats(
        mean=swap(stats.mean),
        std=swap(stats.std),
        q01=swap(stats.q01),
        q99=swap(stats.q99),
    )
    return result


def compose(transforms: Sequence):
    def apply(data: dict[str, Any]) -> dict[str, Any]:
        for transform in transforms:
            data = transform(data)
        return data

    return apply
