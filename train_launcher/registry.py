"""RoboCasa task-group resolution and dataset metadata builders for finetune launch."""

from __future__ import annotations


def resolve_group(group_name: str, expected_max_index: int) -> list[str]:
    """Resolve a canonical group to an ordered task list and validate expected size."""
    from robocasa.utils.dataset_registry import TARGET_TASKS, TASK_SET_REGISTRY

    eval_target_50 = (
        list(TARGET_TASKS["atomic_seen"])
        + list(TARGET_TASKS["composite_seen"])
        + list(TARGET_TASKS["composite_unseen"])
    )

    if group_name == "pretrain_all_300":
        tasks = list(TASK_SET_REGISTRY["pretrain300"])
    elif group_name == "pretrain_only_266":
        eval_set = set(eval_target_50)
        tasks = [task for task in TASK_SET_REGISTRY["pretrain300"] if task not in eval_set]
    elif group_name == "atomic_seen":
        tasks = list(TARGET_TASKS["atomic_seen"])
    elif group_name == "composite_seen":
        tasks = list(TARGET_TASKS["composite_seen"])
    elif group_name == "composite_unseen":
        tasks = list(TARGET_TASKS["composite_unseen"])
    elif group_name == "eval_target_50":
        tasks = eval_target_50
    else:
        raise ValueError(f"Unknown group_name: {group_name}")

    if len(tasks) != expected_max_index:
        raise ValueError(
            f"Resolved group '{group_name}' has size {len(tasks)} but expected max_index={expected_max_index}."
        )

    return tasks


def group_split_cap(group_name: str) -> int:
    """Return split-specific maximum demos per task."""
    if group_name in {"pretrain_all_300", "pretrain_only_266"}:
        return 100

    if group_name in {"atomic_seen", "composite_seen", "composite_unseen", "eval_target_50"}:
        return 500

    raise ValueError(f"Unknown group_name: {group_name}")


def build_meta(task_name: str, group_name: str, num_demos: int) -> dict:
    """Build one dataset metadata dict for a resolved task and demo count."""
    if not (1 <= num_demos <= 500):
        raise ValueError(f"num_demos must be in [1, 500], got {num_demos}.")

    split_cap = group_split_cap(group_name)
    if num_demos > split_cap:
        raise ValueError(
            f"num_demos={num_demos} exceeds split cap {split_cap} for group '{group_name}'."
        )

    from robocasa.utils.dataset_registry_utils import get_ds_meta

    split = "pretrain" if group_name in {"pretrain_all_300", "pretrain_only_266"} else "target"
    meta = get_ds_meta(task=task_name, split=split, source="human", demo_fraction=1.0)
    if meta is None:
        raise ValueError(
            f"Dataset metadata not available for task '{task_name}' with split '{split}'."
        )

    meta = dict(meta)
    meta["filter_key"] = f"{num_demos}_demos"
    return meta
