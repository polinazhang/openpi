import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    """Convert potentially float/CHW tensors to uint8 HWC arrays."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected 3-dimensional image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255 * image).astype(np.uint8)
    if image.shape[0] in (1, 3):
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OpenArmInputs(transforms.DataTransformFn):
    """Repackage OpenArm dataset samples into Pi-friendly observations."""

    def __call__(self, data: dict) -> dict:
        left = _parse_image(data["left_wrist_image"])
        right = _parse_image(data["right_wrist_image"])

        if "head_image" in data:
            base = _parse_image(data["head_image"])
            base_mask = np.True_
        else:
            base = np.zeros_like(left)
            base_mask = np.False_

        state = np.asarray(data["state"], dtype=np.float32)
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": base_mask,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        prompt = data.get("prompt", data.get("task"))
        if prompt is not None:
            if isinstance(prompt, (list, tuple)):
                prompt = prompt[0]
            if isinstance(prompt, np.ndarray):
                prompt = prompt.item()
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class OpenArmOutputs(transforms.DataTransformFn):
    """Extract the OpenArm action slice from Pi predictions."""

    action_dim: int = 16

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, : self.action_dim])
        return {"actions": actions}
