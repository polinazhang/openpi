from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies.metadata_logger import MetadataLogger
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class _TorchActivationCapture:
    """Collects PyTorch tensors emitted by the model activation recorder."""

    def __init__(self) -> None:
        self._vt_layers: dict[str, torch.Tensor] = {}
        self._extras: dict[str, torch.Tensor] = {}
        self._active = False

    def begin_step(self) -> None:
        self._vt_layers.clear()
        self._extras.clear()
        self._active = True

    def cancel_step(self) -> None:
        self._vt_layers.clear()
        self._extras.clear()
        self._active = False

    def finish_step(self) -> dict[str, torch.Tensor]:
        data = {**self._vt_layers, **self._extras}
        self.cancel_step()
        return data

    def __call__(self, branch: str, layer_idx: int, tensor: torch.Tensor | None) -> None:
        if not self._active or tensor is None:
            return
        target: dict[str, torch.Tensor]
        name: str
        if branch == "action_expert_vt":
            target = self._vt_layers
            name = f"vt_layer_{layer_idx}"
        elif branch.startswith("extra:"):
            target = self._extras
            name = branch.split(":", 1)[1] or f"extra_{layer_idx}"
        else:
            return
        target[name] = tensor.detach().to(device="cpu").clone()


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        evaluation_suite_name: str,
        data_dir: str,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._metadata_logger: MetadataLogger | None = None
        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._activation_capture = _TorchActivationCapture()
            if hasattr(self._model, "register_activation_recorder"):
                self._model.register_activation_recorder(self._activation_capture)
            self._metadata_logger = MetadataLogger(data_dir, evaluation_suite_name)
        else:
            # JAX model setup (metadata recording unavailable for now).
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            logging.warning(
                "Metadata recording is currently only implemented for PyTorch checkpoints; "
                "JAX policies will run without logging vt/noise tensors."
            )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if self._is_pytorch_model:
            self._activation_capture.begin_step()
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)

            sample_kwargs = dict(self._sample_kwargs)
            if noise is not None:
                noise = torch.from_numpy(noise).to(self._pytorch_device)
                if noise.ndim == 2:
                    noise = noise[None, ...]
                sample_kwargs["noise"] = noise

            observation = _model.Observation.from_dict(inputs)
            start_time = time.monotonic()
            try:
                actions = self._sample_actions(self._pytorch_device, observation, **sample_kwargs)
            except Exception:
                self._activation_capture.cancel_step()
                raise
            model_time = time.monotonic() - start_time
            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)

            outputs = self._output_transform(outputs)
            outputs["policy_timing"] = {
                "infer_ms": model_time * 1000,
            }
            artifacts = self._activation_capture.finish_step()
            prepared = self._prepare_artifacts_for_logging(artifacts)
            if self._metadata_logger is not None:
                self._metadata_logger.record_step(prepared)
            return outputs

        # JAX inference path (no metadata recording).
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            jax_noise = jnp.asarray(noise)
            if jax_noise.ndim == 2:
                jax_noise = jax_noise[None, ...]
            sample_kwargs["noise"] = jax_noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def end_trajectory(self) -> None:
        if self._metadata_logger is not None:
            self._metadata_logger.end_trajectory()

    def _prepare_artifacts_for_logging(self, tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        expected = {"diffusion_noise", "predicted_action_chunk"}
        missing = [key for key in expected if key not in tensors]
        if missing:
            raise RuntimeError(f"Missing required metadata tensors: {missing}")
        arrays: dict[str, np.ndarray] = {}
        for name, tensor in tensors.items():
            target_name = "actions" if name == "predicted_action_chunk" else name
            arrays[target_name] = tensor.detach().cpu().float().numpy()
        return arrays


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    def end_trajectory(self) -> None:
        if hasattr(self._policy, "end_trajectory"):
            self._policy.end_trajectory()
