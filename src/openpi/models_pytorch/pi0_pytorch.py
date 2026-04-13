import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        self._activation_recorder = None

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _, _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        self._record_activation("extra:diffusion_noise", -1, noise)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        self._record_activation("extra:predicted_action_chunk", -1, x_t)
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        *,
        output_hidden_states: bool = False,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _, suffix_hidden_states = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            output_hidden_states=output_hidden_states,
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        if output_hidden_states:
            return v_t, suffix_hidden_states, adarms_cond
        return v_t

    def register_activation_recorder(self, callback):
        self._activation_recorder = callback

        def action_hook(layer_idx, tensor, cond):
            if self._activation_recorder is None or tensor is None:
                return
            chunk = tensor[:, -self.config.action_horizon :, :]
            normed, _ = self.paligemma_with_expert.gemma_expert.model.norm(chunk, cond=cond)
            vt_layer = self.action_out_proj(normed.to(dtype=torch.float32))
            self._record_activation("action_expert_vt", layer_idx, vt_layer)

        if callback is None:
            self.paligemma_with_expert.register_action_expert_hook(None)
        else:
            self.paligemma_with_expert.register_action_expert_hook(action_hook)

    def _record_activation(self, branch, layer_idx, tensor):
        if self._activation_recorder is None or tensor is None:
            return
        try:
            self._activation_recorder(branch, layer_idx, tensor)
        except Exception:
            pass

    @torch.no_grad()
    def _prepare_static_prefix_context(self, observation):
        """Build reusable prefix cache for static-only analysis routines."""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return state, prefix_pad_masks, past_key_values

    def _compute_static_guidance_raw(self, x_t, tau, v_t, actions):
        """Compute unscaled guidance VJP term:

        (A* - A_hat0)^T * d(A_hat0)/d(A_t^tau), where A_hat0 = A_t^tau - tau * v(A_t^tau, o, tau).
        """
        tau_expanded = tau[:, None, None].to(dtype=torch.float32, device=x_t.device)
        a_hat0 = x_t - tau_expanded * v_t
        residual = (actions - a_hat0).detach()
        guidance_raw = torch.autograd.grad(
            outputs=a_hat0,
            inputs=x_t,
            grad_outputs=residual,
            retain_graph=False,
            create_graph=False,
        )[0]
        return guidance_raw, a_hat0

    def compute_static_gradient_guidance_training(self, observation, actions, *, noise=None, time=None):
        """Static-only gradient guidance under training-time condition."""
        state, prefix_pad_masks, past_key_values = self._prepare_static_prefix_context(observation)
        device = state.device
        actions = actions.to(device=device, dtype=torch.float32)
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        noise = noise.to(device=device, dtype=torch.float32)
        if time is None:
            time = self.sample_time(actions.shape[0], device)
        time = time.to(device=device, dtype=torch.float32)

        time_expanded = time[:, None, None]
        x_t = (time_expanded * noise + (1 - time_expanded) * actions).detach().requires_grad_(True)
        v_t = self.denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            time,
        )
        guidance_raw, _ = self._compute_static_guidance_raw(x_t, time, v_t, actions)
        target = noise - actions
        final_loss = torch.linalg.norm(v_t.detach() - target, dim=-1)
        return {
            "gradient_steps": [guidance_raw.detach()],
            "tau_steps": [time.detach()],
            "final_layer_loss": final_loss.detach().unsqueeze(1),
        }

    def compute_static_gradient_guidance_inference(self, observation, actions, *, noise=None, num_steps=10):
        """Static-only gradient guidance under inference-time condition."""
        state, prefix_pad_masks, past_key_values = self._prepare_static_prefix_context(observation)
        device = state.device
        actions = actions.to(device=device, dtype=torch.float32)
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        noise = noise.to(device=device, dtype=torch.float32)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        target = noise - actions
        x_t = noise.detach()
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        gradient_steps: list[torch.Tensor] = []
        tau_steps: list[torch.Tensor] = []
        final_losses: list[torch.Tensor] = []
        for _ in range(num_steps):
            tau = time.expand(actions.shape[0])
            x_t_step = x_t.detach().requires_grad_(True)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t_step,
                tau,
            )
            guidance_raw, _ = self._compute_static_guidance_raw(x_t_step, tau, v_t, actions)
            gradient_steps.append(guidance_raw.detach())
            tau_steps.append(tau.detach())
            final_losses.append(torch.linalg.norm(v_t.detach() - target, dim=-1))

            x_t = (x_t_step + dt * v_t).detach()
            time = time + dt

        return {
            "gradient_steps": gradient_steps,
            "tau_steps": tau_steps,
            "final_layer_loss": torch.stack(final_losses, dim=1),
        }

    def _compute_static_vt_layers(self, state, prefix_pad_masks, past_key_values, x_t, tau):
        """Run one static denoise step and return final/layer-wise projected velocities."""
        final_vt, suffix_hidden_states, adarms_cond = self.denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            tau,
            output_hidden_states=True,
        )
        if suffix_hidden_states is None:
            raise RuntimeError("Action expert hidden states were not returned; cannot compute vt layers.")

        vt_layers: dict[int, torch.Tensor] = {}
        # Hidden states tuple includes the embedding output at index 0.
        for layer_idx, hidden in enumerate(suffix_hidden_states[1:]):
            chunk = hidden[:, -self.config.action_horizon :, :]
            normed, _ = self.paligemma_with_expert.gemma_expert.model.norm(chunk, cond=adarms_cond)
            vt_layer = self.action_out_proj(normed.to(dtype=torch.float32))
            vt_layers[layer_idx] = vt_layer
        return final_vt, dict(sorted(vt_layers.items()))

    @torch.no_grad()
    def compute_static_cosine_targets(self, observation, actions, *, noise=None, time=None, num_steps=10):
        """Static-only cosine targets for both conditions with shared noise."""
        state, prefix_pad_masks, past_key_values = self._prepare_static_prefix_context(observation)
        device = state.device
        actions = actions.to(device=device, dtype=torch.float32)
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        noise = noise.to(device=device, dtype=torch.float32)
        if time is None:
            time = self.sample_time(actions.shape[0], device)
        time = time.to(device=device, dtype=torch.float32)

        target = noise - actions

        # condition-training: x_t = tau * noise + (1 - tau) * actions, one denoise step.
        time_expanded = time[:, None, None]
        x_t_training = time_expanded * noise + (1 - time_expanded) * actions
        ctraining_final, ctraining_layers = self._compute_static_vt_layers(
            state, prefix_pad_masks, past_key_values, x_t_training, time
        )

        # condition-inference: start from pure noise and run fixed Euler rollout.
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t_inference = noise
        time_step = torch.tensor(1.0, dtype=torch.float32, device=device)
        cinference_layer_steps: dict[int, list[torch.Tensor]] = {}
        cinference_final_steps: list[torch.Tensor] = []
        for _ in range(num_steps):
            tau = time_step.expand(actions.shape[0])
            final_vt, vt_layers = self._compute_static_vt_layers(
                state, prefix_pad_masks, past_key_values, x_t_inference, tau
            )
            cinference_final_steps.append(final_vt)
            for layer_idx, vt in vt_layers.items():
                cinference_layer_steps.setdefault(layer_idx, []).append(vt)
            x_t_inference = (x_t_inference + dt * final_vt).detach()
            time_step = time_step + dt

        cinference_layers = {
            layer_idx: torch.stack(step_values, dim=1) for layer_idx, step_values in cinference_layer_steps.items()
        }
        cinference_final = torch.stack(cinference_final_steps, dim=1)

        return {
            "target": target,
            "ctraining_vt_layers": ctraining_layers,
            "cinference_vt_layers": dict(sorted(cinference_layers.items())),
            "ctraining_final_prediction": ctraining_final,
            "cinference_final_prediction": cinference_final,
        }

    def _build_static_prefix_with_slices(self, images, img_masks, lang_tokens, lang_masks):
        """Build prefix embeddings plus token ranges for vision/language perturbations."""
        embs = []
        pad_masks = []
        att_masks = []

        num_vision_tokens = 0
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs
            num_vision_tokens += num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_tokens = lang_emb.shape[1]
        att_masks += [0] * num_lang_tokens

        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        att_tensor = torch.tensor(att_masks, dtype=torch.bool, device=prefix_embs.device)
        prefix_att_masks = att_tensor[None, :].expand(prefix_pad_masks.shape[0], len(att_masks))

        return {
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_att_masks": prefix_att_masks,
            "vision_slice": slice(0, num_vision_tokens),
            "language_slice": slice(num_vision_tokens, num_vision_tokens + num_lang_tokens),
        }

    def _build_static_suffix_with_slices(self, state, x_t, time):
        """Build suffix embeddings plus token ranges for action/state/time perturbations."""
        embs = []
        pad_masks = []
        att_masks = []
        state_slice = None

        if not self.pi05:
            state_emb = self.state_proj(state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device
            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)
            att_masks += [1]
            state_slice = slice(0, 1)

        time_emb = create_sinusoidal_pos_embedding(
            time, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=time.device
        )
        time_emb = time_emb.type(dtype=time.dtype)
        action_emb = self.action_in_proj(x_t)

        if not self.pi05:
            time_tokens = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_tokens], dim=2)
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            action_time_emb = self.action_time_mlp_out(x)
            adarms_cond = None
            time_cond = None
        else:
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            adarms_cond = F.silu(x)
            action_time_emb = action_emb
            time_cond = adarms_cond

        embs.append(action_time_emb)
        bsize, action_len = action_time_emb.shape[:2]
        action_mask = torch.ones(bsize, action_len, dtype=torch.bool, device=x_t.device)
        pad_masks.append(action_mask)
        att_masks += [1] + ([0] * (action_len - 1))

        suffix_embs = torch.cat(embs, dim=1)
        suffix_pad_masks = torch.cat(pad_masks, dim=1)
        att_tensor = torch.tensor(att_masks, dtype=torch.bool, device=suffix_embs.device)
        suffix_att_masks = att_tensor[None, :].expand(bsize, len(att_masks))

        action_start = 1 if not self.pi05 else 0
        action_slice = slice(action_start, action_start + action_len)

        return {
            "suffix_embs": suffix_embs,
            "suffix_pad_masks": suffix_pad_masks,
            "suffix_att_masks": suffix_att_masks,
            "state_slice": state_slice,
            "action_slice": action_slice,
            "time_cond": time_cond,
            "adarms_cond": adarms_cond,
        }

    def _compute_static_pi0_loss_tensor(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        suffix_embs,
        suffix_pad_masks,
        suffix_att_masks,
        adarms_cond,
        u_t,
    ):
        """Compute PI0 static loss tensor with canonical reduction: mean over action dim only."""
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _, _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        loss_tensor = torch.mean(torch.square(v_t - u_t), dim=-1)
        return loss_tensor, v_t

    def _compute_static_perturbance_single_embedding(
        self,
        embedding_type,
        prefix_pack,
        suffix_pack,
        u_t,
        *,
        step_num,
        step_size,
    ):
        """Run perturbance optimization trajectory for one embedding type."""
        prefix_base = prefix_pack["prefix_embs"]
        suffix_base = suffix_pack["suffix_embs"]
        base_adarms = suffix_pack["adarms_cond"]
        delta_target = None
        is_noop_embedding = False

        if embedding_type == "vision":
            delta_target = prefix_base[:, prefix_pack["vision_slice"], :]
        elif embedding_type == "language":
            delta_target = prefix_base[:, prefix_pack["language_slice"], :]
        elif embedding_type == "action":
            delta_target = suffix_base[:, suffix_pack["action_slice"], :]
        elif embedding_type == "state":
            if suffix_pack["state_slice"] is None:
                is_noop_embedding = True
                delta_target = torch.zeros(
                    (u_t.shape[0], 1, suffix_base.shape[-1]),
                    dtype=suffix_base.dtype,
                    device=suffix_base.device,
                )
            else:
                delta_target = suffix_base[:, suffix_pack["state_slice"], :]
        elif embedding_type == "time":
            if suffix_pack["time_cond"] is None:
                is_noop_embedding = True
                delta_target = torch.zeros(
                    (u_t.shape[0], suffix_base.shape[-1]),
                    dtype=suffix_base.dtype,
                    device=suffix_base.device,
                )
            else:
                delta_target = suffix_pack["time_cond"]
        else:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")

        delta = torch.zeros_like(delta_target).detach().requires_grad_(True)
        gradnorm_steps: list[torch.Tensor] = []
        loss_steps: list[torch.Tensor] = []
        delta_steps: list[torch.Tensor] = []

        for step_idx in range(step_num + 1):
            prefix_embs = prefix_base
            suffix_embs = suffix_base
            adarms_cond = base_adarms

            if embedding_type == "vision":
                prefix_embs = prefix_base.clone()
                prefix_embs[:, prefix_pack["vision_slice"], :] = prefix_embs[:, prefix_pack["vision_slice"], :] + delta
            elif embedding_type == "language":
                prefix_embs = prefix_base.clone()
                prefix_embs[:, prefix_pack["language_slice"], :] = (
                    prefix_embs[:, prefix_pack["language_slice"], :] + delta
                )
            elif embedding_type == "action":
                suffix_embs = suffix_base.clone()
                suffix_embs[:, suffix_pack["action_slice"], :] = suffix_embs[:, suffix_pack["action_slice"], :] + delta
            elif embedding_type == "state":
                suffix_embs = suffix_base.clone()
                suffix_embs[:, suffix_pack["state_slice"], :] = suffix_embs[:, suffix_pack["state_slice"], :] + delta
            elif embedding_type == "time":
                adarms_cond = base_adarms + delta

            loss_tensor, _ = self._compute_static_pi0_loss_tensor(
                prefix_embs,
                prefix_pack["prefix_pad_masks"],
                prefix_pack["prefix_att_masks"],
                suffix_embs,
                suffix_pack["suffix_pad_masks"],
                suffix_pack["suffix_att_masks"],
                adarms_cond,
                u_t,
            )
            scalar_loss = torch.mean(loss_tensor)
            grad = torch.autograd.grad(
                scalar_loss,
                delta,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                grad = torch.zeros_like(delta)
            gradnorm = torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), dim=1)
            gradnorm_steps.append(gradnorm.detach())
            loss_steps.append(loss_tensor.detach())
            delta_steps.append(delta.detach())

            if step_idx < step_num:
                if is_noop_embedding:
                    delta = delta.detach().requires_grad_(True)
                else:
                    delta = (delta - step_size * grad).detach().requires_grad_(True)

        return {
            "gradnorm_steps": gradnorm_steps,
            "loss_steps": loss_steps,
            "delta_steps": delta_steps,
        }

    def compute_static_perturbance_targets(
        self,
        observation,
        actions,
        *,
        embedding_types,
        step_num=0,
        step_size=1e-2,
        noise=None,
        time=None,
    ):
        """Static-only perturbance analysis under condition-training."""
        if step_num < 0:
            raise ValueError(f"step_num must be >= 0, got {step_num}")
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )
        device = state.device
        actions = actions.to(device=device, dtype=torch.float32)
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        noise = noise.to(device=device, dtype=torch.float32)
        if time is None:
            time = self.sample_time(actions.shape[0], device)
        time = time.to(device=device, dtype=torch.float32)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_pack = self._build_static_prefix_with_slices(images, img_masks, lang_tokens, lang_masks)
        suffix_pack = self._build_static_suffix_with_slices(state, x_t, time)

        embedding_results = {}
        for embedding_type in embedding_types:
            embedding_results[embedding_type] = self._compute_static_perturbance_single_embedding(
                embedding_type,
                prefix_pack,
                suffix_pack,
                u_t,
                step_num=step_num,
                step_size=step_size,
            )

        return {
            "tau": time.detach(),
            "embeddings": embedding_results,
        }

    def _compute_static_perturbance_noise_single_embedding(
        self,
        embedding_type,
        prefix_pack,
        suffix_pack,
        u_t,
    ):
        """Compute only first-order perturbance signal (N=0) for one embedding."""
        prefix_base = prefix_pack["prefix_embs"]
        suffix_base = suffix_pack["suffix_embs"]
        base_adarms = suffix_pack["adarms_cond"]
        delta_target = None
        is_noop_embedding = False

        if embedding_type == "vision":
            delta_target = prefix_base[:, prefix_pack["vision_slice"], :]
        elif embedding_type == "language":
            delta_target = prefix_base[:, prefix_pack["language_slice"], :]
        elif embedding_type == "state":
            if suffix_pack["state_slice"] is None:
                is_noop_embedding = True
                delta_target = torch.zeros(
                    (u_t.shape[0], 1, suffix_base.shape[-1]),
                    dtype=suffix_base.dtype,
                    device=suffix_base.device,
                )
            else:
                delta_target = suffix_base[:, suffix_pack["state_slice"], :]
        elif embedding_type == "time":
            if suffix_pack["time_cond"] is None:
                is_noop_embedding = True
                delta_target = torch.zeros(
                    (u_t.shape[0], suffix_base.shape[-1]),
                    dtype=suffix_base.dtype,
                    device=suffix_base.device,
                )
            else:
                delta_target = suffix_pack["time_cond"]
        else:
            raise ValueError(f"Unsupported embedding_type for perturbance-noise: {embedding_type}")

        delta = torch.zeros_like(delta_target).detach().requires_grad_(True)
        prefix_embs = prefix_base
        suffix_embs = suffix_base
        adarms_cond = base_adarms

        if embedding_type == "vision":
            prefix_embs = prefix_base.clone()
            prefix_embs[:, prefix_pack["vision_slice"], :] = prefix_embs[:, prefix_pack["vision_slice"], :] + delta
        elif embedding_type == "language":
            prefix_embs = prefix_base.clone()
            prefix_embs[:, prefix_pack["language_slice"], :] = prefix_embs[:, prefix_pack["language_slice"], :] + delta
        elif embedding_type == "state":
            suffix_embs = suffix_base.clone()
            if suffix_pack["state_slice"] is not None:
                suffix_embs[:, suffix_pack["state_slice"], :] = suffix_embs[:, suffix_pack["state_slice"], :] + delta
        elif embedding_type == "time":
            if base_adarms is not None:
                adarms_cond = base_adarms + delta

        loss_tensor, _ = self._compute_static_pi0_loss_tensor(
            prefix_embs,
            prefix_pack["prefix_pad_masks"],
            prefix_pack["prefix_att_masks"],
            suffix_embs,
            suffix_pack["suffix_pad_masks"],
            suffix_pack["suffix_att_masks"],
            adarms_cond,
            u_t,
        )
        scalar_loss = torch.mean(loss_tensor)
        grad = torch.autograd.grad(
            scalar_loss,
            delta,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad is None or is_noop_embedding:
            grad = torch.zeros_like(delta)
        gradnorm = torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), dim=1)
        return {
            "gradnorm": gradnorm.detach(),
            "final_layer_loss": loss_tensor.detach(),
        }

    def compute_static_perturbance_noise_targets(
        self,
        observation,
        actions,
        *,
        embedding_types,
        num_steps=10,
        step_num=0,
        step_size=1e-2,
        save_displacement_trace=False,
        noise=None,
    ):
        """Static-only perturbance-noise analysis under condition-inference."""
        del step_num, step_size  # Reserved CLI toggles; approach-2 descent is intentionally disabled in this mode.
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}")

        state, prefix_pad_masks, past_key_values = self._prepare_static_prefix_context(observation)
        images, img_masks, lang_tokens, lang_masks, _ = self._preprocess_observation(observation, train=False)
        device = state.device
        actions = actions.to(device=device, dtype=torch.float32)
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        noise = noise.to(device=device, dtype=torch.float32)
        u_t = noise - actions

        prefix_pack = self._build_static_prefix_with_slices(images, img_masks, lang_tokens, lang_masks)
        x_t = noise.detach()
        x_t_initial = x_t
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        displacement_steps: list[torch.Tensor] = []
        displacement_norm_steps: list[torch.Tensor] = []

        embedding_results = {
            embedding_type: {
                "gradnorm_steps": [],
                "final_layer_loss_steps": [],
                "tau_steps": [],
            }
            for embedding_type in embedding_types
        }

        for _ in range(num_steps):
            tau = time.expand(actions.shape[0])
            suffix_pack = self._build_static_suffix_with_slices(state, x_t, tau)
            for embedding_type in embedding_types:
                output = self._compute_static_perturbance_noise_single_embedding(
                    embedding_type,
                    prefix_pack,
                    suffix_pack,
                    u_t,
                )
                embedding_results[embedding_type]["gradnorm_steps"].append(output["gradnorm"])
                embedding_results[embedding_type]["final_layer_loss_steps"].append(output["final_layer_loss"])
                embedding_results[embedding_type]["tau_steps"].append(tau.detach())

            with torch.no_grad():
                v_t = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    tau,
                )
            x_t = (x_t + dt * v_t).detach()
            if save_displacement_trace:
                displacement = (x_t - x_t_initial).detach()
                displacement_steps.append(displacement)
                displacement_norm = torch.linalg.vector_norm(displacement.reshape(displacement.shape[0], -1), dim=1)
                displacement_norm_steps.append(displacement_norm.detach())
            time = time + dt

        return {
            "embeddings": embedding_results,
            "displacement_steps": displacement_steps,
            "displacement_norm_steps": displacement_norm_steps,
        }

    @torch.no_grad()
    def compute_static_targets(self, observation, actions, *, noise=None, time=None):
        """Run a teacher-forced pass that returns vt tensors for every layer.

        This is used by the static evaluation script and is intentionally isolated
        from the regular inference/training code paths.
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )
        device = state.device
        actions = actions.to(device=device, dtype=torch.float32)
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(actions.shape[0], device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        self._record_activation("extra:diffusion_noise", -1, noise)

        expanded_time = time.expand(state.shape[0])
        final_vt, vt_layers = self._compute_static_vt_layers(
            state, prefix_pad_masks, past_key_values, x_t, expanded_time
        )

        return {
            "vt_layers": vt_layers,
            "target": u_t,
            "final_prediction": final_vt,
        }
