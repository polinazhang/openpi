In all notations, $\tau$ descends from 1 to 0.

condition-training refers to static inference under training-time conditions, where the model starts from x_t = tau*noise + (1-tau)*actions, with sampled tau, and runs one step.

condition-inference refers to static inference under inference-time conditions, where the model starts from x_t = noise, and runs num_steps matching the model inference settings.

This file explains the difference between them regarding implementation.

## condition-training
This condition was previously implemented in the openarm static inference pipeline. For each input:

Sample a random $\tau$ from the beta distribution: Beta(1.5, 1.0) * 0.999 + 0.001
  - x_t = t*noise + (1-t)*actions
  - u_t = noise-actions


prompts/documentation.md contains additional details about **only** the condition-training. If you need more details and that file is not yet in your context window, read it.


## condition-inference

This condition hasn't been implemented. You should set num_steps default to 10. For each input:

  1. Initialize x_1 = noise, t=1.
  2. Repeatedly compute v_t = v(x_t, o, t).
  3. Update x_{t+dt} = x_t + dt * v_t, with dt = -1/num_steps.
  4. Decrease t accordingly.

## Specs
The calculated metric values, cosine_{layer_idx}, u, v, gradient_guidance_vector, etc, should be per inference step. Therefore, there will be 1 of those latents for condition-training, and 10 of those latents for condition-inference. See prompts/new-metric-definition.md for calculation details.

## Storage

Depending on the running flag, stoagre should be structured differently. Cosine similarity and guidance vector will be run in different runs.

**cosine**

The latents that should be saved as files are (use these as file names followed by .npy as well):
- meta/u             (this should be the prediction target: u_t = noise - ground_truth_actions_from_demo)
- meta/ctraining-v_{layer_idx}   [this should be the layer-wise predicted latents by the model starting from x_t(or in another notation, $A_t^0$) = ground truth action]
- meta/cinference-v_{layer_idx}  [this should be the layer-wise predicted latents by the model starting from x_t(or in another notation, $A_t^0$) = noise]
- final_layer_loss
- ctraining-cosine_{layer_idx}
- cinference-cosine_{layer_idx}

The content inside /meta are decided by the flag `--save_meta=True` passed in to the static inference script

**gradient guidance vector**

The latents that should be saved as files are (use these as file names followed by .npy as well):
- gradient_step_{step_idx} for condition-training
- gradient_step_{step_idx} as well for condition-inference

`--save_meta=True/False` should not affect the behavior here.