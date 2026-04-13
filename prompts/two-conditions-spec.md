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
- ctraining_final_layer_loss
- cinference_final_layer_loss
- ctraining-cosine_{layer_idx}
- cinference-cosine_{layer_idx}

The content inside /meta are decided by the flag `--save_meta=True` passed in to the static inference script

**gradient guidance vector**

The latents that should be saved as files are (use these as file names followed by .npy as well):
- gradient_step_{diffusion_step_idx} for condition-training
- gradient_step_{diffusion_step_idx} as well for condition-inference

`--save_meta=True/False` should not affect the behavior here.

**perturbance**

Note that here step_idx refers to the perturbation step defined in approach 2 inside `prompts/perturbance-calculation-formalization.md`, and is fundamentally different from the diffusion_step_idx in the gradient mode. The latents that should be saved as files are (use these as file names followed by .npy as well):

Take vision embedding as the example here, which could be swapped with action / language / time / state embedding as defined in the formulation file. For those, just replace "vision" with action / language / time / state. 

- gradnorm_vision_step_{step_idx}  [this should be the local sensitivity score $||\nabla_{h_v}L^{(n)}||_2$, where n is the step idx]
- perturb_loss_vision_step_{step_idx}  [this should be the full loss trajectory $L^{(0)}$,...,$L^{(N)}$ from N perturbation steps on the vision embedding, where $L^{(0)}$ is the loss before perturbation with $\delta_v=0$]
- meta/perturb_delta_vision_step_{step_idx}  [this should be the full perturbation vector trajectory $\delta_v^{(0)}$,...,$\delta_v^{(N)}$ from N perturbation steps on the vision embedding, where $\delta_v^{(0)}$=0]

The content inside meta/ are decided by the flag `--save_meta=True` passed in to the static inference script

**perturbance-noise**


Note that here diffusion_step_idx refers to the step in condition-inference, default to the pi model default number. (The user believe this number is 10, cross check with the code, if you discover a discrepancy anytime, pause immediately and raise the issue to the user.)

It is **not** the perturbation step defined in approach 2 inside `prompts/perturbance-calculation-formalization.md` (discarded for this mode). The latents that should be saved as files are (use these as file names followed by .npy as well):

Take vision embedding as the example here, which could be swapped with language / time / state embedding as defined in the formulation file. For those, just replace "vision" with language / time / state. 

- gradnorm_vision_step_{diffusion_step_idx}  [this should be the local sensitivity score $||\nabla_{h_v}L^{(n)}||_2$, where n is the step idx]
- final_layer_loss_vision_step_{diffusion_step_idx}  [this should be the full loss trajectory across diffusion steps

The content inside meta/ are decided by the flag `--save_meta=True` passed in to the static inference script


**perturbance-noise with `save_displacement_trace` flag on**

Except from saving all the perturbance-noise latents (those should always exist), also save the displacement vectors
- displacement_norm  [this should be the ten l2 norms of displacement vectors from the first norm(delta(x_0,x_1)) to the last norm(delta(x_0,x_10))] (the l2 norm should be a single value, so the stored shape should thus be shape [10])
- meta/displacement_step_{step_idx}  [this should be the full displacement vectors, measuring delta(x_0,x_1), delta(x_0,x_2), ... delta(x_0,x_10)]

The content inside meta/ are decided by the flag `--save_meta=True` passed in to the static inference script

Note: Displacement outputs must go to parallel folder perturbance-noise-displacement/, not inside perturbance-noise/.
You should create a separate folder in the result directory, i.e. folders perturbance-noise-displacement/ and perturbance-noise/ should be under a same mother directory.

