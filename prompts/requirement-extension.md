See prompts/completed.md for context.

I want you to create a new perturbance mode: D. Keep the old code unchanged and add new function blocks to support this functionality. Name the new mode 'perturbance-noise`. Details are specified below:

## Formulation: Perturbance Noise
Mostly follow the same prompts/perturbance-calculation-formalization.md, but two differences (1) the model should calculate all 10 inference steps with pure noise (condition-inference) instead of 1 step with randomly sampled time and noise deducted by ground truth action, (2) the model should calculate only the first gradient (approach 1) and stop the gradient descent steps (approach 2) Therefore, the refined formulation is as follows:


Let the vision-language-action model be denoted by

$$
\hat{A}^0 = \Phi_\theta(A_t^\tau, h_v, h_\ell, h_s, h_t),
$$

The noisy action latent $A_t^\tau$ starts from noise and follows the update convention 
$$
A_t^{\tau-\Delta} = A_t^\tau - \Delta \,\mathbf{v} \left(A_t^\tau, o_t, \tau\right)
$$
Each step recomputes a one-step surrogate of the final sample. Think of it as computing 0.9->0, 0.8->0, 0.7->0. At inference step k, $A_t^\tau$ is the current rollout state carried from step k-1 (initialized only once at $A_t^1=\epsilon$).



### D. `--metric="perturbance-noise"`: (you need to implement this)
only condition-inference should be calculated. neglect condition-training for this mode. 

Set the default step number to 0 (so that only N=0 is calculated; step_size should not be essential for calculation at this point). There should be two toggles `static_inference.py .. --perturbance_step_num=N, --perturbance_step_size=1e-2` for this mode (should be ignored for other modes).

There should be another toggle `static_inference.py .. --embedding_type=list(str)`, which is default to `["vision"]` but could take any sublist of `["vision", "state", "time", "language"]`. This should decide the perturbation on which embedding(s) are calculated. Disregard this flag for other modes. For all specified embedding types, respective results should be saved as different files in the same folder following the storage convention.


### Loss Parity Requirement (Strict)

For the perturbance and perturbance-noise mode, your implementation should match the original torch/jax PI0 loss semantics exactly:

- Reference implementation: `src/openpi/models/pi0.py::compute_loss`
- Loss definition to match: `mean((v_t - u_t)^2, axis=-1)` where `u_t = noise - actions`
- Time/noise construction to match exactly:
- `tau ~ Beta(1.5, 1.0) * 0.999 + 0.001`
- `x_t = tau * noise + (1 - tau) * actions`

Important clarification:
- JAX and Torch are functionally equivalent in this repo for pi0.5 training objective.
- JAX returns `[B, H]` then training takes global mean.
- Torch forward may return unreduced `[B, H, D]`, but training takes global mean.
- Therefore implementations in this milestone must preserve this functional equivalence and must not introduce a different objective.

Implementation rule for new static/perturbance code:
- Do not invent a new loss.


## Milestones

You should complete all milestones strictly one by one and wait for the user to verify each before proceeding to the next.
- Read all instructions and code. Read also the files that the instructions refer to. Read the codebase and understand the current static inference pipeline.
- Implement mode `--metric="perturbance-noise"`. 
- Modify corresponding sbatch scripts and launcher. (Don't delete any lines, just comment them out, and make the script only launch perturbance-noise for all datasets) Change the test to launch perturbance-noise, then wait for the user to run the test, then inspect the output and verify it looks correct (sanity checking size and tensor shape is sufficient).