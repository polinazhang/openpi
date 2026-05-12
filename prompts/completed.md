This file is an instruction specifying required functionalities in this repo, which have **already been completed** faithfully. It serves as a reference describing the modifications made to the codebase.

For context, read `prompts/static-inference-context.md` and all the files it refers to before reading this document.

## Required Code Functionality

You should extend the current flag `--metric=""|options:"cosine","gradient"` to support another option "perturbance". The "cosine" and "gradient" modes are already implemented and are provided for context, while "perturbance" is specified separately. For the context modes, **read only the referenced files as needed** rather than loading all files by default.

For all modes, there should be a skip frame functionality so not all the inputs are taken. Instead, for every x state-actions, one will be taken. All calculations would require to take a chunk of ground truth action instead of a single action from the sample trajectory. When there's not enough actions left in the trajectory, you should stop computing the metrics. Don't pad, just discard the tail.

### A. `--metric="cosine"`:

If cosine is selected, condition-training and condition-inference should be calculated simutanously in the same run. For inference on every input, the two conditions should use the same randomized noise initialization, only different should be one starts from $x_t = tau*noise + (1-tau)*actions$ and one starts from $x_t = noise$. Basically, given one datapoint (one noise and one input), two static inferences should be both be done: condition-training and condition-inference; then repeat on the entire trajectory; which should repeat across trajectories. 

In this mode, there should be a toggle `static_inference.py .. --save_meta=True` means that latents should be saved: u, ctraining-v, cinference-v, which is default to False. Details about these latents specified in the storage section in `prompts/two-conditions-spec.md`. 

### B. `--metric="gradient"`:
condition-training and condition-inference should be calculated in separate runs. There should be a toggle `static_inference.py .. --condition=training/inference` controlling that. The condition toggle should be disregarded if other modes are selected as metric. For how to calculate, details are included in `prompts/new-metric-definition.md`.

Disregard the `--save_meta=True` flag for this mode only.

### C. `--metric="perturbance"`:
only condition-training should be calculated. neglect condition-inference for this mode. 

Look at `prompts/perturbance-calculation-formalization.md` for formulation details. Set the default step number to 0 (so that only N=0 is calculated; step_size should not be essential for calculation at this point). There should be two toggles `static_inference.py .. --perturbance_step_num=N, --perturbance_step_size=1e-2` for this mode (should be ignored for other modes).

There should be another toggle `static_inference.py .. --embedding_type=list(str)`, which is default to `["vision", "action"]` but could take any sublist of `["vision", "action", "state", "time", "language"]`. This should decide the perturbation on which embedding(s) are calculated. Disregard this flag for other modes. For all specified embedding types, respective results should be saved as different files in the same folder following the storage convention.

Also in this mode, there should be a toggle `static_inference.py .. --save_meta=True` means that latents should be saved: perturb_delta_vision_step_{step_idx}, which is default to False. Details are specified in the storage section in `prompts/two-conditions-spec.md`. 

Save all the way to N instead of N-1.


### D. `--metric="perturbance-noise"`:
only condition-inference should be calculated. neglect condition-training for this mode. 

Set the default step number to 0 (so that only N=0 is calculated; step_size should not be essential for calculation at this point). There should be two toggles `static_inference.py .. --perturbance_step_num=N, --perturbance_step_size=1e-2` for this mode (should be ignored for other modes).

There should be another toggle `static_inference.py .. --embedding_type=list(str)`, which is default to `["vision"]` but could take any sublist of `["vision", "state", "time", "language"]`. This should decide the perturbation on which embedding(s) are calculated. Disregard this flag for other modes. For all specified embedding types, respective results should be saved as different files in the same folder following the storage convention.


### Perturbance Noise Formulation Details
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


### D. `--metric="perturbance-noise"` (Amendment 1: displacement vector):
I want you to add a toggle `static_inference.py .. --save_displacement_trace=True` that's default to True under mode D: perturbance-noise. Other modes should safely ignore this flag. 

If this flag is true, mode D should also calculate and store the norm of the displacement vector, and the latent of the displacement vector itself if `--save_meta=True`. These share the same 10 diffusion step share the same model passes with the standard mode D and are thus computationally efficient to be put together in one mode.

###  Displacement vector formulation details

Compute, at each denoising step (k), the displacement vector from the initial pure-noise latent to the current predicted-action latent, and record its trajectory over all 10 steps.

$$
  \Delta A^{(k)} := A_t^{\tau_k}-A_t^1
$$
with $A_t^\tau \equiv x_t$ and $A_t^1$ as the initial pure-noise latent.

(In other notations) Compute the Latent-space displacement $$\Delta x^{(k)} = x^{(k)} - x^{(0)}$$

### D. `--metric="perturbance-noise"` (Amendment 2: cosine-inference):
Add a toggle `static_inference.py .. --save_cosine=True` that's default to True under mode D: perturbance-noise. Other modes should safely ignore this flag. If this flag is true, mode D should also calculate and store the cosine like what mode A did, but only for condition-inference, since mode D is only supposed to run in the inference mode.

All storage/naming rules and mathematical formulations are identical to mode A except that only one condition-inference should run and its latent should store. Ignore condition-training. (mode A did both condition-training and condition-inference)

### D. Storage
The new rule you should implement is that, regardless of whether save_displacement_trace and save_cosine is True or False, save the results into perturbance-all/. 




## Loss Parity Requirement (Strict)

For the perturbance mode, your implementation should match the original torch/jax PI0 loss semantics exactly:

- Reference implementation: `src/openpi/models/pi0.py::compute_loss`
- Loss definition to match: `mean((v_t - u_t)^2, axis=-1)` where `u_t = noise - actions`
- Time/noise construction to match exactly:
- `tau ~ Beta(1.5, 1.0) * 0.999 + 0.001`
- `x_t = tau * noise + (1 - tau) * actions`

Important clarification:
- JAX and Torch are functionally equivalent in this repo for pi0.5 training objective.
- JAX returns `[B, H]` then training takes global mean.
- Torch forward may return unreduced `[B, H, D]`, but training takes global mean.
- Therefore implementations in this milestone must preserve this functional equivalence and must not
introduce a different objective.

Implementation rule for new static/perturbance code:
- Do not invent a new loss.

## Implementation

You should create separate functions in class PI0Pytorch(nn.Module) src/openpi/models_pytorch/pi0_pytorch.py for static inference that must NOT interfere with any original training or inference functionalities in this codebase, just like compute_static_targets, the previous static inference implementation for cosine and edr.  Keep zero interference by making new methods self-contained and never called by normal forward/sample_actions. Functions for different modes should also be separate, especially because for cosine two conditions need to be done at once while others only one condition is done at a time. Only add necessary functions, don't delete old function blocks.

Note that for the calculation of the perturbance mode there's a back propagation stage. Make sure to separate the math logic in prompts/perturbance-calculation-formalization.md into another function if possible so that the logic is clear.

For perturbance optimization, use the strict PI0 loss tensor (`mean over action-dim only`) as the
canonical loss definition; if a scalar is needed for backprop, only apply a final aggregate
reduction after computing that canonical tensor.


## Storage

You should follow the previous storage format as described in `documentation.md`, only with small file naming and location changes as specified in the storage section in prompts/two-conditions-spec.md. 

For specific contents inside individual trajectory folders, look at the storage section in prompts/two-conditions-spec.md. 

## Additional Notes

Some old documentations may still refer to a version of static inference code in openarm/ which is no longer maintained and already removed from the codebase. For this implementation, you should only look at and work in static_inference/. Ignore the openarm/ folder entirely.

In the last milestone, which is writing post processing data scripts, you will also want to modify the data processing scripts. For prior milestones, stick to the static inference logic. The new code under static_inference/ should be flat scripts, **NOT** a package. 

When running tests, you should always activate the python environment first `source /coc/testnvme/xzhang3205/openpi/.venv/bin/activate`

## sbatch scripts


There's already `launch_static.sbatch` for running the 4 datasets and a `static_launcher.py` (no additional args except a test flag allowed, must run as it is) for launching all conditions, including cosine (with save_meta=False), gradient training, gradient inference. The output roots are created by the launcher python file inside `/coc/testnvme/xzhang3205/static/franka_full` with dataset names franka_object  franka_object_plus  franka_object_two  franka_on_top, inside which cosine/ , gradient-training/ , gradient-inference/ , perturbance/, perturbance-all/ are the actual paths where the program writes files to. I want you extend the two files to support the launch of perturbance mode. `--save_meta=True` should be stated true for both testing and actual run, make it explicit in the launch script instead of modifying the default arg though (default should still be False).

In the python launcher the first few lines should have `folder_name='franka_full'` which I can always manually change later.
For the test flag, only franka_on_top should be launched, and the folder name should be test_currentime instead of franka_full. The skip frames should be set to very high so that the runs finish quickly (allow at least 5 steps actually computed though, for inference it will become 5*10=50). For the actual launch, keep the skip frame number in the previous code.

## Static inference norm stats override

`static_inference.py` supports an optional `--norm-stats-dir /path/to/stats_dir` argument for runs that need to override the train-config norm stats source. The provided directory must contain `norm_stats.json`. The override replaces the in-memory `data_config.norm_stats` before constructing the static inference transform, so it affects the normalized observations/actions and therefore all static inference metric calculations, including condition-inference. It does not modify the training config, checkpoint, checkpoint assets, or model weights on disk. Use only when the checkpoint genuinely lacks a corresponding usable training assets config.
