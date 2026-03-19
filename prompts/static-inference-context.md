## Context
**Static inference** is an analysis-only mode. Given demonstration samples, the VLA takes the state and other inputs, runs a “fake” forward denoising process, and does not execute the resulting actions. The generated latents are then compared against the ground-truth actions from the demonstration trajectories.

**Static inference metrics** measure the discrepancy between the direction predicted by the model and the ground-truth direction it is expected to follow. Examples include cosine similarity and EDR, defined in `prompts/edr-cosine.md`.

**condition-training** refers to static inference under training-time conditions, where the model starts from x_t = tau*noise + (1-tau)*actions, with sampled tau, and runs one step.

**condition-inference** refers to static inference under inference-time conditions, where the model starts from x_t = noise, and runs num_steps matching the model inference settings.

## Current state of implementation
The **condition-training** for cosine similarity and EDR has been implemented, and the values are saved during running automatically. See `prompts/documentation.md` (especially the static inference section) for details on what files and functionalities have been added to the openpi repo already. Note that in that file there's a dynamic inference section as well, you don't need to take care of what it means, but read it to have a rough sense of what files have been added by that so that you can distinguish those extra files/functions from the original openpi ones.

The **condition-inference** for cosine similarity and EDR has NOT been implemented.

Both **condition-training** and **condition-inference** for gradient guidance vectorhave NOT been implemented.

The launch method is `static.sbatch`, which calculates cosine and EDR and final layer loss for static inference of the pi05 model on the specified dataset. Note that cosine and EDR are per layer metrics so per layer latents are saved, but final loss and gradient guidance vector are per inference.