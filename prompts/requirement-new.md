Read prompts/completed.md for context.

I want you to add a toggle `static_inference.py .. --save_displacement_trace=True` that's default to True under mode D: perturbance-noise. Other modes should safely ignore this flag. 

If this flag is true, mode D should also calculate and store the norm of the displacement vector, and the latent of the displacement vector itself if `--save_meta=True`. These share the same 10 diffusion step share the same model passes with the standard mode D and are thus computationally efficient to be put together in one mode.


## Formulation details of displacement vector

Compute, at each denoising step (k), the displacement vector from the initial pure-noise latent to the current predicted-action latent, and record its trajectory over all 10 steps.

$$
  \Delta A^{(k)} := A_t^{\tau_k}-A_t^1
$$
with $A_t^\tau \equiv x_t$ and $A_t^1$ as the initial pure-noise latent.

(In other notations) Compute the Latent-space displacement $$\Delta x^{(k)} = x^{(k)} - x^{(0)}$$


## Milestones

You should complete all milestones strictly one by one and wait for the user to verify each before proceeding to the next.
- Read all instructions and code. Read also the files that the instructions refer to. Read the codebase and understand the current static inference pipeline.
- Implement mode D addition `--metric="perturbance-noise --save_displacement_trace=True"`. 
- Modify corresponding sbatch scripts and launcher. (Don't delete any lines, just comment them out, and make the script only launch perturbance-noise with displacement for all datasets) Change the test to launch perturbance-noise with displacement, then wait for the user to run the test, then inspect the output and verify it looks correct (sanity checking size and tensor shape is sufficient).