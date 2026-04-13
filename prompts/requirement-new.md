Read prompts/completed.md for context.

Previously I asked you to implement an amendment (already done)
### D. `--metric="perturbance-noise"` (Amendment 1: displacement vector):
Add a toggle `static_inference.py .. --save_displacement_trace=True` that's default to True under mode D: perturbance-noise. Other modes should safely ignore this flag. If this flag is true, mode D should also calculate and store the norm of the displacement vector.

Now, the task is to implement a second amendent
### D. `--metric="perturbance-noise"` (Amendment 2: cosine-inference):
Add a toggle `static_inference.py .. --save_cosine=True` that's default to True under mode D: perturbance-noise. Other modes should safely ignore this flag. If this flag is true, mode D should also calculate and store the cosine like what mode A did, but only for condition-inference, since mode D is only supposed to run in the inference mode.

All storage/naming rules and mathematical formulations are identical to mode A except that only one condition-inference should run and its latent should store. Ignore condition-training. (mode A did both condition-training and condition-inference)

For --save_cosine=False in metric=perturbance-noise:
- Keep all existing non-cosine outputs exactly as before (gradnorm_*, final_layer_loss_*, and displacement outputs when enabled).
- Only omit cosine artifacts.

## Result Storage modification

Previously, if `--metric="perturbance-noise" --save_displacement_trace=False`, the results are saved into perturbance-noise/. If `--metric="perturbance-noise" --save_displacement_trace=True`, the results are saved into perturbance-noise-displacement/.

The new rule you should implement is that, regardless of whether save_displacement_trace and save_cosine is True or False, save the results into perturbance-all/. Under no condition should you save into perturbance-noise/ or perturbance-noise-displacement/ again. trajectory sublayout should be the same.


## Milestones

You should complete all milestones strictly one by one and wait for the user to verify each before proceeding to the next.
- Read all instructions and code. Read also the files that the instructions refer to. Read the codebase and understand the current static inference pipeline.
- Implement the requirements. 
- Modify corresponding sbatch scripts and launcher. (Don't delete any lines, just comment them out, and make the script only launch perturbance-noise with displacement for all datasets) Change the test to launch perturbance-noise with displacement, then wait for the user to run the test, then inspect the output and verify it looks correct (sanity checking size and tensor shape is sufficient).