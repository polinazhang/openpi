# Objective

Launch static inference on one atomic seen task given the task number. Datasets are in lerobot format and located in `/coc/testnvme/xzhang3205/vla-adaptation/datasets/robocasa365/atomic-seen-splits/task_{i}_demo_50` where i is 1 to 18. Only use these datasets and ignore the rest. 18 runs should be launched separately so you should write code that launches one sbatch given one task number, each takes 1 A40 gpu.

Your objective is to write code to support static inference runs. For context see static-inference-context.md, cosine-similarity.md, vision-grad-norm.md, robocasa.md.

For launching and the main body of code, you should write inside `/coc/testnvme/xzhang3205/vla-adaptation/models/openpi/static_inference`. Since the goals require some change in openpi architecture, you're allowed to modify ``/coc/testnvme/xzhang3205/vla-adaptation/models/openpi`.


# Storage

Cosine similarity and vision grad norm will be saved in the same run.

The content inside meta/ are decided by the flag `--save_meta=True` passed in to the static inference script

The latents that should be saved as files are (use these as file names followed by .npy as well):
- meta/u             (this should be the prediction target: v = groud_truth_actions_from_demo − noise)
- meta/v_{diffusion_step_idx}  [this should be the velocity prediction by the model]
- final_loss_{diffusion_step_idx} [this should be the loss value of the inference computation; the same loss computed in training]
- cosine_{diffusion_step_idx}
- gradnorm_vision_step_{diffusion_step_idx}  [this should be the local sensitivity score $||\nabla_{h_v}L^{(n)}||_2$, where n is the step idx]
Note that here diffusion_step_idx refers to the number of inference steps, set 10 from pi05 default.
stored u/v should be masked to real dims.

All values are calculated per frame. When a rollout episode finishes (corresponds to one demo trajectory used), they should be stacked together and saved as npy files. The stacking mechanism should be described in a documentation so other users upon viewing can know exactly what is what.

The content inside /meta should be decided by the flag `--save_meta=True` passed in to the static inference script


