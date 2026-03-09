## TO DO


In `examples/franka_real/config.py`, before every run, set:

1) the model checkpoint and its normalization-stats path

2) `POLICY_EVALUATION_SUITE_NAME`, which specifies the folder where the evaluation latents are saved

3) `POLICY_LANGUAGE_INSTRUCTION`, which specifies the current task to the VLA.

Also set one-time inside `RobotRuntimeConfig`
- max_allowed_inferences_per_episode
- max_allowed_episode_seconds

Note that default latent storage path is `/data3/openpi`. Try not to change it; this directory has the most spare storage.
