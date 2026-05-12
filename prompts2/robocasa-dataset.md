Read prompts/completed.md for context. 

I want you to adjust the code run static inference (cosine similarity and on the robocasa datasets. You should run mode D only `--metric="perturbance-noise"`, but with all tags on `--save_displacement_trace=True --save_meta=True --save_cosine=True`.
The results should be stored in `/coc/testnvme/xzhang3205/static/robocasa/{mm-dd-time, hours and minutes only}` (the code should create that folder based on current time) 

`/coc/testnvme/xzhang3205/vla-adaptation/envs/robocasa/inference_prototype` shows how to load checkpoints for robocasa inference.

`/coc/testnvme/xzhang3205/vla-adaptation/envs/robocasa/dataset_loading_propotype` shows how to load the robocasa datasets.

Those two folder are for your references only. You should not depend your code on the two folders. You should definitely depend your code on envs/robocasa/ if necessary, but those prototypes folders are just examples and will be deleted later. If there're code block you'd like to reuse, just copy them here.

For checkpoint directories, you must use
checkpoint: /coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/base/pi05_base_torch
norm stats: /coc/testnvme/xzhang3205/vla-adaptation/checkpoints/openpi/base/pi05_base_torch/assets/franka/norm_stats.json

Use the norm stats override option specified in prompts/completed.md for this.

My hypothesis is that you don't need to change the data loading mechanism much, but if you do, make sure to separate the data loading logic in another file rather than static_inference/static_inference.py so we have a clean separation for different benchmarks franka, mesa, robocasa, openarm, etc, apart from the static inference logic.

Extend launcher.py **without deleting any of the current code**. If a previous run is no longer needed, just **comment it out**. It should launch and only launch all robocasa conditions. Also modify the previous result storing logic so that it stores at `/coc/testnvme/xzhang3205/static/{type}/{mm/dd/time, hours and minutes only}` instead of the hard-coded folder. type could be mesa, robocasa, openarm, ood, franka for now and more in the future.

The robocasa datasets you should run are: atomic-seen, composite-seen, composite-unseen. Treat them as different benchmarks (like how franka_on_top relates to franka_object_two). For normstats, you should always use the norm stats for the base pi05 checkpoint.

## clarifications

Robocasa suites should be separate --dataset values
launcher.py --test should run a very lightweight check to make sure the framework can run, e.g. one task one trajectory. Feel free to wipe the previous content in --test. For content change in --test, you **must not** modify any content out of the --test code block.
a local fallback loader is acceptable if LeRobotDataset rejects the dataset format
on-disk mode naming should remain perturbance-all/

## Milestones

You should complete all milestones strictly one by one and wait for the user to verify each before proceeding to the next.
- Read all instructions and code. Read also the files that the instructions refer to. Read the codebase and understand the current static inference pipeline.
- Implement the requirements. 
- Implement a small smoke test (modify launch.py --test), launch it via sbatch, and verify its results.

You should not launch full run sbatch scripts. Just set launcher.py to launch robocasa benchmarks and the user will manually launch.

codex resume 019e193a-b6d7-7b70-a6b1-42aad076e014

# Conditional RoboCasa Norm Stats Override Padding

## Summary

For RoboCasa static inference only, repair explicit norm-stats overrides only when they would otherwise fail due to dimension mismatch.
- Add a RoboCasa-only helper in static_inference.py used only inside:
if args.norm_stats_dir is not None and dataset_cfg.get("loader") == "robocasa":
- The helper inspects override stats for state and actions.
- If a stat key’s last dimension already equals or exceeds train_config.model.action_dim, leave it unchanged.
- If a stat key’s last dimension is smaller than train_config.model.action_dim, pad that key to model action dim:
    - keep original override values exactly
    - pad mean with 0.0
    - pad std with 1.0
    - pad q01/q99 similarly only if present
- Print a clear warning only when padding happens, including key, original dim, target dim, and override path.
- Do not modify MESA behavior.
- Do not modify training-config norm stats when no override path is passed.