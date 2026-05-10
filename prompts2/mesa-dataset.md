Read prompts/completed.md for context. 

I want you to adjust the code run static inference (cosine similarity and on the mesa datasets. You should run mode D only `--metric="perturbance-noise"`, but with all tags on `--save_displacement_trace=True --save_meta=True --save_cosine=True`.
The results should be stored in `/coc/testnvme/xzhang3205/static/mesa/{mm/dd/time, hours and minutes only}` (the code should create that folder based on current time) 

`/coc/testnvme/xzhang3205/vla-adaptation/envs/mesa-env/inference_prototype` shows how to load the pi05 mesa checkpoint for inference.

`/coc/testnvme/xzhang3205/vla-adaptation/envs/mesa-env/dataset_loading_propotype` shows how to load the mesa datasets.

Those two folder are for your references only. You should not depend your code on the two folders. You should definitely depend your code on envs/mesa-env/ if necessary, but those prototypes folders are just examples and will be deleted later. If there're code block you'd like to reuse, just copy them here.

My hypothesis is that you don't need to change the data loading mechanism much, but if you do, make sure to separate the data loading logic in another file rather than static_inference/static_inference.py so we have a clean separation for different benchmarks franka, mesa, openarm, etc, apart from the static inference logic.

Extend launcher.py **without deleting any of the current code**. If a previous run is no longer needed, just **comment it out**. It should launch and only launch all mesa conditions. Also modify the previous result storing logic so that it stores at `/coc/testnvme/xzhang3205/static/{type}/{mm/dd/time, hours and minutes only}` instead of the hard-coded folder. type could be mesa, openarm, ood, franka for now and more in the future.

The mesa datasets you should run are: mesa-70, mesa-instance, mesa-spatial, mesa composite (even if incomplete). Leave mesa-category out since there's no data. Treat them as different benchmarks (like how franka_on_top relates to franka_object_two). For normstats, you should always use the norm stats for the mesa pi05 checkpoint.

## clarifications

MESA suite cache should be auto-built if missing
MESA suites should be separate --dataset values
"always use the MESA pi05 checkpoint norm stats" but enforce it implicitly by using config pi05_mesa is fine, if the smoke test printed the correct directory. I will check that. Don't explicitly rule other norm stats out at the root of the code.
launcher.py --test should run a very lightweight check to make sure the framework can run, e.g. one task one trajectory. 
a local fallback loader is acceptable if LeRobotDataset rejects the dataset format
on-disk mode naming should remain perturbance-all/

## Milestones

You should complete all milestones strictly one by one and wait for the user to verify each before proceeding to the next.
- Read all instructions and code. Read also the files that the instructions refer to. Read the codebase and understand the current static inference pipeline.
- Implement the requirements. 
- Implement a small smoke test (modify launch.py --test), launch it via sbatch, and verify its results.

You should not launch full run sbatch scripts. Just set launcher.py to launch mesa benchmarks and the user will manually launch.

codex resume 019dfb5b-eaba-78a1-8fda-701693b392c0