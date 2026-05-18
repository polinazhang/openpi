Read prompts/completed.md (if descriptions contradict with the current code base status, always calibrate to the current codebase, that file should only serve as a general description for you to understand what the codebase is doing on a high level), prompts2/robocasa-dataset.md and prompts2/mesa-dataset.md (which has also been completed) for context. 

Objective: for all benchmarks datasets: robocasa, mesa, franka, (1) run the static-inference metric on exactly one episode, (2) render the content of that trajectory, and (3) visualize the results similar to the previous rendering in models/openpi/static_inference/analyze_results.py, just that now it takes the content of one trajectory only. Do it for the first episode of the first task (if there is a task separation).

For data gathering, you should write a separate launcher prototype_launcher.py that resembles the structure of launcher.py but only launches per episode job. You should also consider how to render the specific trajectories (if the rendering already exist, copy them to the desired location, otherwise grab from the training data and render)

Rendering just means creating a video from the data, no need to start the simulators (actually, don't start them). Also store the action values with the videos for that trajectory so that they can be aligned later. You can decide this format.

Completing all the requirements in the file should not interfere with the functionalities of this codebase. 

All results should exist in /coc/testnvme/xzhang3205/vla-adaptation/prototype, the static inference results in /coc/testnvme/xzhang3205/vla-adaptation/prototype/static/.. (same with the naming convention in the static inference code) while the trajectory rendering in 


Render the videos in mp4 format (and if they're larger than 99mb, compress them to stay within this limit). all used cameras should be rendered (3 or 2 depending on the benchmark)

Visualize following the convention of the prior visualization code but doesn't have to share the same script. Creating a different script is prefered if too much change needs to be made to the original script.


## Milestones

You should complete all milestones strictly one by one and wait for the user to verify each before proceeding to the next.
- Read all instructions and code. Read also the files that the instructions refer to. Read the codebase and understand the current static inference pipeline.
- Implement the requirements. 
- Comment out mesa and franka temporarily, submit robocasa via sbatch, and verify its results. No need for a separating testing script since it's a;ready single episode. This should store in /coc/testnvme/xzhang3205/vla-adaptation/prototype/tmp and be deleted later. Verify the results make sense
- Uncomment mesa and franka and launch all benchmarks, storing in the specified location.
- Write a README.md in /coc/testnvme/xzhang3205/vla-adaptation/prototype/visuals specifying how to align the videos with the stored action values if one wants to write a script to snapshot a timestamp and get the aligned action & frame.
- Write the visualization script and put the results in /coc/testnvme/xzhang3205/vla-adaptation/prototype/results

codex resume 019e1a37-a66c-7383-a207-b3ff31c2e363