For context, read `prompts/static-inference-context.md` and all the files it refers to before reading this document.

Extend the current code to support both **condition-training** and **condition-inference**. Stop calculating edr for now, but don't remove the code that supported it. Instead, comment them out and label that they're for EDR calculation. The new pipeline I want you to implement should not calculate edr but should keep the door if we want to add it in future.

## Required Code Functionality

There should be a flag `--metric=""|options:"cosine","gradient"`. The two conditions are dicussed separately:

A. `--metric="cosine"`: 

If cosine is selected, condition-training and condition-inference should be calculated simutanously in the same run. For inference on every input, the two conditions should use the same randomized noise initialization, only different should be one starts from x_t = tau*noise + (1-tau)*actions and one starts from x_t = noise. Basically, given one datapoint (one noise and one input), two static inferences should be both be done: condition-training and condition-inference; then repeat on the entire trajectory; which should repeat across trajectories. 

In this mode, there should be a toggle `static_inference.py .. --save_meta=True` means that latents should be saved: u, ctraining-v, cinference-v, which is default to False. Details about these latents specified in the storage section in `prompts/two-conditions-spec.md`. if metric flag is not cosine, save_meta should just be neglected.

There was a skip frame functionality in the old code so not all the inputs are taken. Instead, for every x state-actions, one will be taken. Keep it.

B. `--metric="gradient"`: 
condition-training and condition-inference should be calculated in separate runs. There should be a toggle `static_inference.py .. --condition=training/inference` controlling that. The condition toggle should be disregarded if cosine is selected as metric. For how to calculate, details are included in `prompts/new-metric-definition.md`.

As for the skip frame functionality, you should repeat the pattern but with one caveat. The calculation of gradient guidance vector would require to take a chunk of ground truth action instead of a single action from the sample trajectory. When there's not enough actions left in the trajectory, you should stop computing the gradient. Don't pad, just discard the tail. 

## Implementation

You should create separate functions in class PI0Pytorch(nn.Module) src/openpi/models_pytorch/pi0_pytorch.py for static inference that must NOT interfere with any original training or inference functionalities in this codebase, just like compute_static_targets, the previous static inference implementation for cosine and edr.  Keep zero interference by making new methods self-contained and never called by normal forward/sample_actions. Functions for consine and gradient guidance vector should also be separate, especially because for cosine two conditions need to be done at once while for gradient guidance vector, only one condition is done at a time.

Note that for the calculation of gradient guidance vector there's a back propagation stage. For the calculation of gradient guidance vector, make sure to separate the math logic in prompts/new-metric-definition.md into another function if possible so that the logic is clear.


## Storage

You should follow the previous storage format as described in `documentation.md`, only with small file naming and location changes as specified in the storage section in prompts/two-conditions-spec.md. 

For specific contents inside individual trajectory folders, look at the storage section in prompts/two-conditions-spec.md. 

## Additional Notes

As you've already read from the documentations, the previous static inference code is in openarm/, but no since we're going to make substantial changes, I want you to create static_inference/, copy the relevant files from openarm/, and work there. Don't modify anything in openarm/, just treat it as outdated. Only copy all necessary files for the workflow and abort standalone scripts not actively used. After you finish, also create launch_static.sbatch that launches the job from static_inference/ instead of openarm/. Explicitly state `--save_meta=True` there though because during testing I want to see everything. In the last milestone, which is writing post processing data scripts, you will also want to modify the data processing scripts you copied. For prior milestones, stick to the static inference logic. The new code under static_inference/ should be flat scripts, **NOT** a package. 

## sbatch scripts

There should be `launch_static.sbatch` for running the 4 datasets but with refined code (`see static.sbatch` on how previously it was done). There should also be a `static_launcher.py` (no additional args except a test flag allowed, must run as it is) for launching all conditions, including cosine (with save_meta=False), gradient training, gradient inference. The output roots should be created by the launcher python file inside `/coc/testnvme/xzhang3205/static/franka_full` with dataset names franka_object  franka_object_plus  franka_object_two  franka_on_top, inside which cosine/ , gradient-training/ , gradient-inference/ are the actual paths where the program writes files to.
In the python launcher the first few lines should have `folder_name='franka_full'` which I can always manually change later.
For the test flag, only gradient training and inference for franka_on_top should be launched, and the folder name should be test_currentime instead of franka_full. The skip frames should be set to very high so that the runs finish quickly (allow at least 5 steps actually computed though, for inference it will become 5*10=50). For the actual launch, keep the skip frame number in the previous code.
Since the implementation is gradient first and cosine next (will be specified later), when you first implement this, only launch all gradient sbatch scripts. After cosine is completed, refine the script to launch all.

## Milestones

You should complete all milestones strictly one by one and wait for the user to verify each before proceeding to the next.
- Read all instructions and code. Read also the files that the instructions refer to. Read the codebase and understand the current static inference pipeline.
- Copy the relevant codes to the static_inference folder and plan what adjustments need to be made.
- Implement mode `--metric="gradient"`. Leave room in the code for cosine when you do this, just don't sabotage future architecture extension. 
- Implement corresponding sbatch scripts and launcher. Wait for the user to run the test, then inspect the output and verify it looks correct (sanity checking size and tensor shape is sufficient).
- Write data post-processing scripts to print the final layer losses and the gradient values [*] for the franka_full run. Hard code all paths. Result should be saved to static_results/result_gradient.csv and print in command window as well. 
[*] for gradient post processing, you should combine the two conditions. Denote the gradient generated by condition-training as v_action, and the gradient generated by condition-inference as v_all, both should be vectors of the same dimensions. Calculate v_vision = v_all - v_action, then calculate the L2 norm and squared L2 norm of v_action, v_vision, and v_all, and report all these values. They should be averaged across datapoints and trajectories evenly but you also need to calculate their std.

- Implement mode `--metric="cosine"`. 
- Modify corresponding sbatch scripts and launcher. Change the test to launch cosine, then wait for the user to run the test, then inspect the output and verify it looks correct (sanity checking size and tensor shape is sufficient).
- On the result dataset, calculate cosine(ctraining-v_{layer_idx}, u), cosine(cinference-v_{layer_idx}, u), and compare them with ctraining-cosine_{layer_idx} and cinference-cosine_{layer_idx}. They should be identical, if not, debugging why it fails and explain.
- Write data post-processing scripts to print the final layer losses and the cosine values for the franka_full run. Hard code all paths. Result should be saved to static_results/result_cosine.csv and print in command window as well.