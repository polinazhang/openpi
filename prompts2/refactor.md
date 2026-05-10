Objective: Revert a previous code change that is no longer needed. This should not affect the new functionalities.

Context: ../prompts/completed.md and files it refers to contain description about the new functionalities, which we abbreviated by static-new. In the documentations, a previous implementation was quoted, which was abbreviated by static-openarm (quoted by "Some old documentations may still refer to a version of static inference code in openarm/ which is no longer maintained.") static-openarm was to support cosine similarity calculations on the openarm dataset but used a legacy framework that I want you to completely rip without affecting all scripts supported by static-new. At development, the two functionalities were explicitly separated and shouldn't share any infrasctructure level functions (if you find out that they do, immediately stop your current task and let me know)

You're currently working on the static git branch, which contains both modifications from static-new and static-openarm. The main git branch contains only modifications from static-openarm. I want you to investigate the main branch to inspect all commits including and after cdac8768e6fe44dca0b6c21a0d2f8e8d1d69076d (all descendants from that commit to current main (cdac8768..main)), which refers to static-openarm changes. Additionally, an earlier commit 3145167dfbb8aa439f87cf61ee74771fedf3f12a is also an static-openarm change and needs to be included. Here, by "changes", I specifically mean absoulte modifications compared to the previous codebase within the main branch. You must NOT compare anything with the static/branch during the inspection.

After you identified all changes made, I want you to revert all static-openarm changes. Check individually if they will affect static-new, if not, revert them. However, one exception is applied: if the modified file is inside folder openarm/, keep all modifications. The folder openarm/ should not be modified at all for reference. But all modifications outside openarm/ should be wiped.

All work should be done from the static branch only. main should be used only for investigation and never modified.

Also, one exception -- if the file path contains /coc/testnvme/.., always keep it. if the file path contains /work/nvme/xzhang42... always discard it. This overrides all other instructions, but at the file path level only.

## Milestones

You should complete all milestones strictly one by one and wait for the user to verify each before proceeding to the next.

- Investigate the documentations and the repo
- Investigate the git changes in the main branch
- Revert the changes. (pi0_torch.py is already completed but others still needs to be modified.) For each change, make sure it does not affect static-new, and then wipe. If it does affect static-new, stop, describe the situation, and wait for the user's input
- Run a lightweight test on all 4 modes in static-new (you only need one or two trajectories for this sanity check) to make sure everything is still running, and static_launcher.py can **run in its current form**. You can modify the try mode in the launcher but don't interfere with anything else in the static-new pipeline. 

## Reference
codex resume 019deaab-fd93-7db3-aba1-c19d8528a1fb