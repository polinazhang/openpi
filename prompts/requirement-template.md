Read prompts/completed.md for context.


## Milestones

You should complete all milestones strictly one by one and wait for the user to verify each before proceeding to the next.
- Read all instructions and code. Read also the files that the instructions refer to. Read the codebase and understand the current static inference pipeline.
- Implement the requirements. 
- Modify corresponding sbatch scripts and launcher. (Don't delete any lines, just comment them out, and make the script only launch perturbance-noise with displacement for all datasets) Change the test to launch perturbance-noise with displacement, then wait for the user to run the test, then inspect the output and verify it looks correct (sanity checking size and tensor shape is sufficient).