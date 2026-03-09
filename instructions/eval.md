You should support the following behaviors when starting the robot communicator:

A. The default mode to launch by the bash script and if you directly run the code without additional arugments should be evaluation mode. In this mode, you should start multiple evaluation episodes. You should start one, then requiring user's eplicit input, then start another one, ..., until the user kills the code. 

For every episode, you create the folder POLICY_EVALUATION_SUITE_NAME at the target dir if it doesn't exist. Then read how many folders are inside the folder with name POLICY_EVALUATION_SUITE_NAME, and create a subfolder inside the folder with the number. If you just created the folder, you should create subfolder 0000; if the folder already exists and contains 2 subfolders, you should create subfolder 0002. 

During an episode, if the user press any key, you should stop that episode and also follow the convention in `instructions/custom_openpi.md`. [IMPORTANT] **you must explictly call policy.end_trajectory() when an episode is saved for the latents to save**. After that, you should then echo in the terminal "Please reset the robot. Has the reset finished? y/N", and wait for a y input from the user to start the next episode. 

B. You should also create a test mode the user should run with --test argument. It should run 5 inferences and execute them and then stop. For this, you should ignore POLICY_EVALUATION_SUITE_NAME and just the latents in the test directory (create it if doesn't exist!) inside the target save dir.

You should also modify the bash script launcher to make this command window the biggest in the multi-terminal.