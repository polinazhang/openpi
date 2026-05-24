1. Train set and eval set have same tasks, but the action recordings are separate. (Registry points to different pretrain vs target dataset paths.)

2. Currently, train_launcher.py only supports per-single-task finetuning for robocasa. Finetuning on multiple tasks combined still to be added.

3. Finetuning on mesa is also not supported currently, to be added.
