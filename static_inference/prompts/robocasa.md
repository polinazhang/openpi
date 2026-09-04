For how to launch normal inference, see

/coc/testnvme/xzhang3205/vla-adaptation/inference/run_one.py
/coc/testnvme/xzhang3205/vla-adaptation/inference/run.sbatch


This is for your reference when writing static inference code. For static inference you should always use model `checkpoints/openpi/base/pi05_base_torch` and its statistic.json. You must not recompute statistics.json -- use that