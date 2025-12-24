#!/bin/bash
#SBATCH --job-name=BPT
#SBATCH --output=results/run-%J.out
#SBATCH --error=results/run-%J.err
#SBATCH --cpus-per-task=12
#SBATCH --time=1:00:00
#SBATCH --account=bfbo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1

module load cuda/12.6.1
source ~/.bashrc
cd /work/nvme/bfbo/xzhang42/openpi
source .venv/bin/activate
export OPENPI_DATA_HOME=/work/nvme/bfbo/xzhang42/openpi/.cache

# python scripts/compute_norm_stats.py \
# --config-name pi05_openarm_tea_continuous \
# --repo-root /work/nvme/bfbo/xzhang42/datasets/openarm_processed/openarm/tea_continuous

# python scripts/compute_norm_stats.py \
# --config-name pi05_openarm_tea_discrete \
# --repo-root /work/nvme/bfbo/xzhang42/datasets/openarm_processed/openarm/tea_discrete

python scripts/train.py \
pi05_openarm_tea_continuous \
--data.repo_id /work/nvme/bfbo/xzhang42/datasets/openarm_processed/openarm/tea_continuous \
--exp-name run14

# python scripts/train.py \
# pi05_openarm_tea_discrete \
# --data.repo_id /work/nvme/bfbo/xzhang42/datasets/openarm_processed/openarm/tea_discrete \
# --exp-name run8