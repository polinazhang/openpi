#!/bin/bash
#SBATCH --job-name=block-qt1
#SBATCH --output=results/block-qt1-%J.out
#SBATCH --error=results/block-qt1-%J.err
#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --mem=440G
#SBATCH --nodelist=qt-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos="short"
#SBATCH --time=00:01:00
#SBATCH --exclusive

set -euo pipefail

echo "Holding qt-1 for 60 seconds"
hostname
sleep 60
