#!/usr/bin/env bash
#SBATCH --job-name=OPI-STATIC-WORKFLOW
#SBATCH --chdir=/coc/testnvme/xzhang3205/vla-adaptation/models/openpi/static_inference/launchers
#SBATCH --partition=kira-lab
#SBATCH --qos=long
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=ordered-workflow-%j.out
#SBATCH --error=ordered-workflow-%j.err

set -uo pipefail

INTERVAL_SECONDS=1200
MAX_RUNTIME_SECONDS=$((7 * 24 * 3600))
SAFETY_MARGIN_SECONDS=60
PYTHON=/coc/testnvme/xzhang3205/vla-adaptation/models/openpi/.venv/bin/python
MANAGER=launch_robocasa365_ordered.py

start_time=$(date +%s)
end_time=$((start_time + MAX_RUNTIME_SECONDS - SAFETY_MARGIN_SECONDS))

"${PYTHON}" "${MANAGER}" --initialize
initialize_status=$?
if (( initialize_status != 0 )); then
    echo "workflow initialization failed with exit code ${initialize_status}" >&2
    exit "${initialize_status}"
fi

echo "ordered static-inference workflow started at $(date --iso-8601=seconds)"
echo "controller job: ${SLURM_JOB_ID:-unknown}; maximum active A40 jobs: 20"

while true; do
    now=$(date +%s)
    if (( now >= end_time )); then
        echo "seven-day cutoff reached; exiting"
        break
    fi

    "${PYTHON}" "${MANAGER}" --run-once
    pass_status=$?
    if (( pass_status != 0 )); then
        echo "workflow pass failed with exit code ${pass_status}; next pass will retry" >&2
    fi

    now=$(date +%s)
    elapsed=$((now - start_time))
    next_run=$((start_time + ((elapsed / INTERVAL_SECONDS) + 1) * INTERVAL_SECONDS))
    if (( next_run >= end_time )); then
        echo "next pass would exceed cutoff; exiting"
        break
    fi
    sleep_seconds=$((next_run - now))
    if (( sleep_seconds > 0 )); then
        echo "sleeping ${sleep_seconds}s until next 20-minute pass"
        sleep "${sleep_seconds}"
    fi
done

echo "ordered static-inference workflow stopped at $(date --iso-8601=seconds)"
