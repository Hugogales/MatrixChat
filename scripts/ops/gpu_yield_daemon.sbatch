#!/bin/bash
# Always-on GPU yield/reclaim daemon (teaching partition, 0 GPUs, minimal
# resources) -- see scripts/ops/gpu_yield_daemon.py's module docstring for
# the full behavior and safety rails.
#
# SAFE FIRST LAUNCH (dry-run, logs only, touches nothing):
#   sbatch scripts/ops/gpu_yield_daemon.sbatch
#
# LIVE (actually holds/cancels/resubmits jobs) -- only after reviewing a
# burn-in period of dry-run output in logs/gpu_yield_daemon/decisions.jsonl:
#   LIVE=1 sbatch scripts/ops/gpu_yield_daemon.sbatch
#
# Faster polling (hits slurmctld harder -- see module docstring):
#   INTERVAL_SECONDS=1 sbatch scripts/ops/gpu_yield_daemon.sbatch
#
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_gpu_yield_daemon
#SBATCH --partition=teaching
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -uo pipefail

LIVE="${LIVE:-0}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-15}"

ARGS=(--interval-seconds "$INTERVAL_SECONDS")
if [[ "$LIVE" == "1" ]]; then
  ARGS+=(--live)
fi

python scripts/ops/gpu_yield_daemon.py "${ARGS[@]}"
