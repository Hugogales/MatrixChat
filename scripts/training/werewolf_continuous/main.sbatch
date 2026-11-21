#!/bin/bash
# Always-on Werewolf RL: one chain job (h100 / v100 / v100b / gapN), submitted by
# start.sh or the keeper and renewed by itself. Resources come from the
# submitter (start.sh CHAINS / RUN/launch.env). Everything else lives in
# chain.sh, which is read when the job starts (Slurm freezes this file at
# submit time, and follow-up jobs are queued days ahead).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -uo pipefail
CHAIN_SH="$REPO/scripts/training/werewolf_continuous/chain.sh"
# chain_start runs the whole job and never returns. If chain.sh cannot be read
# (or is broken mid-edit), keep the GPUs and try again every minute until STOP
# or the time limit, instead of failing and burning through the queued jobs.
while [ ! -f "$RUN_DIR/STOP" ]; do
  if bash -n "$CHAIN_SH" 2>/dev/null && source "$CHAIN_SH"; then
    chain_start
  fi
  echo "[main] $(date '+%F %T') could not run $CHAIN_SH; keeping the GPUs, trying again in a minute"
  sleep 60
done
