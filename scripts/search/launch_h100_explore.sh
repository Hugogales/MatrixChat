#!/usr/bin/env bash
# Launch the widened H100 explore search (replaces combined_v2).
set -euo pipefail

ROOT="/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat"
SEARCH_DIR="$ROOT/searches/h100_2026_09_explore"
SEARCH_CONFIG="$ROOT/scripts/search/configs/h100_2026_09_explore.json"

mkdir -p "$SEARCH_DIR" logs/slurm

export SEARCH_DIR
export SEARCH_CONFIG

cd "$ROOT"

echo "Stopping prior combined_v2 search jobs if still running..."
for job in 274369 274370 274371; do
  scancel "$job" 2>/dev/null || true
done

echo "Submitting explore search controller/watchdog/workers..."
CONTROLLER_JOB=$(sbatch --parsable \
  --export=ALL,SEARCH_DIR,SEARCH_CONFIG \
  scripts/search/controller.sbatch)
WATCHDOG_JOB=$(sbatch --parsable \
  --export=ALL,SEARCH_DIR \
  scripts/search/watchdog.sbatch)
WORKERS_JOB=$(sbatch --parsable \
  --export=ALL,SEARCH_DIR \
  scripts/search/worker_pool.sbatch)

echo "controller=$CONTROLLER_JOB watchdog=$WATCHDOG_JOB workers=$WORKERS_JOB"
