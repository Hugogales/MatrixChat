#!/usr/bin/env bash
# Launch the widened V100 explore search on dgx1-2 + dgx1-3 (14 GPUs within
# the 16-GPU dgx budget while manual arms use 2 on dgx1-1). Worker pools are
# registered in job_queue so gpu_yield_daemon cancels and re-queues them when
# foreign users need GPUs.
set -euo pipefail

ROOT="/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat"
SEARCH_DIR="$ROOT/searches/v100_2026_09_explore"
SEARCH_CONFIG="$ROOT/scripts/search/configs/v100_2026_09_explore.json"

mkdir -p "$SEARCH_DIR" logs/slurm

export SEARCH_DIR
export SEARCH_CONFIG

cd "$ROOT"

echo "Submitting V100 explore controller + watchdog..."
CONTROLLER_JOB=$(sbatch --parsable \
  --export=ALL,SEARCH_DIR,SEARCH_CONFIG \
  scripts/search/controller.sbatch)
WATCHDOG_JOB=$(sbatch --parsable \
  --export=ALL,SEARCH_DIR \
  scripts/search/watchdog.sbatch)

echo "controller=$CONTROLLER_JOB watchdog=$WATCHDOG_JOB"

echo "Queueing V100 worker pools (yield-daemon managed, 7 GPUs each on dgx1-2/3)..."
python scripts/ops/job_queue.py add \
  --name v100_explore_workers_dgx1_2 \
  --sbatch scripts/search/worker_pool_v100.sbatch \
  --partition dgx \
  --gpu-count 7 \
  --priority 20 \
  --gres "gpu:v100:7" \
  --mem 350G \
  --nodelist dh-dgx1-2 \
  --ntasks 7 \
  --env "SEARCH_DIR=$SEARCH_DIR" "WORKERS=7" \
  --notes "V100 explore search worker pool on dgx1-2; yield-managed"

python scripts/ops/job_queue.py add \
  --name v100_explore_workers_dgx1_3 \
  --sbatch scripts/search/worker_pool_v100.sbatch \
  --partition dgx \
  --gpu-count 7 \
  --priority 21 \
  --gres "gpu:v100:7" \
  --mem 350G \
  --nodelist dh-dgx1-3 \
  --ntasks 7 \
  --env "SEARCH_DIR=$SEARCH_DIR" "WORKERS=7" \
  --notes "V100 explore search worker pool on dgx1-3; yield-managed"

echo "Done. gpu_yield_daemon (job 274345) will submit pools when dgx headroom allows."
echo "controller=$CONTROLLER_JOB watchdog=$WATCHDOG_JOB"
