#!/usr/bin/env bash
# Launch only after preprocess_clean_handoff.sbatch validation has approved the
# new dataset. The worker pools enter job_queue and yield automatically.
set -euo pipefail

ROOT="/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat"
SEARCH_DIR="$ROOT/searches/clean_handoff_2026_09_explore"
SEARCH_CONFIG="$ROOT/scripts/search/configs/clean_handoff_2026_09_explore.json"

if [[ ! -f "$HOME/matrixchat/processed_clean_handoff_20260917_v1/manifest.json" ]]; then
  echo "[error] cleaned dataset is not ready" >&2
  exit 2
fi
if [[ -e "$SEARCH_DIR/search_state.json" ]]; then
  echo "[error] search already initialized: $SEARCH_DIR" >&2
  exit 2
fi

mkdir -p "$SEARCH_DIR" "$ROOT/logs/slurm"
cd "$ROOT"
export SEARCH_DIR SEARCH_CONFIG
CONTROLLER_JOB=$(sbatch --parsable --export=ALL,SEARCH_DIR,SEARCH_CONFIG \
  scripts/search/controller.sbatch)
WATCHDOG_JOB=$(sbatch --parsable --export=ALL,SEARCH_DIR \
  scripts/search/watchdog.sbatch)

for node in dh-dgx1-1 dh-dgx1-2 dh-dgx1-3; do
  python scripts/ops/job_queue.py add \
    --name "clean_handoff_workers_${node##*-}" \
    --sbatch scripts/search/worker_pool_v100.sbatch \
    --partition dgx \
    --gpu-count 8 \
    --priority 100 \
    --gres "gpu:v100:8" \
    --mem 400G \
    --nodelist "$node" \
    --ntasks 8 \
    --env "SEARCH_DIR=$SEARCH_DIR" "WORKERS=8" \
    --notes "Clean-handoff HPO worker pool; yield-managed and resumable"
done

echo "controller=$CONTROLLER_JOB watchdog=$WATCHDOG_JOB"
echo "Queued 24 V100 workers; the live yield daemon submits only idle capacity."
