#!/usr/bin/env bash
# Launch the learned QKV-gated + GRU clean-handoff search on 23 V100s,
# reserving one GPU for diagnostics/evaluation work.
set -euo pipefail

ROOT="/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat"
SEARCH_DIR="$ROOT/searches/clean_handoff_adaptive_2026_09_v4"
SEARCH_CONFIG="$ROOT/scripts/search/configs/clean_handoff_adaptive_2026_09.json"

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
controller=$(sbatch --parsable --export=ALL,SEARCH_DIR,SEARCH_CONFIG \
  scripts/search/controller.sbatch)
watchdog=$(sbatch --parsable --export=ALL,SEARCH_DIR \
  scripts/search/watchdog.sbatch)

for spec in "dh-dgx1-1:7" "dh-dgx1-2:8" "dh-dgx1-3:8"; do
  node="${spec%%:*}"
  workers="${spec##*:}"
  python scripts/ops/job_queue.py add \
    --name "adaptive_clean_workers_${node##*-}" \
    --sbatch scripts/search/worker_pool_v100.sbatch \
    --partition dgx \
    --gpu-count "$workers" \
    --priority 100 \
    --gres "gpu:v100:${workers}" \
    --mem "$((workers * 50))G" \
    --nodelist "$node" \
    --ntasks "$workers" \
    --env "SEARCH_DIR=$SEARCH_DIR" "WORKERS=$workers" \
    --notes "Learned gated-QKV + GRU adaptive clean-handoff HPO; yield-managed"
done

echo "controller=$controller watchdog=$watchdog"
