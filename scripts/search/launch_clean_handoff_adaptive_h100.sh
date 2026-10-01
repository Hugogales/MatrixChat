#!/usr/bin/env bash
# Launch the learned QKV-gated + GRU adaptive search on seven H100s,
# retaining one H100 as headroom for other users.
set -euo pipefail

ROOT="/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat"
SEARCH_DIR="$ROOT/searches/clean_handoff_adaptive_h100_2026_09_v3"
SEARCH_CONFIG="$ROOT/scripts/search/configs/clean_handoff_adaptive_h100_2026_09.json"

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

python scripts/ops/job_queue.py add \
  --name adaptive_clean_h100_workers \
  --sbatch scripts/search/worker_pool.sbatch \
  --partition dgxh100 \
  --gpu-count 7 \
  --priority 100 \
  --gres "gpu:h100:7" \
  --mem 600G \
  --ntasks 7 \
  --env "SEARCH_DIR=$SEARCH_DIR" "WORKERS=7" \
  --notes "Learned gated-QKV + GRU adaptive clean-handoff HPO; yield-managed"

echo "controller=$controller watchdog=$watchdog"
