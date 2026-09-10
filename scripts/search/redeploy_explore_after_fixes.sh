#!/usr/bin/env bash
# Apply post-screening fixes: pause new sampling, restart controllers/workers
# with worker.py + broad_sweep_score changes, restart H100 pool, relaunch struct_mix.
set -euo pipefail

ROOT="/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat"
cd "$ROOT"

patch_search_state() {
  local dir="$1"
  python3 - <<PY
import json
from pathlib import Path
p = Path("$dir/search_state.json")
state = json.loads(p.read_text())
state["config"]["target_active_candidates"] = 0
state["config"]["warm_start_leader_checkpoint"] = "checkpoints/final40_h100_c106_s176106"
p.write_text(json.dumps(state, indent=2) + "\n")
print("patched", p)
PY
}

echo "Patching live search_state (pause sampling, c106 warm-start)..."
patch_search_state searches/h100_2026_09_explore
patch_search_state searches/v100_2026_09_explore

echo "Restarting controllers/watchdogs..."
for job in 274420 274421 274422 274393; do scancel "$job" 2>/dev/null || true; done
sleep 2
for dir in searches/h100_2026_09_explore searches/v100_2026_09_explore; do
  rm -f "$dir/controller.lock" 2>/dev/null || true
done

export SEARCH_DIR=searches/h100_2026_09_explore
export SEARCH_CONFIG=scripts/search/configs/h100_2026_09_explore.json
H100_CTRL=$(sbatch --parsable --export=ALL,SEARCH_DIR,SEARCH_CONFIG scripts/search/controller.sbatch)
H100_WD=$(sbatch --parsable --export=ALL,SEARCH_DIR scripts/search/watchdog.sbatch)

export SEARCH_DIR=searches/v100_2026_09_explore
export SEARCH_CONFIG=scripts/search/configs/v100_2026_09_explore.json
V100_CTRL=$(sbatch --parsable --export=ALL,SEARCH_DIR,SEARCH_CONFIG scripts/search/controller.sbatch)
V100_WD=$(sbatch --parsable --export=ALL,SEARCH_DIR scripts/search/watchdog.sbatch)

echo "Restarting worker pools (pick up worker.py fix)..."
for job in 274424 274434 274463; do scancel "$job" 2>/dev/null || true; done
sleep 2

export SEARCH_DIR=searches/v100_2026_09_explore
V100_W2=$(sbatch --parsable --partition=dgx --nodelist=dh-dgx1-2 --gres=gpu:v100:7 --ntasks=7 --mem=350G \
  --export=ALL,SEARCH_DIR,WORKERS=7 scripts/search/worker_pool_v100.sbatch)
V100_W3=$(sbatch --parsable --partition=dgx --nodelist=dh-dgx1-3 --gres=gpu:v100:7 --ntasks=7 --mem=350G \
  --export=ALL,SEARCH_DIR,WORKERS=7 scripts/search/worker_pool_v100.sbatch)

export SEARCH_DIR=searches/h100_2026_09_explore
H100_W=$(sbatch --parsable --export=ALL,SEARCH_DIR scripts/search/worker_pool.sbatch)

echo "Relaunching struct_mix manual arm (fresh from c106, exclude dh-dgx1-1)..."
STRUCT=$(sbatch --parsable --partition=dgx --gres=gpu:1 --exclude=dh-dgx1-1 \
  --job-name=mc_explore_struct_warm \
  --export=ALL,RUN_NAME=explore_structure_mix_warm_s176106,SEED=176106,TURN_REWARD_MODE=balanced_ce,FLOOR_CONTROL_ADAPTIVE_WEIGHT_ALPHA=0.75,DATASET_WEIGHTS="bazinga=0.4,when2speak=0.35,molweni=0.25",INIT_FROM=checkpoints/final40_h100_c106_s176106,INIT_CHECKPOINT_PREFER=best_probe,QUALITY_OVERSAMPLE_FACTOR=2.0,PROCESSED_DIR="$HOME/matrixchat/processed_final_20260814_lookback192_allspk" \
  scripts/training/train_dynstate_ablation.sbatch)

echo "h100_ctrl=$H100_CTRL h100_wd=$H100_WD h100_workers=$H100_W"
echo "v100_ctrl=$V100_CTRL v100_wd=$V100_WD v100_workers=$V100_W2 $V100_W3"
echo "struct_mix=$STRUCT"
