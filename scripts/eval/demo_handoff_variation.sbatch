#!/bin/bash
# Full sweep: prompts x agent-counts x seeds x context-kinds (cold, primed
# handoff with 0/3-column gap, post-intro, lengthy conversation), with
# chain-continuation to look for 2nd/3rd handoffs -- see
# scripts/eval/demo_handoff_variation.py.
#
# 2026-08-25: switched from the frozen scripts/eval/compiled/*.pyc (last
# recompiled 08-05) back to the live .py -- the frozen bytecode's inlined
# `_score_handoff` unpacking (8 values) went stale once the shared scoring
# helper was updated to also return `interruption_handoff` (9 values),
# causing an immediate `ValueError: too many values to unpack` on every run.
# See .cursor/rules/stale-file-cache-null-bytes.mdc's "other symptoms"
# section for this exact class of bug (stale compiled bytecode vs. live
# source, not a raw-bytes corruption this time).
#
# 2026-08-25 (later same day): the 4h time limit is not always enough for
# the full 576-trial sweep -- job 274206 was killed by the time limit at
# roughly 50% progress with ZERO output written (no incremental save
# existed), losing the entire 4 GPU-hours. Fixed `demo_handoff_variation.py`
# to periodically dump partial progress to `<OUTPUT>.partial.json` and added
# `--resume_from`/`RESUME_FROM` so a future timeout can pick back up instead
# of restarting from trial 0. Also bumped the time limit itself (6h) since
# even a resumed run still needs real wall-clock, this just makes a second
# timeout much less likely and much cheaper if it happens anyway.
#SBATCH --account=undergrad_research
#SBATCH --job-name=demo_variation_v2
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CONTAINER="/data/containers/msoe-tf2x.sif"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
CKPT_DIR="${CKPT_DIR:-checkpoints/race_20260720_P_kitchen_sink/last}"
OUTPUT="${OUTPUT:-logs/probe_results/demo_handoff_variation_v2_result.json}"
RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
  RESUME_ARGS=(--resume_from "$RESUME_FROM")
fi

singularity exec --nv -B /data:/data "$CONTAINER" python scripts/eval/demo_handoff_variation.py \
  --checkpoint_dir "$CKPT_DIR" \
  --model_path "$MODEL_PATH" \
  --output "$OUTPUT" \
  --device cuda \
  "${RESUME_ARGS[@]}"
