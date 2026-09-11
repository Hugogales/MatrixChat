#!/bin/bash
# Frozen-checkpoint factorial diagnostic: agent count x activity threshold x
# temperature x rollout horizon, plus a compact context-kind slice.
#SBATCH --account=undergrad_research
#SBATCH --job-name=listener_sensitivity
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to a frozen checkpoint directory}"
OUTPUT="${OUTPUT:?set OUTPUT to the result JSON path}"

singularity exec --nv -B /data:/data "$CONTAINER" \
  python scripts/eval/listener_response_sensitivity.py \
  --checkpoint_dir "$CKPT_DIR" \
  --model_path "$MODEL_PATH" \
  --output "$OUTPUT" \
  --agent_counts "${AGENT_COUNTS:-3,4}" \
  --activity_thresholds "${ACTIVITY_THRESHOLDS:-0.35,0.40,0.45,0.50,0.55,0.60}" \
  --temperatures "${TEMPERATURES:-0.0,0.8}" \
  --horizons "${HORIZONS:-16,48}" \
  --stochastic_seeds "${STOCHASTIC_SEEDS:-0,1,2,3}" \
  --num_prompts "${NUM_PROMPTS:-6}" \
  --context_kinds "${CONTEXT_KINDS:-cold,primed_handoff_gap3,lengthy}" \
  --device cuda
