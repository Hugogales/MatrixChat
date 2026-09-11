#!/bin/bash
# Generate one free-form conversation from an environment-selected checkpoint.
# Example:
#   CHECKPOINT_DIR=checkpoints/final40_h100_c106_s176106 \
#   SEED_TEXT="What should we build next?" \
#     sbatch scripts/eval/run_best_model_conversations.sbatch
#SBATCH --account=undergrad_research
#SBATCH --job-name=best_model_conversations
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/final40_h100_c106_s176106}"
SEED_TEXT="${SEED_TEXT:-What should we work on together today?}"
NUM_AGENTS="${NUM_AGENTS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-120}"
TEMPERATURE="${TEMPERATURE:-0.8}"
ACTIVITY_THRESHOLD="${ACTIVITY_THRESHOLD:-0.5}"
SEED="${SEED:-0}"
OUTPUT="${OUTPUT:-logs/best_model_conversations/conversation_${SLURM_JOB_ID}.json}"

mkdir -p "$(dirname "$OUTPUT")"

singularity exec --nv -B /data:/data "$CONTAINER" \
  python scripts/eval/demo_freeform_conversation.py \
    --model_path "$MODEL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --seed_text "$SEED_TEXT" \
    --num_agents "$NUM_AGENTS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --activity_threshold "$ACTIVITY_THRESHOLD" \
    --seed "$SEED" \
    --device cuda \
    --output "$OUTPUT"
