#!/bin/bash
# =============================================================================
# MatrixChat Stage-2 self-distillation (GPU): generate the qwen_distill source.
#
# Runs Qwen3-4B on a prompt set and stores its own responses as the retention /
# anti-forgetting tier. Output JSONL lands under $RAW_DIR/qwen_distill so the
# preprocess job can convert it like any other source.
#
# Submit:  sbatch scripts/data_prep/distill_qwen.sbatch
# =============================================================================
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_distill
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

RAW_DIR="${RAW_DIR:-$HOME/matrixchat/raw}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
PROMPTS_FILE="${PROMPTS_FILE:-$RAW_DIR/distill_prompts.txt}"
OUT_DIR="$RAW_DIR/qwen_distill"
MAX_PROMPTS=20000
MAX_NEW_TOKENS=512
BATCH_SIZE=8
TEMPERATURE=0.7

export HF_HOME="$RAW_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "host: $(hostname)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not available"
mkdir -p "$OUT_DIR"

python -m data.distill \
  --model_path "$MODEL_PATH" \
  --prompts_file "$PROMPTS_FILE" \
  --out_dir "$OUT_DIR" \
  --max_prompts "$MAX_PROMPTS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --batch_size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE"

echo "done -> $OUT_DIR/qwen_distill.jsonl"
