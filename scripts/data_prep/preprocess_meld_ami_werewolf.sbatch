#!/bin/bash
# MatrixChat: preprocess MELD + timestamp-aware AMI + Werewolf (private-context,
# timed) into Arrow shards.
#
# Prerequisite (internet/login node):
#   SOURCES="meld ami werewolf" bash scripts/data_prep/download_data.sh
#
# Submit:
#   sbatch scripts/data_prep/preprocess_meld_ami_werewolf.sbatch
# Optional smoke:
#   LIMIT=20 sbatch scripts/data_prep/preprocess_meld_ami_werewolf.sbatch
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_meld_ami_ww_prep
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

RAW_DIR="${RAW_DIR:-$HOME/matrixchat/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-$HOME/matrixchat/processed_meld_ami}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
LIMIT="${LIMIT:-0}"

PLACEHOLDER_TOKEN_ID=0
MAX_AGENTS=8
MAX_FLAT_LEN=1536
PAUSE_THRESHOLD_SECONDS=1
PAUSE_QUANTUM_SECONDS=1
MAX_PAUSE_COLUMNS=8
# Real-history lookback for timed (AMI/Werewolf) chunks -- see ConvertConfig
# docstring in data/convert.py. 0 = old cold-start-every-chunk behavior.
CONTEXT_LOOKBACK_COLUMNS="${CONTEXT_LOOKBACK_COLUMNS:-0}"
CONTENT_SUPERVISION_MODE="${CONTENT_SUPERVISION_MODE:-target_only}"
SEED=0

export HF_HOME="$RAW_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "host:          $(hostname)"
echo "RAW_DIR:       $RAW_DIR"
echo "PROCESSED_DIR: $PROCESSED_DIR"
mkdir -p "$PROCESSED_DIR"

python -m data.build_dataset --mode peek \
  --sources meld ami werewolf \
  --raw_dir "$RAW_DIR" \
  --peek_n 1

python -m data.build_dataset --mode convert \
  --sources meld ami werewolf \
  --raw_dir "$RAW_DIR" \
  --processed_dir "$PROCESSED_DIR" \
  --model_path "$MODEL_PATH" \
  --placeholder_token_id "$PLACEHOLDER_TOKEN_ID" \
  --max_agents "$MAX_AGENTS" \
  --max_flat_len "$MAX_FLAT_LEN" \
  --pause_threshold_seconds "$PAUSE_THRESHOLD_SECONDS" \
  --pause_quantum_seconds "$PAUSE_QUANTUM_SECONDS" \
  --max_pause_columns "$MAX_PAUSE_COLUMNS" \
  --context_lookback_columns "$CONTEXT_LOOKBACK_COLUMNS" \
  --content_supervision_mode "$CONTENT_SUPERVISION_MODE" \
  --permute_agents true \
  --limit "$LIMIT" \
  --seed "$SEED"

python -m data.inspect \
  --processed_dir "$PROCESSED_DIR" \
  --source werewolf \
  --num 1 \
  --max_steps 60 \
  --model_path "$MODEL_PATH"

echo "done. Manifest: $PROCESSED_DIR/manifest.json"
