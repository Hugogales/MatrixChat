#!/bin/bash
# =============================================================================
# MatrixChat Stage-2 data PREPROCESS / TOKENIZE (CPU, offline).
#
# Heavy CPU lifting: adapters -> matrix conversion (tokenize, turn-major layout,
# silence/EOS labels, agent permutation) -> Arrow shards + manifest on /data.
# Run AFTER scripts/download_data.sh has populated $RAW_DIR.
#
# Submit:  sbatch scripts/preprocess_data.sbatch
# =============================================================================
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_preprocess
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- Paths / hyperparameters (edit) -----------------------------------------
# $HOME is writable and mounted on all nodes; override with /data/ai_club/... if
# you prefer the shared pool (you are not in the 'ur' group).
RAW_DIR="${RAW_DIR:-$HOME/matrixchat/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-$HOME/matrixchat/processed}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
SOURCES="${SOURCES:-molweni meld werewolf conversation_chronicles qwen_distill}"
SILENCE_TOKEN_ID=151669
MAX_AGENTS=8
MIN_TURN_GAP=0            # all-silence columns between turns (no EOS delimiter)
MAX_TURN_GAP=3           # every silent cell is supervised; strength tuned at train time
PERMUTE_AGENTS=true
NUM_PROC=16
LIMIT="${LIMIT:-0}"        # 0 = all; set small (e.g. 50) for a quick test run
SEED=0

# Read from the local /data copy only; never hit the network on a compute node.
export HF_HOME="$RAW_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "host:           $(hostname)"
echo "RAW_DIR:        $RAW_DIR"
echo "PROCESSED_DIR:  $PROCESSED_DIR"
mkdir -p "$PROCESSED_DIR"

python -m data.build_dataset --mode convert \
  --sources $SOURCES \
  --raw_dir "$RAW_DIR" \
  --processed_dir "$PROCESSED_DIR" \
  --model_path "$MODEL_PATH" \
  --silence_token_id "$SILENCE_TOKEN_ID" \
  --max_agents "$MAX_AGENTS" \
  --min_turn_gap "$MIN_TURN_GAP" \
  --max_turn_gap "$MAX_TURN_GAP" \
  --permute_agents "$PERMUTE_AGENTS" \
  --num_proc "$NUM_PROC" \
  --limit "$LIMIT" \
  --seed "$SEED"

echo "done. Manifest: $PROCESSED_DIR/manifest.json"
