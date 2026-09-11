#!/bin/bash
# MatrixChat: preprocess Bazinga! (Big Bang Theory transcripts, content_bearing=
# True, PTB-detokenized via data/adapters/_text_cleaning.py, adapter rewritten
# 2026-08-25 to bypass the removed datasets.load_dataset script loader -- see
# data/adapters/bazinga.py) into Arrow shards.
#
# ADDITIVE ONLY: writes a new "bazinga" subdir + merges one manifest.json key
# into the target --processed_dir, matching preprocess_molweni.sbatch's
# established safe pattern (does NOT touch/rewrite existing subdirs/manifest
# entries). Safe to point this at the frozen `processed_final_20260814_
# lookback192_allspk` leader directory for a same-tree data-ablation
# comparison (train with vs without bazinga in DATASET_WEIGHTS).
#
# Prerequisite: raw/bazinga/ already downloaded via huggingface_hub (done
# 2026-08-25 using the user's HF token) -- HF_HUB_OFFLINE below assumes this.
#
# Submit:
#   PROCESSED_DIR=$HOME/matrixchat/processed_final_20260814_lookback192_allspk \
#     sbatch scripts/data_prep/preprocess_bazinga.sbatch
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_bazinga_prep
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

RAW_DIR="${RAW_DIR:-$HOME/matrixchat/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-$HOME/matrixchat/processed_final_20260814_lookback192_allspk}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
LIMIT="${LIMIT:-0}"

PLACEHOLDER_TOKEN_ID=0
MAX_AGENTS=8
MAX_FLAT_LEN=1536
CONTEXT_LOOKBACK_COLUMNS=192
CONTENT_SUPERVISION_MODE=all_speakers
SEED=0

export HF_HOME="$RAW_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "host:          $(hostname)"
echo "RAW_DIR:       $RAW_DIR"
echo "PROCESSED_DIR: $PROCESSED_DIR"
mkdir -p "$PROCESSED_DIR"

python -m data.build_dataset --mode peek \
  --sources bazinga \
  --raw_dir "$RAW_DIR" \
  --peek_n 2

python -m data.build_dataset --mode convert \
  --sources bazinga \
  --raw_dir "$RAW_DIR" \
  --processed_dir "$PROCESSED_DIR" \
  --model_path "$MODEL_PATH" \
  --placeholder_token_id "$PLACEHOLDER_TOKEN_ID" \
  --max_agents "$MAX_AGENTS" \
  --max_flat_len "$MAX_FLAT_LEN" \
  --context_lookback_columns "$CONTEXT_LOOKBACK_COLUMNS" \
  --content_supervision_mode "$CONTENT_SUPERVISION_MODE" \
  --permute_agents true \
  --limit "$LIMIT" \
  --seed "$SEED"

python -m data.inspect \
  --processed_dir "$PROCESSED_DIR" \
  --source bazinga \
  --num 1 \
  --max_steps 60 \
  --model_path "$MODEL_PATH"

echo "done. Manifest: $PROCESSED_DIR/manifest.json"
