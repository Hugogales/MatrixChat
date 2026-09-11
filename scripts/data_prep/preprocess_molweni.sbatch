#!/bin/bash
# MatrixChat: preprocess Molweni (Ubuntu multi-party chat, content_bearing=
# False, grammar/spelling + PTB-detokenization cleaning applied by default
# since 2026-08-24 -- see data/adapters/_text_cleaning.py) into Arrow shards.
#
# ADDITIVE ONLY: writes a new "molweni" subdir + merges one manifest.json key
# into the target --processed_dir. Does NOT touch/rewrite the existing
# meld/ami/werewolf subdirs or their manifest entries (data/manifest.py's
# update_source only ever inserts/replaces its own source key). Safe to point
# this at the frozen `processed_final_20260814_lookback192_allspk` leader
# directory for a same-tree data-ablation comparison.
#
# Prerequisite (internet/login node, only if raw/molweni/ doesn't exist yet):
#   SOURCES="molweni" bash scripts/data_prep/download_data.sh
#
# Submit:
#   PROCESSED_DIR=$HOME/matrixchat/processed_final_20260814_lookback192_allspk \
#     sbatch scripts/data_prep/preprocess_molweni.sbatch
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_molweni_prep
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
  --sources molweni \
  --raw_dir "$RAW_DIR" \
  --peek_n 2

python -m data.build_dataset --mode convert \
  --sources molweni \
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
  --source molweni \
  --num 1 \
  --max_steps 60 \
  --model_path "$MODEL_PATH"

echo "done. Manifest: $PROCESSED_DIR/manifest.json"
