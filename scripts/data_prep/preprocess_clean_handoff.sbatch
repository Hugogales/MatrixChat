#!/bin/bash
# Build a new, coherence-cleaned MELD/AMI/Werewolf dataset. Never overwrite
# the trusted processed_final_20260814_lookback192_allspk baseline.
# Example: sbatch scripts/data_prep/preprocess_clean_handoff.sbatch
# Smoke:   LIMIT=20 sbatch scripts/data_prep/preprocess_clean_handoff.sbatch
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_clean_handoff_prep
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

RAW_DIR="${RAW_DIR:-$HOME/matrixchat/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-$HOME/matrixchat/processed_clean_handoff_20260917_v1}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
LIMIT="${LIMIT:-0}"
CONTEXT_LOOKBACK_COLUMNS="${CONTEXT_LOOKBACK_COLUMNS:-192}"
CONTENT_SUPERVISION_MODE="${CONTENT_SUPERVISION_MODE:-all_speakers}"

export HF_HOME="$RAW_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ "$PROCESSED_DIR" == "$HOME/matrixchat/processed_final_20260814_lookback192_allspk" ]]; then
  echo "[error] refusing to overwrite canonical baseline dataset" >&2
  exit 2
fi
if [[ -e "$PROCESSED_DIR" && "$LIMIT" == 0 ]]; then
  echo "[error] output already exists: $PROCESSED_DIR" >&2
  exit 2
fi

mkdir -p "$PROCESSED_DIR"
python -m data.build_dataset --mode convert \
  --sources meld ami werewolf \
  --raw_dir "$RAW_DIR" \
  --processed_dir "$PROCESSED_DIR" \
  --model_path "$MODEL_PATH" \
  --placeholder_token_id 0 \
  --max_agents 8 \
  --max_flat_len 1536 \
  --pause_threshold_seconds 1 \
  --pause_quantum_seconds 1 \
  --max_pause_columns 8 \
  --context_lookback_columns "$CONTEXT_LOOKBACK_COLUMNS" \
  --content_supervision_mode "$CONTENT_SUPERVISION_MODE" \
  --clean_timed_overlaps true \
  --ami_overlap_keep_tokens 2 \
  --werewolf_overlap_keep_tokens 3 \
  --clean_inter_turn_gap_seconds 0.05 \
  --permute_agents true \
  --limit "$LIMIT" \
  --seed 0

python -m data.inspect \
  --processed_dir "$PROCESSED_DIR" \
  --source ami \
  --num 2 \
  --max_steps 60 \
  --model_path "$MODEL_PATH"
