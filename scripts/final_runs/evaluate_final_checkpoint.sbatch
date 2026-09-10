#!/bin/bash
# Evaluate one completed stage checkpoint against the frozen held-out
# development contract, then emit publication-format raw data/statistics/plots.
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_final_eval
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

CONTRACT="${CONTRACT:?Set CONTRACT to the frozen development contract}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a completed checkpoint ROOT (checkpoints/<run_name>)}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for raw and analyzed artifacts}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
TEMPERATURE="${TEMPERATURE:-0.0}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
# "auto" (default): prefer best_probe/ (best actual turn-taking behavior,
# see training/race.py::probe_quality_score) > root (best val_loss) > last/
# (terminal step). Set to "best_probe" / "root" / "last" to force one.
CHECKPOINT_PREFER="${CHECKPOINT_PREFER:-auto}"
TRUST_FROZEN_CONTRACT="${TRUST_FROZEN_CONTRACT:-0}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUTPUT_DIR"
RAW="$OUTPUT_DIR/paired_continuations.jsonl"

resolve_checkpoint_dir() {
  local root="$1" prefer="$2"
  if [ "$prefer" = "best_probe" ] || { [ "$prefer" = "auto" ] && [ -f "$root/best_probe/model_state.pt" ]; }; then
    echo "$root/best_probe"
  elif [ "$prefer" = "last" ] || { [ "$prefer" = "auto" ] && ! [ -f "$root/model_state.pt" ]; }; then
    echo "$root/last"
  else
    echo "$root"
  fi
}

RESOLVED_CHECKPOINT_DIR="$(resolve_checkpoint_dir "$CHECKPOINT_DIR" "$CHECKPOINT_PREFER")"
if [ ! -f "$RESOLVED_CHECKPOINT_DIR/model_state.pt" ]; then
  echo "[error] no model_state.pt under resolved checkpoint dir: $RESOLVED_CHECKPOINT_DIR" >&2
  echo "[error] checked CHECKPOINT_DIR=$CHECKPOINT_DIR with CHECKPOINT_PREFER=$CHECKPOINT_PREFER" >&2
  exit 1
fi
echo "[info] resolved checkpoint dir (prefer=$CHECKPOINT_PREFER): $RESOLVED_CHECKPOINT_DIR"

TRUST_ARGS=()
if [[ "$TRUST_FROZEN_CONTRACT" == "1" ]]; then
  TRUST_ARGS+=(--trust-frozen-contract)
fi

singularity exec --nv -B /data:/data "$CONTAINER" \
  python scripts/eval/run_held_out_continuation_eval.py \
    --contract "$CONTRACT" \
    --output "$RAW" \
    --split development \
    --checkpoint-dir "$RESOLVED_CHECKPOINT_DIR" \
    --model-path "$MODEL_PATH" \
    --device cuda \
    --temperature "$TEMPERATURE" \
    --fractions 0.5 \
    --generation-seeds "$GENERATION_SEEDS" \
    --attn-implementation "$ATTN_IMPLEMENTATION" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --summary "$OUTPUT_DIR/eval_summary.json" \
    --metadata "$OUTPUT_DIR/eval_metadata.json" \
    "${TRUST_ARGS[@]}"

singularity exec -B /data:/data "$CONTAINER" \
  python scripts/analysis/analyze_held_out_continuations.py "$RAW" \
    --output-dir "$OUTPUT_DIR/analysis" \
    --bootstrap-resamples 10000 \
    --seed 0
