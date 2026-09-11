#!/bin/bash
# Vanilla Qwen3-4B-Instruct round-robin baseline on the frozen development
# contract. Unfinetuned CausalLM, no MatrixQwen/LoRA. Matched-length
# continuation, temperature 1.0. Pin to idle node3; do not land on node1.
# Does not touch final_test.
#SBATCH --account=undergrad_research
#SBATCH --job-name=qwen_rr_t1
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=dh-dgx1-1
#SBATCH --nodelist=dh-dgx1-3
#SBATCH --chdir=/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

CONTRACT="${CONTRACT:-logs/final_runs_20260814/contracts/publication_full.json}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for raw artifacts}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
TEMPERATURE="${TEMPERATURE:-1.0}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUTPUT_DIR"
RAW="$OUTPUT_DIR/paired_continuations.jsonl"

EXTRA=()
if [[ -n "${MAX_EXAMPLES}" ]]; then
  EXTRA+=(--max-examples "$MAX_EXAMPLES")
fi

singularity exec --nv -B /data:/data "$CONTAINER" \
  python scripts/eval/run_held_out_continuation_eval.py \
    --contract "$CONTRACT" \
    --output "$RAW" \
    --split development \
    --protocol round_robin \
    --round-robin-start next_after_cut_reference \
    --model-path "$MODEL_PATH" \
    --device cuda \
    --temperature "$TEMPERATURE" \
    --fractions 0.5 \
    --generation-seeds "$GENERATION_SEEDS" \
    --summary "$OUTPUT_DIR/eval_summary.json" \
    --metadata "$OUTPUT_DIR/eval_metadata.json" \
    "${EXTRA[@]}"
