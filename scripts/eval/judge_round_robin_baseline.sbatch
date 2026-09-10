#!/bin/bash
# Blinded Llama 4 Scout judge for the vanilla Qwen round-robin JSONL,
# then cluster-weighted analysis. CPU-only. Does not touch final_test.
#SBATCH --account=undergrad_research
#SBATCH --job-name=qwen_rr_judge
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
INPUT="${INPUT:?Set INPUT to the round-robin paired_continuations.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for judged artifacts}"
JUDGED="$OUTPUT_DIR/paired_continuations_judged.jsonl"

mkdir -p "$OUTPUT_DIR"
singularity exec -B /data:/data "$CONTAINER" \
  python scripts/eval/judge_held_out_jsonl.py \
    --input "$INPUT" \
    --output "$JUDGED"

singularity exec -B /data:/data "$CONTAINER" \
  python scripts/analysis/analyze_held_out_continuations.py "$JUDGED" \
    --output-dir "$OUTPUT_DIR/analysis_judged" \
    --bootstrap-resamples 10000 \
    --seed 0 \
    --recompute-metrics \
    --exclude-sources meld
