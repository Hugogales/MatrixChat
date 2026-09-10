#!/bin/bash
# Blinded five-axis LLM judge on the existing 997 paired development
# continuations. CPU-only; talks to the shared judge endpoint. Resume-safe.
# Does not touch final_test.
#SBATCH --account=undergrad_research
#SBATCH --job-name=pub_judge_c106s2106
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
INPUT="${INPUT:-logs/final_runs_20260814/full_development/final3_h100_c106_s2106_t1/paired_continuations.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/final_runs_20260817_publication/final3_h100_c106_s2106_t1}"
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
    --seed 0
