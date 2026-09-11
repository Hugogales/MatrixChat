#!/usr/bin/env bash
# Resume optimized publication eval with metadata for safe resume.
set -euo pipefail

ROOT="/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat"
cd "$ROOT"
mkdir -p logs/slurm

SLOW_DIR="paper_results_adaptive_a075_20260830"
OUTPUT_DIR="${OUTPUT_DIR:-$SLOW_DIR}"
CONTRACT="logs/final_runs_20260827_recontract/publication_full.json"
CHECKPOINT_DIR="checkpoints/adaptive_hfreq_a075_s176106"

EVAL_JOB=$(sbatch --parsable \
  --export=ALL,CONTRACT="$CONTRACT",CHECKPOINT_DIR="$CHECKPOINT_DIR",OUTPUT_DIR="$OUTPUT_DIR",TEMPERATURE=1.0,GENERATION_SEEDS=0,ATTN_IMPLEMENTATION=sdpa,EVAL_BATCH_SIZE=4,CHECKPOINT_PREFER=auto \
  scripts/final_runs/evaluate_final_checkpoint.sbatch)

POST_JOB=$(sbatch --parsable --dependency=afterok:"$EVAL_JOB" \
  --export=ALL,INPUT_DIR="$OUTPUT_DIR",BASELINE_DIR=paper_results_176106_20260819,OUTPUT_DIR="$OUTPUT_DIR" \
  scripts/final_runs/publication_bundle_postprocess.sbatch)

echo "fast_eval=$EVAL_JOB postprocess=$POST_JOB"
