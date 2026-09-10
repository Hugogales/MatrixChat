#!/bin/bash
# Correctness + speed benchmark for eval optimizations (KV cache, SDPA, batching).
#SBATCH --account=undergrad_research
#SBATCH --job-name=mc_eval_speed_bench
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
CHECKPOINT="${CHECKPOINT:-checkpoints/adaptive_hfreq_a075_s176106/best_probe}"
CONTRACT="${CONTRACT:-logs/final_runs_20260827_recontract/development_screening.json}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${OUTPUT:-logs/benchmarks/eval_speedups_${STAMP}.json}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$(dirname "$OUTPUT")"

singularity exec --nv -B /data:/data "$CONTAINER" \
  python scripts/eval/benchmark_eval_speedups.py \
    --checkpoint-dir "$CHECKPOINT" \
    --examples-jsonl paper_results_adaptive_a075_20260830/paired_continuations.jsonl \
    --device cuda \
    --max-examples 5 \
    --eval-batch-size 4 \
    --continuation-columns 96 \
    --repeats 2 \
    --output "$OUTPUT"

echo "[info] benchmark complete: $OUTPUT"
