#!/bin/bash
# End-to-end smoke: 3 publication-shaped trials at T=1.0 with optimized eval path.
#SBATCH --account=undergrad_research
#SBATCH --job-name=mc_eval_opt_smoke
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=00:45:00
#SBATCH --chdir=/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
CHECKPOINT="checkpoints/adaptive_hfreq_a075_s176106/best_probe"
CONTRACT="logs/final_runs_20260827_recontract/publication_full.json"
OUT="logs/benchmarks/opt_smoke_paired.jsonl"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/benchmarks

START=$(date +%s)
singularity exec --nv -B /data:/data "$CONTAINER" \
  python scripts/eval/run_held_out_continuation_eval.py \
    --contract "$CONTRACT" \
    --output "$OUT" \
    --split development \
    --checkpoint-dir "$CHECKPOINT" \
    --model-path models/Qwen3-4B-Instruct-2507 \
    --device cuda \
    --temperature 1.0 \
    --fractions 0.5 \
    --generation-seeds 0 \
    --max-examples 3 \
    --attn-implementation sdpa \
    --eval-batch-size 4
END=$(date +%s)
ELAPSED=$((END-START))
LINES=$(wc -l < "$OUT")
python3 - <<PY
elapsed = $ELAPSED
lines = $LINES
print(f"[timing] wall_sec={elapsed} trials={lines} per_trial_sec={elapsed/max(1,lines):.2f}")
PY
