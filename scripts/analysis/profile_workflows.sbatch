#!/bin/bash
# Profile ~10 training steps + ~10 held-out eval examples on one GPU.
#SBATCH --account=undergrad_research
#SBATCH --job-name=mc_profile_wf
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-logs/profiling/${STAMP}}"
CHECKPOINT="${CHECKPOINT:-checkpoints/adaptive_hfreq_a075_s176106/best_probe}"
CONTRACT="${CONTRACT:-logs/final_runs_20260827_recontract/development_screening.json}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_DIR"

singularity exec --nv -B /data:/data "$CONTAINER" \
  python scripts/analysis/profile_workflows.py \
    --mode real \
    --training-steps 10 \
    --eval-examples 10 \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --contract "$CONTRACT"

echo "[info] profiling complete: $OUTPUT_DIR/SUMMARY.md"
