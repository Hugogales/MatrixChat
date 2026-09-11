#!/bin/bash
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_search_smoke_ctl
#SBATCH --partition=teaching
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
SEARCH_DIR="${SEARCH_DIR:?Set SEARCH_DIR}"
SEARCH_CONFIG="${SEARCH_CONFIG:-scripts/search/configs/smoke.json}"

python scripts/search/controller.py \
  --search-dir "$SEARCH_DIR" \
  --config "$SEARCH_CONFIG" \
  --poll-seconds 10 \
  --max-runtime-hours 5.5 \
  --no-self-chain
