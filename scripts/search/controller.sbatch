#!/bin/bash
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_month_controller
#SBATCH --partition=teaching
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
SEARCH_DIR="${SEARCH_DIR:?Set SEARCH_DIR to the persistent search directory}"
SEARCH_CONFIG="${SEARCH_CONFIG:-}"

ARGS=(
  --search-dir "$SEARCH_DIR"
  --poll-seconds 30
  --max-runtime-hours 116
  --stale-lease-hours 3
  --controller-sbatch scripts/search/controller.sbatch
)
if [[ -n "$SEARCH_CONFIG" ]]; then
  ARGS+=(--config "$SEARCH_CONFIG")
fi

python scripts/search/controller.py "${ARGS[@]}"
