#!/bin/bash
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_month_watchdog
#SBATCH --partition=teaching
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
SEARCH_DIR="${SEARCH_DIR:?Set SEARCH_DIR to the persistent search directory}"

python scripts/search/watchdog.py \
  --search-dir "$SEARCH_DIR" \
  --interval-seconds 300 \
  --lease-seconds 900 \
  --max-runtime-hours 116 \
  --controller-sbatch scripts/search/controller.sbatch \
  --watchdog-sbatch scripts/search/watchdog.sbatch
