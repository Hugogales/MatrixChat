#!/bin/bash
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_search_smoke_gpu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
SEARCH_DIR="${SEARCH_DIR:?Set SEARCH_DIR}"
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
SMOKE_WORKERS="${SMOKE_WORKERS:-2}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/search/pool_manager.py \
  --search-dir "$SEARCH_DIR" \
    --workers "$SMOKE_WORKERS" \
  --max-runtime-hours 5.5 \
  --drain-hours 0.25 \
  --container "$CONTAINER" \
  --no-self-chain
