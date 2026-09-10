#!/bin/bash
# Secondary H100 worker pool: adds 2 more workers to an existing H100 search
# (e.g. searches/h100_2026_08_maxmem) alongside its primary 5-worker
# worker_pool.sbatch, using genuinely idle H100 capacity rather than taking
# anything from the shared DGX/V100 fleet.
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_h100mm_workers_extra
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
SEARCH_DIR="${SEARCH_DIR:?Set SEARCH_DIR to the persistent search directory}"
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
WORKERS="${WORKERS:-2}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/search/pool_manager.py \
    --search-dir "$SEARCH_DIR" \
    --workers "$WORKERS" \
    --max-runtime-hours 116 \
    --drain-hours 3 \
    --container "$CONTAINER" \
    --worker-sbatch scripts/search/worker_pool_h100_extra.sbatch
