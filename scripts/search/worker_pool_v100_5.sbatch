#!/bin/bash
# Secondary V100 worker pool used when the manually managed fleet is reduced.
# Runs alongside worker_pool_v100.sbatch against the same atomic ticket queue.
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_month_workers_v100_5
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:5
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
SEARCH_DIR="${SEARCH_DIR:?Set SEARCH_DIR to the persistent search directory}"
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/search/pool_manager.py \
    --search-dir "$SEARCH_DIR" \
    --workers 5 \
    --gpu-type v100 \
    --cpus-per-worker 8 \
    --max-runtime-hours 116 \
    --drain-hours 3 \
    --container "$CONTAINER" \
    --worker-sbatch scripts/search/worker_pool_v100_5.sbatch
