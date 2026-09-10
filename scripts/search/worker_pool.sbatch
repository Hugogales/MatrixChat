#!/bin/bash
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_month_workers
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:5
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
SEARCH_DIR="${SEARCH_DIR:?Set SEARCH_DIR to the persistent search directory}"
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
# Overridable so this pool can be right-sized to whatever H100 capacity is
# genuinely free right now (the dgxh100 partition is shared with other
# users) without needing a one-off sbatch variant. Must match the actual
# number of GPUs the job was submitted with (--gres/--ntasks), which sbatch
# CLI overrides can adjust independently of this script default.
WORKERS="${WORKERS:-5}"

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
    --worker-sbatch scripts/search/worker_pool.sbatch
