#!/bin/bash
# V100 variant of worker_pool.sbatch: 7 one-GPU workers on the `dgx`
# partition instead of 5 on `dgxh100`. See scripts/search/controller.py's
# gpu_memory_gb-gated memory clamp -- candidates sampled/resumed against a
# search using this pool MUST have config["gpu_memory_gb"] <= 40 (32GB V100)
# so batch_size never exceeds 1.
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_month_workers_v100
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:7
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
SEARCH_DIR="${SEARCH_DIR:?Set SEARCH_DIR to the persistent search directory}"
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
# Overridable so this pool can be temporarily right-sized (e.g. shrunk to free
# GPUs for a manual architecture sweep) without needing a one-off sbatch
# variant. Must match the actual number of GPUs the job was submitted with
# (--gres/--ntasks), which sbatch CLI overrides can adjust independently of
# this script default.
WORKERS="${WORKERS:-7}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/search/pool_manager.py \
    --search-dir "$SEARCH_DIR" \
    --workers "$WORKERS" \
    --gpu-type v100 \
    --cpus-per-worker 8 \
    --max-runtime-hours 116 \
    --drain-hours 3 \
    --container "$CONTAINER" \
    --worker-sbatch scripts/search/worker_pool_v100.sbatch
