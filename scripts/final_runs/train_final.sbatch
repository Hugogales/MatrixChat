#!/bin/bash
# Generic one-GPU final-run training stage. Override partition/GRES/memory with
# sbatch flags for H100 launches.
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_final_train
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=2-12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:?Set CONFIG_PATH to a frozen final-run JSON config}"
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host: $(hostname)"
echo "config: $CONFIG_PATH"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

singularity exec --nv -B /data:/data "$CONTAINER" \
  python scripts/final_runs/run_training_from_config.py --config "$CONFIG_PATH"
