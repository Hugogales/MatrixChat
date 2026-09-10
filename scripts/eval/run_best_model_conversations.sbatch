#!/bin/bash
# Generate free-form conversations from fixed starters against the project's
# current best checkpoints -- see scripts/eval/run_best_model_conversations.py.
#SBATCH --account=undergrad_research
#SBATCH --job-name=best_model_conversations
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CONTAINER="/data/containers/msoe-tf2x.sif"

singularity exec --nv -B /data:/data "$CONTAINER" python scripts/eval/run_best_model_conversations.py
