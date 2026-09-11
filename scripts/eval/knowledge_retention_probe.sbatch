#!/bin/bash
# Text-native generic-content retention diagnostic for a MatrixChat checkpoint.
# Submit:
#   CKPT_DIR=checkpoints/<run>/rung1_step1500 \
#   OUTPUT=logs/probe_results/knowledge_<run>_step1500.json \
#   sbatch scripts/eval/knowledge_retention_probe.sbatch
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_knowledge_probe
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --exclude=dh-dgx1-1
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the checkpoint directory}"
OUTPUT="${OUTPUT:?Set OUTPUT to the result JSON path}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

singularity exec --nv -B /data:/data "$CONTAINER" \
  python scripts/eval/knowledge_retention_probe.py \
    --checkpoint_dir "$CKPT_DIR" \
    --model_path "$MODEL_PATH" \
    --device cuda \
    --output "$OUTPUT"
