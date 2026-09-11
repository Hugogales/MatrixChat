#!/bin/bash
# Broad stochastic handoff probe (the "big probe"): prompts x agent-counts x
# seeds x context kinds, with chain continuation. Temperature 0.8 matches the
# historical verification contract. Does not touch final_test.
#SBATCH --account=undergrad_research
#SBATCH --job-name=pub_probe_c106s2106
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
CKPT_DIR="${CKPT_DIR:-checkpoints/final3_h100_c106_s2106/last}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
OUTPUT="${OUTPUT:-logs/final_runs_20260817_publication/final3_h100_c106_s2106_t1/broad_probe.json}"

mkdir -p "$(dirname "$OUTPUT")"
singularity exec --nv -B /data:/data "$CONTAINER" \
  python scripts/eval/demo_handoff_variation.py \
    --checkpoint_dir "$CKPT_DIR" \
    --model_path "$MODEL_PATH" \
    --output "$OUTPUT" \
    --device cuda
