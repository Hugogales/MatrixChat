#!/usr/bin/env bash
# Launch two V100 manual explore arms from the widened-search plan.
set -euo pipefail

ROOT="/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat"
cd "$ROOT"
mkdir -p logs/slurm

PROCESSED_DIR="${PROCESSED_DIR:-$HOME/matrixchat/processed_final_20260814_lookback192_allspk}"
LEADER_CKPT="checkpoints/final40_h100_c106_s176106"

echo "Submitting structure_mix_warm..."
STRUCT_JOB=$(sbatch --parsable --partition=dgx --gres=gpu:1 \
  --job-name=mc_explore_struct_warm \
  --export=ALL,RUN_NAME=explore_structure_mix_warm_s176106,SEED=176106,TURN_REWARD_MODE=balanced_ce,FLOOR_CONTROL_ADAPTIVE_WEIGHT_ALPHA=0.75,DATASET_WEIGHTS="bazinga=0.4,when2speak=0.35,molweni=0.25",INIT_FROM="$LEADER_CKPT",INIT_CHECKPOINT_PREFER=best_probe,QUALITY_OVERSAMPLE_FACTOR=2.0,PROCESSED_DIR="$PROCESSED_DIR" \
  scripts/training/train_dynstate_ablation.sbatch)

echo "Submitting arch_relation_bias..."
ARCH_JOB=$(sbatch --parsable --partition=dgx --gres=gpu:1 \
  --job-name=mc_explore_rel_bias \
  --export=ALL,RUN_NAME=explore_arch_relation_bias_s176106,SEED=176106,TURN_REWARD_MODE=balanced_ce,FLOOR_CONTROL_ADAPTIVE_WEIGHT_ALPHA=0.75,DATASET_WEIGHTS="ami=0.5,meld=0.15,werewolf=0.35",INIT_FROM="$LEADER_CKPT",INIT_CHECKPOINT_PREFER=best_probe,AGENT_SAME_ATTENTION_BIAS=true,AGENT_RELATION_BIAS_MODE=same_diff,PROCESSED_DIR="$PROCESSED_DIR" \
  scripts/training/train_dynstate_ablation.sbatch)

echo "structure_mix_warm=$STRUCT_JOB arch_relation_bias=$ARCH_JOB"
