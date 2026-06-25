#!/bin/bash
# =============================================================================
# MatrixChat smoke-training SLURM script for Rosie (MSOE HPC).
#
# This is a SMOKE TEST, not a full training job. All hyperparameters are set as
# bash variables below and passed to main.py as CLI args (no key settings hidden
# in Python defaults).
#
# Submit with:   sbatch scripts/train_matrix_qwen.sbatch
# Monitor with:  squeue -u $USER
# Cancel with:   scancel <jobid>
# Inspect GPUs:  sinfo -o "%P %N %G"
# =============================================================================

# ---- SLURM directives -------------------------------------------------------
#SBATCH --account=undergrad_research          # researcher account: unlimited jobs, up to 1 week
#SBATCH --job-name=matrix_qwen
#SBATCH --partition=teaching                  # default student partition (4x Tesla T4 nodes)
#SBATCH --gres=gpu:1                          # request 1 GPU via GRES
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00                     # smoke default. Researcher max is 7-00:00:00 (1 week)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# -----------------------------------------------------------------------------
# Rosie GPU tiers (confirm exact partition / GRES names with: sinfo -o "%P %N %G")
#   teaching / HighMem -> 4x Tesla T4 per node        (smoke + small jobs)
#   DGX-1              -> 8x Tesla V100-SXM2 per node  (e.g. --gres=gpu:v100:N)
#   DGX H100 (2024)    -> 8x H100 per node, NVLink     (e.g. --gres=gpu:h100:N)
#
# For the REAL Qwen/Qwen3-4B-Instruct-2507 run, target the DGX H100 nodes and set
# USE_TINY_MODEL=false, e.g.:
#   #SBATCH --partition=dgxh100   # confirm real name with sinfo
#   #SBATCH --gres=gpu:h100:1
#   #SBATCH --time=7-00:00:00
# -----------------------------------------------------------------------------

set -euo pipefail

echo "host: $(hostname)"
echo "date: $(date)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not available"

# ---- Environment ------------------------------------------------------------
# On Rosie, run inside a Singularity / NGC PyTorch container or activate a conda
# environment, then install the project requirements. Adjust to your setup:
#   module load singularity
#   conda activate matrixchat
# pip install -r requirements.txt

# ---- Hyperparameters (edit these) -------------------------------------------
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
USE_TINY_MODEL=true
MAX_AGENTS=8
NUM_AGENTS=4
SEQ_LEN=16
BATCH_SIZE=2
POSITION_MODE="flat"            # choices: flat, column
USE_AGENT_EMBEDDINGS=true
USE_CHANNEL_EMBEDDINGS=false    # optional first pass
ATTN_MASK_MODE="matrix_causal"  # choices: none, matrix_causal
QUERY_TOKEN_ID=0
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.0
NUM_STEPS=3
LORA_ENABLE=false
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
FREEZE_BASE_MODEL=true
UNFREEZE_LAST_N_LAYERS=0
TORCH_DTYPE="bfloat16"
DEVICE="cuda"

# ---- Launch -----------------------------------------------------------------
python main.py \
  --model_name "$MODEL_NAME" \
  --use_tiny_model "$USE_TINY_MODEL" \
  --max_agents "$MAX_AGENTS" \
  --num_agents "$NUM_AGENTS" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --position_mode "$POSITION_MODE" \
  --use_agent_embeddings "$USE_AGENT_EMBEDDINGS" \
  --use_channel_embeddings "$USE_CHANNEL_EMBEDDINGS" \
  --attn_mask_mode "$ATTN_MASK_MODE" \
  --query_token_id "$QUERY_TOKEN_ID" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --num_steps "$NUM_STEPS" \
  --lora_enable "$LORA_ENABLE" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --freeze_base_model "$FREEZE_BASE_MODEL" \
  --unfreeze_last_n_layers "$UNFREEZE_LAST_N_LAYERS" \
  --torch_dtype "$TORCH_DTYPE" \
  --device "$DEVICE"

echo "done."
