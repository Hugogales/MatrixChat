#!/bin/bash
# =============================================================================
# MatrixChat: real training on MELD (LoRA + freezing) on Rosie.
#
# Prereqs (run once):
#   - Model downloaded: models/Qwen3-4B-Instruct-2507  (see .cursor project memory)
#   - MELD preprocessed: $HOME/matrixchat/processed/meld  (scripts/preprocess_data.sbatch)
#   - PEFT installed in the env/container:  pip install peft
#
# Submit:  sbatch scripts/train_meld.sbatch
# Monitor: squeue -u $USER ; tail -f logs/matrix_meld_*.out ; logs/<run_id>/metrics.jsonl
# =============================================================================
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_meld
#SBATCH --partition=dgx            # V100 DGX-1 (32GB). For H100 (80GB): --partition=dgxh100 --gres=gpu:h100:1
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# main.py auto-assigns run_000NNN and writes logs/run_000NNN/ + checkpoints/run_000NNN/.
# (SLURM console goes to logs/<jobname>_<jobid>.out here.)
echo "host: $(hostname)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not available"

# ---- Environment ------------------------------------------------------------
# Run inside the MSOE Singularity container (has torch+CUDA); torch/transformers/
# datasets/peft come from ~/.local. Read the model from the local copy (offline).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOMs
CONTAINER="/data/containers/msoe-tf2x.sif"

MODEL_PATH="models/Qwen3-4B-Instruct-2507"
PROCESSED_DIR="${PROCESSED_DIR:-$HOME/matrixchat/processed}"

# ---- Recommended hyperparameters (edit as needed) ---------------------------
# Interface / silence
MAX_AGENTS=8
POSITION_MODE="column"          # positions = time: each speaker's utterance stays consecutive (in-distribution
                                # per agent) and the position axis is agent-order invariant. "flat" gives
                                # globally unique positions but stretches each utterance by A (num agents).
SILENCE_TOKEN_ID=151669
SILENCE_LOSS_WEIGHT=0.3         # boosted: strengthen "stay silent when it's not your turn"
DROP_SILENCE=false             # keep false during training (silence is supervised)

# LoRA + freezing (start LoRA-only; optionally unfreeze a few layers)
LORA_ENABLE=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
FREEZE_BASE_MODEL=true
UNFREEZE_FIRST_N_LAYERS=2       # let early layers adapt to the agent-augmented input
UNFREEZE_LAST_N_LAYERS=2       # let late layers adapt turn-taking/silence decisions
                               # (more capacity; watch memory/val_loss. Set both 0 for LoRA-only.)

# Optimization (tuned for a 32GB V100 dgx node; on H100 80GB you can set
# BATCH_SIZE=4, GRADIENT_CHECKPOINTING=false, MAX_FLAT_LEN=4096)
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.0
BATCH_SIZE=1                    # 32GB V100: start at 1; try 2 after a run succeeds
GRADIENT_CHECKPOINTING=true    # essential on 32GB (trades compute for memory)
NUM_EPOCHS=5
NUM_STEPS=0                    # 0 = run full epochs (set e.g. 20 for a quick smoke test)
NUM_WORKERS=8
LOG_EVERY=20
EVAL_EVERY=100
VAL_FRACTION=0.05
MAX_FLAT_LEN=1536              # cap num_agents*T (attention is O(S^2)); the main memory knob
SAMPLE_EVERY=1                # generate a qualitative sample every N epochs (0 disables) -> samples.jsonl
SAMPLE_MAX_NEW_TOKENS=32
TORCH_DTYPE="bfloat16"

# ---- Launch -----------------------------------------------------------------
singularity exec --nv -B /data:/data "$CONTAINER" python main.py \
  --use_tiny_model false \
  --model_name "$MODEL_PATH" \
  --processed_data_dir "$PROCESSED_DIR" \
  --dataset_weights "meld=1.0" \
  --max_agents "$MAX_AGENTS" \
  --position_mode "$POSITION_MODE" \
  --use_agent_embeddings true \
  --attn_mask_mode "matrix_causal" \
  --silence_token_id "$SILENCE_TOKEN_ID" \
  --silence_loss_weight "$SILENCE_LOSS_WEIGHT" \
  --drop_silence "$DROP_SILENCE" \
  --lora_enable "$LORA_ENABLE" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --freeze_base_model "$FREEZE_BASE_MODEL" \
  --unfreeze_first_n_layers "$UNFREEZE_FIRST_N_LAYERS" \
  --unfreeze_last_n_layers "$UNFREEZE_LAST_N_LAYERS" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --batch_size "$BATCH_SIZE" \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --num_epochs "$NUM_EPOCHS" \
  --num_steps "$NUM_STEPS" \
  --num_workers "$NUM_WORKERS" \
  --log_every "$LOG_EVERY" \
  --eval_every "$EVAL_EVERY" \
  --val_fraction "$VAL_FRACTION" \
  --max_flat_len "$MAX_FLAT_LEN" \
  --sample_every "$SAMPLE_EVERY" \
  --sample_max_new_tokens "$SAMPLE_MAX_NEW_TOKENS" \
  --sample_prompts "Hi! Tell me one fun fact." "Stay silent and then respond with your thoughts on his fun fact." \
  --torch_dtype "$TORCH_DTYPE" \
  --device cuda \
  --save_checkpoint true

echo "done."
