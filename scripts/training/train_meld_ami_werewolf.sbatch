#!/bin/bash
# MatrixChat: train on MELD + timed AMI meetings + private-context Werewolf,
# EQUALLY weighted (~1/3 each), on a 32GB V100.
#
# Prerequisite:
#   sbatch scripts/data_prep/preprocess_meld_ami_werewolf.sbatch
#
# Submit:
#   sbatch scripts/training/train_meld_ami_werewolf.sbatch
#SBATCH --account=undergrad_research
#SBATCH --job-name=matrix_meld_ami_ww
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

echo "host: $(hostname)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONTAINER="/data/containers/msoe-tf2x.sif"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
PROCESSED_DIR="${PROCESSED_DIR:-$HOME/matrixchat/processed_meld_ami}"
RUN_NAME="${RUN_NAME:-}"
RESUME_FROM="${RESUME_FROM:-}"
INIT_FROM="${INIT_FROM:-}"
INIT_CHECKPOINT_PREFER="${INIT_CHECKPOINT_PREFER:-auto}"
SEED="${SEED:-0}"

# Matrix and Stage-A activity objective.
MAX_AGENTS=8
POSITION_MODE="column"
PLACEHOLDER_TOKEN_ID=0
AGENT_ATTENTION_MODE="${AGENT_ATTENTION_MODE:-none}"
AGENT_ATTENTION_REPRESENTATION="${AGENT_ATTENTION_REPRESENTATION:-learned}"
AGENT_ATTENTION_DIM="${AGENT_ATTENTION_DIM:-32}"
AGENT_ATTENTION_INIT_STD="${AGENT_ATTENTION_INIT_STD:-0.001}"
AGENT_SAME_ATTENTION_BIAS="${AGENT_SAME_ATTENTION_BIAS:-false}"
AGENT_SAME_ATTENTION_BIAS_INIT="${AGENT_SAME_ATTENTION_BIAS_INIT:-0.0}"
# Generalized relation bias (independent of AGENT_SAME_ATTENTION_BIAS above);
# see model/agent_attention.py's AgentRelationBias.
AGENT_RELATION_BIAS_MODE="${AGENT_RELATION_BIAS_MODE:-none}"
AGENT_RELATION_BIAS_REPRESENTATION="${AGENT_RELATION_BIAS_REPRESENTATION:-learned}"
AGENT_RELATION_BIAS_DIM="${AGENT_RELATION_BIAS_DIM:-8}"
AGENT_RELATION_BIAS_INIT_STD="${AGENT_RELATION_BIAS_INIT_STD:-0.001}"
AGENT_RELATION_BIAS_SAME_INIT="${AGENT_RELATION_BIAS_SAME_INIT:-0.0}"
AGENT_RELATION_BIAS_DIFF_INIT="${AGENT_RELATION_BIAS_DIFF_INIT:-0.0}"
# DialogueRNN-inspired dynamic per-agent state; see model/dynamic_agent_state.py.
AGENT_DYNAMIC_STATE_MODE="${AGENT_DYNAMIC_STATE_MODE:-none}"
AGENT_DYNAMIC_STATE_DIM="${AGENT_DYNAMIC_STATE_DIM:-64}"
AGENT_DYNAMIC_STATE_INIT_STD="${AGENT_DYNAMIC_STATE_INIT_STD:-0.001}"
LAMBDA_CONTENT="${LAMBDA_CONTENT:-1.0}"
LAMBDA_ACTIVITY="${LAMBDA_ACTIVITY:-1.0}"
LAMBDA_REWARD="${LAMBDA_REWARD:-0.05}"
LAMBDA_SAME_HANDOFF="${LAMBDA_SAME_HANDOFF:-0.0}"
# 0.5 = balanced (default); >0.5 penalizes missed-speak mistakes harder than
# false-speak, biasing predicted P(speak) upward.
ACTIVITY_POS_WEIGHT="${ACTIVITY_POS_WEIGHT:-0.5}"
# "linear" (default, exact historical behavior) or "balanced_ce" (2026-08-24
# rearchitecture -- see model/turn_reward.py's floor_control_reward). All
# SPEAK_*/SILENCE_*/OVERLAP_*/HANDOFF_BONUS_WEIGHT values below are ignored
# when TURN_REWARD_MODE=balanced_ce.
TURN_REWARD_MODE="${TURN_REWARD_MODE:-linear}"
FLOOR_CONTROL_REF_SILENCE="${FLOOR_CONTROL_REF_SILENCE:-0.0784}"
FLOOR_CONTROL_REF_SAME="${FLOOR_CONTROL_REF_SAME:-0.7805}"
FLOOR_CONTROL_REF_HANDOFF="${FLOOR_CONTROL_REF_HANDOFF:-0.0675}"
FLOOR_CONTROL_REF_OVERLAP="${FLOOR_CONTROL_REF_OVERLAP:-0.0735}"
SPEAK_GRACE="${SPEAK_GRACE:-50}"
SPEAK_TAU="${SPEAK_TAU:-2500}"
SILENCE_GRACE="${SILENCE_GRACE:-1}"
SILENCE_TAU="${SILENCE_TAU:-20}"

# Brief interruptions are tolerated; sustained overlap is materially costlier.
OVERLAP_BASE_WEIGHT="${OVERLAP_BASE_WEIGHT:-0.35}"
OVERLAP_MAX_WEIGHT="${OVERLAP_MAX_WEIGHT:-1.5}"
OVERLAP_GRACE="${OVERLAP_GRACE:-1}"
OVERLAP_TAU="${OVERLAP_TAU:-3}"
# Extra credit for a DIFFERENT agent taking over (on top of baseline p_handoff credit).
HANDOFF_BONUS_WEIGHT="${HANDOFF_BONUS_WEIGHT:-0.0}"

# LoRA + selective base-layer adaptation.
LORA_ENABLE=true
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
FREEZE_BASE_MODEL=true
UNFREEZE_FIRST_N_LAYERS="${UNFREEZE_FIRST_N_LAYERS:-2}"
UNFREEZE_LAST_N_LAYERS="${UNFREEZE_LAST_N_LAYERS:-2}"

# V100-safe optimization.
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
INTERFACE_LR="${INTERFACE_LR:-$LEARNING_RATE}"
LORA_LR="${LORA_LR:-$LEARNING_RATE}"
BASE_LR="${BASE_LR:-$LEARNING_RATE}"
LAMBDA_L2SP="${LAMBDA_L2SP:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-probabilistic}"
CHAIN_RICH_OVERSAMPLE_FACTOR="${CHAIN_RICH_OVERSAMPLE_FACTOR:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
# Mitigates progressive CUDA allocator fragmentation on tighter-memory GPUs
# (V100 32GB) during long runs -- see
# .cursor/rules/v100-progressive-fragmentation-oom.mdc.
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-100}"
NUM_EPOCHS="${NUM_EPOCHS:-15}"
NUM_STEPS="${NUM_STEPS:-0}"
NUM_WORKERS=8
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-100}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"       # enables predefined AMI/MELD/Werewolf validation splits
MAX_FLAT_LEN="${MAX_FLAT_LEN:-1536}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
SAMPLE_MAX_NEW_TOKENS=32
TORCH_DTYPE="bfloat16"

# NOTE: resolve_weights() normalizes an explicit override to sum to 1.0, so
# three equal values give each source ~1/3 of sampling regardless of its
# manifest default_weight or example count (see training/multi_source.py).
DATASET_WEIGHTS="${DATASET_WEIGHTS:-meld=0.3334,ami=0.3333,werewolf=0.3333}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
# Draws a fresh per-example row permutation on every train-split collation
# (on top of data/convert.py's bake-once-at-preprocessing permute_agents) --
# see training/multi_source.py's collate_matrix_batch docstring. Default off
# so every existing recipe's behavior is unaffected.
LIVE_REPERMUTE_AGENTS="${LIVE_REPERMUTE_AGENTS:-false}"

# Periodic in-training "kitchen sink" handoff/listener probe -- a CONTINUOUS
# signal on actual conversation/handoff behavior (not just loss), logged into
# metrics.jsonl under "probe" every PROBE_EVERY steps. 0 = disabled (default;
# it's a real generation-based check and meaningfully slows a step down when
# it fires). Recommend matching CHECKPOINT_EVERY. See SUCCESS_STORIES.md's
# 07-27 correction for why this exists: loss can look great while every
# generated "conversation" is silent or degenerate.
PROBE_EVERY="${PROBE_EVERY:-0}"
PROBE_NUM_AGENTS="${PROBE_NUM_AGENTS:-3}"
PROBE_MAX_NEW_TOKENS="${PROBE_MAX_NEW_TOKENS:-12}"

EXTRA_ARGS=()
if [[ -n "$RUN_NAME" ]]; then
  EXTRA_ARGS+=(--run_name "$RUN_NAME")
fi
if [[ -n "$RESUME_FROM" && -n "$INIT_FROM" ]]; then
  echo "[error] set only one of RESUME_FROM or INIT_FROM" >&2
  exit 2
fi
if [[ -n "$RESUME_FROM" ]]; then
  EXTRA_ARGS+=(--resume_from "$RESUME_FROM")
fi
if [[ -n "$INIT_FROM" ]]; then
  EXTRA_ARGS+=(--init_from "$INIT_FROM" --init_checkpoint_prefer "$INIT_CHECKPOINT_PREFER")
fi

singularity exec --nv -B /data:/data "$CONTAINER" python main.py \
  --use_tiny_model false \
  --model_name "$MODEL_PATH" \
  --processed_data_dir "$PROCESSED_DIR" \
  --dataset_weights "$DATASET_WEIGHTS" \
  --sampling_strategy "$SAMPLING_STRATEGY" \
  --chain_rich_oversample_factor "$CHAIN_RICH_OVERSAMPLE_FACTOR" \
  --seed "$SEED" \
  --max_agents "$MAX_AGENTS" \
  --position_mode "$POSITION_MODE" \
  --use_agent_embeddings true \
  --agent_attention_mode "$AGENT_ATTENTION_MODE" \
  --agent_attention_representation "$AGENT_ATTENTION_REPRESENTATION" \
  --agent_attention_dim "$AGENT_ATTENTION_DIM" \
  --agent_attention_init_std "$AGENT_ATTENTION_INIT_STD" \
  --agent_same_attention_bias "$AGENT_SAME_ATTENTION_BIAS" \
  --agent_same_attention_bias_init "$AGENT_SAME_ATTENTION_BIAS_INIT" \
  --agent_relation_bias_mode "$AGENT_RELATION_BIAS_MODE" \
  --agent_relation_bias_representation "$AGENT_RELATION_BIAS_REPRESENTATION" \
  --agent_relation_bias_dim "$AGENT_RELATION_BIAS_DIM" \
  --agent_relation_bias_init_std "$AGENT_RELATION_BIAS_INIT_STD" \
  --agent_relation_bias_same_init "$AGENT_RELATION_BIAS_SAME_INIT" \
  --agent_relation_bias_diff_init "$AGENT_RELATION_BIAS_DIFF_INIT" \
  --agent_dynamic_state_mode "$AGENT_DYNAMIC_STATE_MODE" \
  --agent_dynamic_state_dim "$AGENT_DYNAMIC_STATE_DIM" \
  --agent_dynamic_state_init_std "$AGENT_DYNAMIC_STATE_INIT_STD" \
  --attn_mask_mode matrix_causal \
  --placeholder_token_id "$PLACEHOLDER_TOKEN_ID" \
  --lambda_content "$LAMBDA_CONTENT" \
  --lambda_activity "$LAMBDA_ACTIVITY" \
  --lambda_reward "$LAMBDA_REWARD" \
  --lambda_same_handoff "$LAMBDA_SAME_HANDOFF" \
  --activity_pos_weight "$ACTIVITY_POS_WEIGHT" \
  --turn_reward_mode "$TURN_REWARD_MODE" \
  --floor_control_ref_silence "$FLOOR_CONTROL_REF_SILENCE" \
  --floor_control_ref_same "$FLOOR_CONTROL_REF_SAME" \
  --floor_control_ref_handoff "$FLOOR_CONTROL_REF_HANDOFF" \
  --floor_control_ref_overlap "$FLOOR_CONTROL_REF_OVERLAP" \
  --speak_grace "$SPEAK_GRACE" \
  --speak_tau "$SPEAK_TAU" \
  --silence_grace "$SILENCE_GRACE" \
  --silence_tau "$SILENCE_TAU" \
  --overlap_base_weight "$OVERLAP_BASE_WEIGHT" \
  --overlap_max_weight "$OVERLAP_MAX_WEIGHT" \
  --overlap_grace "$OVERLAP_GRACE" \
  --overlap_tau "$OVERLAP_TAU" \
  --handoff_bonus_weight "$HANDOFF_BONUS_WEIGHT" \
  --lora_enable "$LORA_ENABLE" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --freeze_base_model "$FREEZE_BASE_MODEL" \
  --unfreeze_first_n_layers "$UNFREEZE_FIRST_N_LAYERS" \
  --unfreeze_last_n_layers "$UNFREEZE_LAST_N_LAYERS" \
  --learning_rate "$LEARNING_RATE" \
  --interface_lr "$INTERFACE_LR" \
  --lora_lr "$LORA_LR" \
  --base_lr "$BASE_LR" \
  --lambda_l2sp "$LAMBDA_L2SP" \
  --weight_decay "$WEIGHT_DECAY" \
  --lr_scheduler "$LR_SCHEDULER" \
  --warmup_steps "$WARMUP_STEPS" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --empty_cache_every "$EMPTY_CACHE_EVERY" \
  --num_epochs "$NUM_EPOCHS" \
  --num_steps "$NUM_STEPS" \
  --num_workers "$NUM_WORKERS" \
  --log_every "$LOG_EVERY" \
  --eval_every "$EVAL_EVERY" \
  --val_fraction "$VAL_FRACTION" \
  --max_flat_len "$MAX_FLAT_LEN" \
  --checkpoint_every "$CHECKPOINT_EVERY" \
  --live_repermute_agents "$LIVE_REPERMUTE_AGENTS" \
  --probe_every "$PROBE_EVERY" \
  --probe_num_agents "$PROBE_NUM_AGENTS" \
  --probe_max_new_tokens "$PROBE_MAX_NEW_TOKENS" \
  --sample_every "$SAMPLE_EVERY" \
  --sample_max_new_tokens "$SAMPLE_MAX_NEW_TOKENS" \
  --sample_prompts \
    "honestly i think we should just go with the second option so what does everyone else think" \
    "" \
  --torch_dtype "$TORCH_DTYPE" \
  --device cuda \
  --save_checkpoint true \
  "${EXTRA_ARGS[@]}"

echo "done."
