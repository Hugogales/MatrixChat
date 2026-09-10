#!/bin/bash
# MatrixChat: DynamicAgentState (DialogueRNN-inspired causal per-agent GRU,
# model/dynamic_agent_state.py) real evaluation arm.
#
# Reproduces the paper-leader recipe (candidate106/seed176106, EXACTLY the
# same config as hyperparameters/rewardredesign_linear_control_s176106.json /
# rewardredesign_balanced_ce_s176106.json) with every field hardcoded below
# to avoid drift from train_meld_ami_werewolf.sbatch's differing defaults --
# changing ONLY --agent_dynamic_state_mode (none -> gru) so this is a clean
# paired comparison against the already-running 274173/274174 control arms.
# TURN_REWARD_MODE is a free variable so this can pair against EITHER control
# (linear against 274173, balanced_ce against 274174).
#
# Submit (two arms, one against each existing control):
#   RUN_NAME=dynstate_gru_linear_s176106   TURN_REWARD_MODE=linear \
#     sbatch --partition=dgxh100 --gres=gpu:h100:1 scripts/training/train_dynstate_ablation.sbatch
#   RUN_NAME=dynstate_gru_balcecfg_s176106 TURN_REWARD_MODE=balanced_ce \
#     sbatch --partition=dgxh100 --gres=gpu:h100:1 scripts/training/train_dynstate_ablation.sbatch
#SBATCH --account=undergrad_research
#SBATCH --job-name=mc_dynstate_gru
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
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
PROCESSED_DIR="${PROCESSED_DIR:-$HOME/matrixchat/processed_final_20260814_lookback192_allspk}"
RUN_NAME="${RUN_NAME:?must set RUN_NAME}"
SEED="${SEED:-176106}"
TURN_REWARD_MODE="${TURN_REWARD_MODE:?must set TURN_REWARD_MODE (linear|balanced_ce)}"
RESUME_FROM="${RESUME_FROM:-}"
# See .cursor/rules/resume-needs-explicit-num-steps.mdc -- main.py computes
# its step cap from NUM_EPOCHS*steps_per_epoch fresh each launch, NOT from
# the resumed checkpoint's own step, so a resume needs an explicit NUM_STEPS
# comfortably past the checkpoint's step or the job trains 0 further steps.
NUM_STEPS="${NUM_STEPS:-2000}"
# Added 2026-08-27 after a real wasted-GPU incident: a queued
# "dynstate_gru_balcecfg_dim128" job (274240) only changed RUN_NAME/SEED/
# TURN_REWARD_MODE via the queue's --env, never actually overrode this
# hardcoded 64 below -- so it silently ran the byte-identical config to the
# s176106 leader (same seed too) for ~9.5 GPU-hours, confirmed after the fact
# by its probe trajectory being IDENTICAL step-for-step to the original.
# Always verify a queued override actually reaches main.py's argv (grep the
# job's own .out banner), don't assume a descriptive RUN_NAME implies the
# knob it names was actually plumbed through.
AGENT_DYNAMIC_STATE_DIM="${AGENT_DYNAMIC_STATE_DIM:-64}"
FLOOR_CONTROL_ADAPTIVE_WEIGHT_ALPHA="${FLOOR_CONTROL_ADAPTIVE_WEIGHT_ALPHA:-0.0}"
FLOOR_CONTROL_ADAPTIVE_PRIOR_STRENGTH="${FLOOR_CONTROL_ADAPTIVE_PRIOR_STRENGTH:-32.0}"
DATASET_WEIGHTS="${DATASET_WEIGHTS:-ami=0.5,meld=0.15,werewolf=0.35}"
INIT_FROM="${INIT_FROM:-}"
INIT_CHECKPOINT_PREFER="${INIT_CHECKPOINT_PREFER:-best_probe}"
QUALITY_OVERSAMPLE_FACTOR="${QUALITY_OVERSAMPLE_FACTOR:-1.0}"
AGENT_SAME_ATTENTION_BIAS="${AGENT_SAME_ATTENTION_BIAS:-false}"
AGENT_RELATION_BIAS_MODE="${AGENT_RELATION_BIAS_MODE:-none}"
PARTITION="${PARTITION:-dgxh100}"

EXTRA_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
  EXTRA_ARGS+=(--resume_from "$RESUME_FROM")
fi
if [[ -n "$INIT_FROM" ]]; then
  EXTRA_ARGS+=(--init_from "$INIT_FROM" --init_checkpoint_prefer "$INIT_CHECKPOINT_PREFER")
fi

# Everything below is hardcoded to exactly match
# hyperparameters/rewardredesign_{linear_control,balanced_ce}_s176106.json --
# the only intended difference from those two control runs is
# --agent_dynamic_state_mode gru (vs none). NUM_STEPS/RESUME_FROM are
# overridable (added 2026-08-26) for train-longer continuations.
singularity exec --nv -B /data:/data "$CONTAINER" python main.py \
  --use_tiny_model false \
  --model_name "$MODEL_PATH" \
  --processed_data_dir "$PROCESSED_DIR" \
  --dataset_weights "$DATASET_WEIGHTS" \
  --sampling_strategy balanced_cycle \
  --chain_rich_oversample_factor 1.0 \
  --quality_oversample_factor "$QUALITY_OVERSAMPLE_FACTOR" \
  --seed "$SEED" \
  --max_agents 8 \
  --num_agents 4 \
  --position_mode column \
  --use_agent_embeddings true \
  --agent_attention_mode qkv_gated \
  --agent_attention_representation simplex \
  --agent_attention_dim 32 \
  --agent_attention_init_std 0.001 \
  --agent_same_attention_bias "$AGENT_SAME_ATTENTION_BIAS" \
  --agent_same_attention_bias_init 0.5 \
  --agent_relation_bias_mode "$AGENT_RELATION_BIAS_MODE" \
  --agent_relation_bias_representation learned \
  --agent_relation_bias_dim 8 \
  --agent_relation_bias_init_std 0.001 \
  --agent_relation_bias_same_init 0.0 \
  --agent_relation_bias_diff_init 0.0 \
  --agent_dynamic_state_mode gru \
  --agent_dynamic_state_dim "$AGENT_DYNAMIC_STATE_DIM" \
  --agent_dynamic_state_init_std 0.001 \
  --attn_mask_mode matrix_causal \
  --placeholder_token_id 0 \
  --lambda_content 1.0 \
  --lambda_activity 0.5 \
  --lambda_reward 0.05 \
  --lambda_same_handoff 0.0 \
  --activity_pos_weight 0.75 \
  --activity_threshold 0.5 \
  --turn_reward_mode "$TURN_REWARD_MODE" \
  --floor_control_ref_silence 0.0784 \
  --floor_control_ref_same 0.7805 \
  --floor_control_ref_handoff 0.0675 \
  --floor_control_ref_overlap 0.0735 \
  --floor_control_adaptive_weight_alpha "$FLOOR_CONTROL_ADAPTIVE_WEIGHT_ALPHA" \
  --floor_control_adaptive_prior_strength "$FLOOR_CONTROL_ADAPTIVE_PRIOR_STRENGTH" \
  --speak_grace 5.0 \
  --speak_tau 200.0 \
  --silence_grace 0.0 \
  --silence_tau 5.0 \
  --overlap_base_weight 1.674 \
  --overlap_max_weight 5.567 \
  --overlap_grace 0.0 \
  --overlap_tau 2.0 \
  --handoff_bonus_weight 1.1 \
  --lora_enable true \
  --lora_r 128 \
  --lora_alpha 256 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --freeze_base_model true \
  --unfreeze_first_n_layers 2 \
  --unfreeze_last_n_layers 2 \
  --learning_rate 5e-05 \
  --interface_lr 5e-05 \
  --lora_lr 5e-05 \
  --base_lr 5e-05 \
  --lambda_l2sp 0.02 \
  --weight_decay 0.0 \
  --lr_scheduler constant \
  --warmup_steps 0 \
  --max_grad_norm 1.0 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --empty_cache_every 100 \
  --num_epochs 30 \
  --num_steps "$NUM_STEPS" \
  --num_workers 8 \
  --log_every 20 \
  --eval_every 250 \
  --val_fraction 0.05 \
  --max_flat_len 2048 \
  --checkpoint_every 250 \
  --live_repermute_agents false \
  --probe_every 250 \
  --probe_num_agents 3 \
  --probe_max_new_tokens 16 \
  --sample_every 0 \
  --sample_max_new_tokens 32 \
  --sample_prompts \
    "honestly i think we should just go with the second option so what does everyone else think" \
    "" \
  --torch_dtype bfloat16 \
  --device cuda \
  --save_checkpoint true \
  --run_name "$RUN_NAME" \
  "${EXTRA_ARGS[@]}"

echo "done."
