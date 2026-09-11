#!/bin/bash
# MatrixChat: data-source ablation arm on a V100 -- reproduces the paper-
# leader recipe (candidate106/seed176106, EXACTLY the confirmed-best config:
# balanced_ce reward + DynamicAgentState gru, see hyperparameters/
# dynstate_gru_balcecfg_s176106.json) with every field hardcoded below,
# changing ONLY --dataset_weights (to add a new source, e.g. bazinga or
# when2speak) and adapting batch shape for a V100's 32GB (batch_size=1 +
# gradient_accumulation_steps=32, same effective batch=32 as the H100
# leader's batch_size=4 x grad_accum=8; max_flat_len kept at the
# already-proven-safe 1536 rather than the H100 leader's 2048 -- see
# .cursor/rules/eager-attention-quadratic-memory-caution.mdc and
# .cursor/rules/v100-progressive-fragmentation-oom.mdc).
#
# Prerequisite: the new source must already be additively converted into
# PROCESSED_DIR (see scripts/data_prep/preprocess_bazinga.sbatch /
# preprocess_when2speak.sbatch) BEFORE this job runs, or main.py will error
# on an unknown dataset_weights key.
#
# Submit (DATASET_WEIGHTS's commas make it UNSAFE to inline in
# `sbatch --export=...` -- see .cursor/rules/sbatch-env-overrides.mdc -- so
# `export` it in the submitting shell first, or use job_queue.py which passes
# entry env via subprocess env=, not an inline --export string):
#   export DATASET_WEIGHTS="ami=0.4,meld=0.12,werewolf=0.28,bazinga=0.2"
#   RUN_NAME=data_abl_bazinga_s176106 sbatch --partition=dgx --gres=gpu:v100:1 \
#     scripts/training/train_data_ablation.sbatch
#SBATCH --account=undergrad_research
#SBATCH --job-name=mc_data_ablation
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
DATASET_WEIGHTS="${DATASET_WEIGHTS:?must set DATASET_WEIGHTS, e.g. ami=0.4,meld=0.12,werewolf=0.28,bazinga=0.2}"
FLOOR_CONTROL_ADAPTIVE_WEIGHT_ALPHA="${FLOOR_CONTROL_ADAPTIVE_WEIGHT_ALPHA:-0.0}"
FLOOR_CONTROL_ADAPTIVE_PRIOR_STRENGTH="${FLOOR_CONTROL_ADAPTIVE_PRIOR_STRENGTH:-32.0}"

# Everything below is hardcoded to exactly match
# hyperparameters/dynstate_gru_balcecfg_s176106.json except dataset_weights
# (the ablated variable) and the V100 batch-shape adaptation noted above.
singularity exec --nv -B /data:/data "$CONTAINER" python main.py \
  --use_tiny_model false \
  --model_name "$MODEL_PATH" \
  --processed_data_dir "$PROCESSED_DIR" \
  --dataset_weights "$DATASET_WEIGHTS" \
  --sampling_strategy balanced_cycle \
  --chain_rich_oversample_factor 1.0 \
  --seed "$SEED" \
  --max_agents 8 \
  --num_agents 4 \
  --position_mode column \
  --use_agent_embeddings true \
  --agent_attention_mode qkv_gated \
  --agent_attention_representation simplex \
  --agent_attention_dim 32 \
  --agent_attention_init_std 0.001 \
  --agent_same_attention_bias false \
  --agent_same_attention_bias_init 0.5 \
  --agent_relation_bias_mode none \
  --agent_relation_bias_representation learned \
  --agent_relation_bias_dim 8 \
  --agent_relation_bias_init_std 0.001 \
  --agent_relation_bias_same_init 0.0 \
  --agent_relation_bias_diff_init 0.0 \
  --agent_dynamic_state_mode gru \
  --agent_dynamic_state_dim 64 \
  --agent_dynamic_state_init_std 0.001 \
  --attn_mask_mode matrix_causal \
  --placeholder_token_id 0 \
  --lambda_content 1.0 \
  --lambda_activity 0.5 \
  --lambda_reward 0.05 \
  --lambda_same_handoff 0.0 \
  --activity_pos_weight 0.75 \
  --activity_threshold 0.5 \
  --turn_reward_mode balanced_ce \
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
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --gradient_checkpointing true \
  --empty_cache_every 100 \
  --num_epochs 30 \
  --num_steps 2000 \
  --num_workers 8 \
  --log_every 20 \
  --eval_every 250 \
  --val_fraction 0.05 \
  --max_flat_len 1536 \
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
  --run_name "$RUN_NAME"

echo "done."
