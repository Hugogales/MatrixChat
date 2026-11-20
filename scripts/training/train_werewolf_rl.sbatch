#!/bin/bash
# MatrixChat Stage-B Werewolf RL (WEREWOLF_RL_PLAN.md) on 1-8 GPUs of one node.
#
# torchrun starts one process per allocated GPU; each plays its own games and
# gradients are summed across processes. dtype is picked per GPU: bf16 on
# H100 (trains LoRA + interface + the SFT-unfrozen base layers), fp16 with
# loss scaling on V100 (LoRA + interface only). Override anything after the
# script name, e.g. a 2-step smoke test on one H100:
#   OUT=checkpoints/werewolf_rl/smoke sbatch --partition=dgxh100 --gres=gpu:h100:1 \
#     scripts/training/train_werewolf_rl.sbatch --steps 2 --games_per_rank 1
# Four V100s, 2 games per GPU per step:
#   OUT=checkpoints/werewolf_rl/v100x4 sbatch --partition=dgx --gres=gpu:v100:4 \
#     scripts/training/train_werewolf_rl.sbatch --steps 100 --games_per_rank 2
# Resume: add --resume_from "$OUT/step_000NN".
#SBATCH --job-name=werewolf_rl
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

echo "host: $(hostname)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# PCIe T4 nodes: NCCL peer-to-peer transfers hang there (one rank's collective
# never completes), so route through host memory. NVLink nodes keep P2P.
if [ "${SLURM_JOB_PARTITION:-}" = "teaching" ]; then
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
fi

CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
OUT="${OUT:-checkpoints/werewolf_rl/run_${SLURM_JOB_ID:-local}}"
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
NGPU="${NGPU:-1}"
[ "$NGPU" -ge 1 ] || NGPU=1
echo "gpus: $NGPU  output: $OUT"

# Host python by default (as scripts/eval/werewolf_game.py runs); for the
# container: RUNNER="singularity exec --nv -B /data:/data $CONTAINER".
RUNNER="${RUNNER:-}"

$RUNNER python -m torch.distributed.run --standalone --nproc_per_node "$NGPU" \
  scripts/training/train_werewolf_rl.py \
  --checkpoint_dir "${CHECKPOINT_DIR:-checkpoints/checkpoint}" \
  --hyperparameters "${HYPERPARAMETERS:-hyperparameters/adaptcleanv3_cand_00003.json}" \
  --model_path "${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}" \
  --output_dir "$OUT" \
  "$@"