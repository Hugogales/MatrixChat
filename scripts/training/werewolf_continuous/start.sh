#!/bin/bash
# Start (or restart) the always-on Werewolf RL loop on Rosie.
#
#   scripts/training/werewolf_continuous/start.sh [RUN_NAME]        (default RUN_NAME=run1)
#
# Submits two self-renewing node jobs -- nothing has to stay open afterwards:
#   wwc-<RUN>-h100  4x H100 (dgxh100): 2 GPUs train, 2 GPUs play
#   wwc-<RUN>-v100  4x V100 (dgx):     2 GPUs train, 2 GPUs play
# Training is one model: all training GPUs (on both nodes, plus borrowed ones)
# share each step through files in the run directory, led by whichever holds
# the lease. Every game from every GPU goes into one pool. The leader borrows
# idle GPUs on EXTRA_PARTITIONS (wwc-<RUN>-extra), alternating each GPU between
# training and playing, and they give them back as soon as anyone else waits.
# Every job queues its successor when it starts, so the loop continues past the
# 24 h limit; 3 crashes in a row stop a chain. Restarting a run resumes it.
#
# Stop:   scripts/training/werewolf_continuous/stop.sh  [RUN_NAME] [--now]
# Status: scripts/training/werewolf_continuous/status.sh [RUN_NAME]
#
# Overridable environment (defaults shown):
#   RUN_DIR=checkpoints/werewolf_rl/continuous/<RUN_NAME>
#   CHECKPOINT_DIR=checkpoints/checkpoint  HYPERPARAMETERS=hyperparameters/adaptcleanv3_cand_00003.json
#   MODEL_PATH=models/Qwen3-4B-Instruct-2507
#   H100_SBATCH_OPTS="--partition=dgxh100 --gres=gpu:h100:4 --cpus-per-task=16 --mem=192G"
#   V100_SBATCH_OPTS="--partition=dgx --gres=gpu:v100:4 --cpus-per-task=16 --mem=192G"
#   TRAIN_GPUS=<half of each job's GPUs>
#   EXTRA_PARTITIONS=dgx,dgxh100   MAX_EXTRA=0 (no cap)
#   TRAINER_ARGS="--games_per_step 16 --save_every 1 --keep_last 3"   (trainer.py flags)
#   PLAYER_ARGS=""                                                     (player.py flags)
#   GAME_ARGS=""   game rules/sampling flags (game.add_game_args); written to the run's
#                  game_config.json on first start (or when given), shared by every job.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
RUN_NAME="${RUN_NAME:-${1:-run1}}"
RUN_DIR="${RUN_DIR:-$REPO/checkpoints/werewolf_rl/continuous/$RUN_NAME}"
case "$RUN_DIR" in /*) ;; *) RUN_DIR="$REPO/$RUN_DIR" ;; esac
H100_NAME="wwc-$RUN_NAME-h100"
V100_NAME="wwc-$RUN_NAME-v100"
EXTRA_NAME="wwc-$RUN_NAME-extra"
CHAIN_NAMES="$H100_NAME,$V100_NAME"

existing="$(squeue -h -u "$USER" -o '%i %j' | awk -v a="$H100_NAME" -v b="$V100_NAME" '$2==a || $2==b')"
if [ -n "$existing" ]; then
  echo "Run '$RUN_NAME' already has jobs in the queue:"; echo "$existing"
  echo "Stop it first (stop.sh $RUN_NAME --now) or pick another RUN_NAME."
  exit 1
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/checkpoint}"
HYPERPARAMETERS="${HYPERPARAMETERS:-hyperparameters/adaptcleanv3_cand_00003.json}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
H100_SBATCH_OPTS="${H100_SBATCH_OPTS:---partition=dgxh100 --gres=gpu:h100:4 --cpus-per-task=16 --mem=192G}"
V100_SBATCH_OPTS="${V100_SBATCH_OPTS:---partition=dgx --gres=gpu:v100:4 --cpus-per-task=16 --mem=192G}"
EXTRA_PARTITIONS="${EXTRA_PARTITIONS-dgx,dgxh100}"
MAX_EXTRA="${MAX_EXTRA:-0}"
TRAINER_ARGS="${TRAINER_ARGS---games_per_step 16 --save_every 1 --keep_last 3}"
PLAYER_ARGS="${PLAYER_ARGS:-}"
GAME_ARGS="${GAME_ARGS:-}"

mkdir -p logs/slurm "$RUN_DIR"
if [ -f "$RUN_DIR/STOP" ]; then
  echo "Removing $RUN_DIR/STOP (left by an earlier stop)"
  rm -f "$RUN_DIR/STOP"
fi

if [ ! -f "$RUN_DIR/game_config.json" ] || [ -n "$GAME_ARGS" ]; then
  # shellcheck disable=SC2086
  python -m training.werewolf_rl.continuous write-config --run_dir "$RUN_DIR" $GAME_ARGS > /dev/null
  echo "Wrote $RUN_DIR/game_config.json"
fi

# An explicit start means "try again": reset the crash counters
rm -f "$RUN_DIR"/chain/h100.json "$RUN_DIR"/chain/v100.json

export REPO RUN_NAME RUN_DIR CHECKPOINT_DIR HYPERPARAMETERS MODEL_PATH EXTRA_PARTITIONS MAX_EXTRA \
  TRAINER_ARGS PLAYER_ARGS CHAIN_NAMES EXTRA_NAME
[ -n "${TRAIN_GPUS:-}" ] && export TRAIN_GPUS
unset PREV_JOB

SCRIPT="$REPO/scripts/training/werewolf_continuous/main.sbatch"
submit() {  # role opts name
  local role=$1 opts=$2 name=$3
  # shellcheck disable=SC2086
  CHAIN_ROLE="$role" CHAIN_SCRIPT="$SCRIPT" CHAIN_SBATCH_OPTS="$opts" \
    sbatch $opts --job-name="$name" --export=ALL "$SCRIPT"
}
submit h100 "$H100_SBATCH_OPTS" "$H100_NAME"
submit v100 "$V100_SBATCH_OPTS" "$V100_NAME"
echo "Run dir: $RUN_DIR"
echo "Status:  scripts/training/werewolf_continuous/status.sh $RUN_NAME"
echo "Stop:    scripts/training/werewolf_continuous/stop.sh $RUN_NAME   (add --now to cancel immediately)"