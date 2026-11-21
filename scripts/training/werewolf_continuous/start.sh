#!/bin/bash
# Start (or resume / repair) the always-on Werewolf RL loop on Rosie.
#
#   scripts/training/werewolf_continuous/start.sh [RUN_NAME]        (default RUN_NAME=run1)
#
# Run it on a login node. Nothing has to stay open afterwards. It sets up:
#   * CHAINS: self-renewing jobs that hold their GPUs (default 4xH100 + 8xV100 +
#     4xV100). Each GPU job runs trainer.py on half its GPUs and player.py on the
#     other half; all trainers form one model (file-coordinated, RUN/train) and
#     all games go into one pool. Each chain keeps CHAIN_DEPTH (14) follow-up jobs
#     queued (--dependency=afterany), so at every 24 h handover the next job has
#     been waiting for two weeks and is first in line for the GPUs. A chain job
#     never gives its GPUs back early: whatever fails, it keeps them (idle) until
#     its time limit, STOP, or a rerun of this script, which makes it retry.
#   * the keeper (cron on this login node, every 5 min, training/werewolf_rl/keeper.py):
#     re-creates a chain that lost all its jobs, tops up its queue, and puts the
#     job slots the chains leave free (Rosie: at most 4 running jobs per user) to
#     work as "gap" jobs sized to the free GPUs, while the chains are still short
#     of their target.
# Running it again is safe: it only adds what is missing, resets crash counters,
# and tells idle (held) GPUs to retry (touches RUN/chain/retry).
# Resizing a chain: change CHAINS and rerun; jobs already queued keep their old
# size (stop.sh --now first to apply it at once).
#
# Pause for an update (keeps every GPU and the place in the queue), then resume:
#         scripts/training/werewolf_continuous/stop.sh [RUN_NAME] --pause; ...; start.sh [RUN_NAME]
# Stop:   scripts/training/werewolf_continuous/stop.sh  [RUN_NAME] [--now]
# start.sh and stop.sh send an ntfy status alert with the queue (README section 8).
# Status: scripts/training/werewolf_continuous/status.sh [RUN_NAME]
#
# Overridable environment (defaults shown):
#   RUN_DIR=checkpoints/werewolf_rl/continuous/<RUN_NAME>
#   CHECKPOINT_DIR=checkpoints/checkpoint  HYPERPARAMETERS=hyperparameters/adaptcleanv3_cand_00003.json
#   MODEL_PATH=models/Qwen3-4B-Instruct-2507
#   CHAINS="h100:dgxh100:4 v100:dgx:8 v100b:dgx:4"   role:partition:GPUs (one node each, <= 8 GPUs)
#   CHAIN_DEPTH=14           follow-up jobs kept queued per chain (days ahead)
#   HOLD_ON_FAILURE=1        0 = a failing chain job exits (3 crashed jobs in a row stop the chain)
#   GAP_PARTITIONS=dgx,dgxh100   where gap jobs may go ("" = no gap jobs)
#   CPUS_PER_GPU=4  MEM_PER_GPU_GB=48   EXTRA_SBATCH=""   (e.g. "--exclude=dh-dgx1-2")
#   TRAIN_GPUS=<half of each job's GPUs>
#   TRAINER_ARGS="--games_per_step 16 --save_every 1 --keep_last 3"   (trainer.py flags)
#   PLAYER_ARGS=""                                                     (player.py flags)
#   GAME_ARGS=""   game rules/sampling flags (game.add_game_args); written to the run's
#                  game_config.json on first start (or when given), shared by every job.
#   KEEPER=1       0 = don't install the keeper's cron line
#   REPORT_MINUTES=60  the keeper sends an ntfy report of every GPU this often, on the clock (0 = off)
#   TIME_MIN=00:30:00  sbatch --time-min of every chain/gap job: Slurm may start it at once with a
#                      shorter limit when an older job's reservation keeps the node free ("" = off)
#   EXTRA_GPU_PARTITIONS=dgxh100,dgx  where the gap slot also grabs GPUs beyond the chain targets
#                      (H100 and V100) while the chains hold theirs, given back as soon as another
#                      user's job needs them ("" = off)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
RUN_NAME="${RUN_NAME:-${1:-run1}}"
RUN_DIR="${RUN_DIR:-$REPO/checkpoints/werewolf_rl/continuous/$RUN_NAME}"
case "$RUN_DIR" in /*) ;; *) RUN_DIR="$REPO/$RUN_DIR" ;; esac

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/checkpoint}"
HYPERPARAMETERS="${HYPERPARAMETERS:-hyperparameters/adaptcleanv3_cand_00003.json}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
CHAINS="${CHAINS:-h100:dgxh100:4 v100:dgx:8 v100b:dgx:4}"
CHAIN_DEPTH="${CHAIN_DEPTH:-14}"
HOLD_ON_FAILURE="${HOLD_ON_FAILURE:-1}"
GAP_PARTITIONS="${GAP_PARTITIONS-dgx,dgxh100}"
CPUS_PER_GPU="${CPUS_PER_GPU:-4}"
MEM_PER_GPU_GB="${MEM_PER_GPU_GB:-48}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"
TRAINER_ARGS="${TRAINER_ARGS---games_per_step 16 --save_every 1 --keep_last 3}"
PLAYER_ARGS="${PLAYER_ARGS:-}"
GAME_ARGS="${GAME_ARGS:-}"
KEEPER="${KEEPER:-1}"
REPORT_MINUTES="${REPORT_MINUTES:-60}"
TIME_MIN="${TIME_MIN-00:30:00}"
EXTRA_GPU_PARTITIONS="${EXTRA_GPU_PARTITIONS-dgxh100,dgx}"

python -c "import torch" 2>/dev/null || {
  echo "The python on PATH ($(command -v python)) has no torch; activate the repo's .venv first."; exit 1; }

mkdir -p logs/slurm "$RUN_DIR/chain"
was=""
idle_marks=$(ls "$RUN_DIR/chain/holding" 2>/dev/null | grep -vc '^$' || true)
[ "${idle_marks:-0}" -gt 0 ] && was=idle
[ -f "$RUN_DIR/PAUSE" ] && was=paused
if [ -f "$RUN_DIR/STOP" ]; then
  echo "Removing $RUN_DIR/STOP (left by an earlier stop)"
  rm -f "$RUN_DIR/STOP"
  was=stopped
fi

if [ ! -f "$RUN_DIR/game_config.json" ] || [ -n "$GAME_ARGS" ]; then
  # shellcheck disable=SC2086
  python -m training.werewolf_rl.continuous write-config --run_dir "$RUN_DIR" $GAME_ARGS > /dev/null
  echo "Wrote $RUN_DIR/game_config.json"
fi

# An explicit start means "try again": reset the crash counters of every chain and gap slot.
for spec in $CHAINS; do rm -f "$RUN_DIR/chain/${spec%%:*}.json"; done
rm -f "$RUN_DIR"/chain/gap[0-9]*.json
# ... and wakes every GPU a chain job holds idle (after a crash, a broken python, ...): it retries within a minute.
touch "$RUN_DIR/chain/retry"

gap_opt=()
[ -z "$GAP_PARTITIONS" ] && gap_opt=(--gap_slots 0)
env_opts=(--env "CHECKPOINT_DIR=$CHECKPOINT_DIR" --env "HYPERPARAMETERS=$HYPERPARAMETERS"
          --env "MODEL_PATH=$MODEL_PATH" --env "TRAINER_ARGS=$TRAINER_ARGS" --env "PLAYER_ARGS=$PLAYER_ARGS"
          --env "EXTRA_PARTITIONS=" --env "MAX_EXTRA=0" --env "EXTRA_NAME=wwc-$RUN_NAME-extra"
          --env "HOLD_ON_FAILURE=$HOLD_ON_FAILURE")
[ -n "${TRAIN_GPUS:-}" ] && env_opts+=(--env "TRAIN_GPUS=$TRAIN_GPUS")
python -m training.werewolf_rl.keeper write-launch --run_dir "$RUN_DIR" --run_name "$RUN_NAME" --repo "$REPO" \
  --chains "$CHAINS" --depth "$CHAIN_DEPTH" --gap_partitions "${GAP_PARTITIONS:-dgx}" "${gap_opt[@]}" \
  --cpus_per_gpu "$CPUS_PER_GPU" --mem_per_gpu_gb "$MEM_PER_GPU_GB" --extra_sbatch "$EXTRA_SBATCH" \
  --report_minutes "$REPORT_MINUTES" --time_min "$TIME_MIN" --extra_partitions "$EXTRA_GPU_PARTITIONS" "${env_opts[@]}"

# Queue whatever is missing now (chains with their follow-ups, a gap job) instead of waiting for cron.
python -m training.werewolf_rl.keeper tick --run_dir "$RUN_DIR"
if [ "$KEEPER" = 1 ]; then
  python -m training.werewolf_rl.keeper install-cron --run_dir "$RUN_DIR"
fi

# Warn about queued jobs whose size differs from CHAINS (they keep their size).
python - "$RUN_DIR" <<'EOF'
import sys
from training.werewolf_rl import keeper as kp
spec = kp.read_launch(sys.argv[1])
roles = kp.by_role(spec, kp.Slurm().jobs(f"wwc-{spec['run_name']}-"))
for c in spec["chains"]:
    odd = [j["id"] for j in roles.get(c["role"], []) if j["gpus"] and j["gpus"] != c["gpus"]]
    if odd:
        print(f"note: {c['role']} jobs {', '.join(odd)} were queued with a different size; "
              f"later follow-ups use {c['gpus']} GPUs (stop.sh --now and start.sh to resize at once)")
EOF
# Resume paused GPUs last, so they start with the settings written above.
if [ -f "$RUN_DIR/PAUSE" ]; then
  rm -f "$RUN_DIR/PAUSE"
  echo "Removed $RUN_DIR/PAUSE: paused GPUs start again within a minute"
fi
case "$was" in
  paused) headline="Werewolf RL $RUN_NAME is running again after the update (start.sh): every paused GPU restarts with the new code within a minute." ;;
  stopped) headline="Werewolf RL $RUN_NAME is running again (start.sh after a stop): chains are queued again." ;;
  idle) headline="Werewolf RL $RUN_NAME: start.sh ran: every idle GPU retries within a minute (status.sh and the hourly report show whether it works now)." ;;
  *) headline="Werewolf RL $RUN_NAME: start.sh ran (nothing was paused, stopped or idle)." ;;
esac
python -m training.werewolf_rl.keeper announce --run_dir "$RUN_DIR" "$headline" || true
echo "Run dir: $RUN_DIR"
echo "Status:  scripts/training/werewolf_continuous/status.sh $RUN_NAME"
echo "Pause:   scripts/training/werewolf_continuous/stop.sh $RUN_NAME --pause   (keeps GPUs and queue; start.sh resumes)"
echo "Stop:    scripts/training/werewolf_continuous/stop.sh $RUN_NAME   (gives up the queue; add --now to cancel immediately)"
