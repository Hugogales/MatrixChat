#!/bin/bash
# Always-on Werewolf RL: one MAIN node job (submitted by start.sh; renews itself).
# Its GPUs are split evenly: the first TRAIN_GPUS (default: half) run trainer.py,
# the rest run player.py. Resources come from H100_SBATCH_OPTS / V100_SBATCH_OPTS
# in start.sh. A process that fails is restarted up to 3 times in a row.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -uo pipefail
source "$REPO/scripts/training/werewolf_continuous/chain.sh"
chain_start

IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NTRAIN="${TRAIN_GPUS:-$(( ${#DEVS[@]} / 2 ))}"
echo "[main] GPUs ${DEVS[*]}: first $NTRAIN train, the rest play"

run_slot() {  # what slot dev
  local what=$1 slot=$2 dev=$3 restarts=0 rc log
  log="$RUN_DIR/logs/${CHAIN_ROLE}_${SLURM_JOB_ID}_${what}${slot}.log"
  while [ ! -f "$RUN_DIR/STOP" ]; do
    if [ "$what" = train ]; then
      # shellcheck disable=SC2086  # TRAINER_ARGS is a list of extra flags
      CUDA_VISIBLE_DEVICES="$dev" python scripts/training/werewolf_continuous/trainer.py \
        --run_dir "$RUN_DIR" --role main --chain_names "$CHAIN_NAMES" \
        --checkpoint_dir "$CHECKPOINT_DIR" --hyperparameters "$HYPERPARAMETERS" --model_path "$MODEL_PATH" \
        --extra_partitions "$EXTRA_PARTITIONS" --max_extra "$MAX_EXTRA" --extra_name "$EXTRA_NAME" \
        --extra_sbatch "$REPO/scripts/training/werewolf_continuous/extra.sbatch" \
        $TRAINER_ARGS >> "$log" 2>&1
    else
      # shellcheck disable=SC2086  # PLAYER_ARGS is a list of extra flags
      CUDA_VISIBLE_DEVICES="$dev" python scripts/training/werewolf_continuous/player.py \
        --run_dir "$RUN_DIR" --slot "$slot" --role main --chain_names "$CHAIN_NAMES" \
        --checkpoint_dir "$CHECKPOINT_DIR" --hyperparameters "$HYPERPARAMETERS" --model_path "$MODEL_PATH" \
        $PLAYER_ARGS >> "$log" 2>&1
    fi
    rc=$?
    [ "$rc" -eq 0 ] && return 0
    restarts=$((restarts + 1))
    echo "[main] $what slot $slot exited with $rc (restart $restarts/3)"
    if [ "$restarts" -ge 3 ]; then
      chain_notify gave_up "Werewolf RL $RUN_NAME: a $what process on $(hostname -s) (job $SLURM_JOB_ID, GPU slot $slot) crashed 3 times in a row (exit $rc); that GPU is idle until the job ends. Log: $log"
      return "$rc"
    fi
    sleep 30
  done
}

pids=()
for i in "${!DEVS[@]}"; do
  if [ "$i" -lt "$NTRAIN" ]; then run_slot train "$i" "${DEVS[$i]}" & else run_slot play "$i" "${DEVS[$i]}" & fi
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
