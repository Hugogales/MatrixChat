# Sourced by main.sbatch (the two self-renewing node job chains, h100 and v100).
#
# chain_start: run the chain guard (STOP file, 3-crashes-in-a-row rule), then
# queue this job's successor right away with --dependency=afterany, so it is
# waiting in the queue (and ageing in priority) the moment this job ends --
# whether it ends at the time limit or by crashing.
#
# Needs (exported by start.sh, inherited job to job): REPO RUN_DIR CHAIN_ROLE
# CHAIN_SCRIPT CHAIN_SBATCH_OPTS; PREV_JOB is set for every successor.

chain_start() {
  cd "$REPO" || exit 1
  _chain_env
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "[chain:$CHAIN_ROLE] $(date '+%F %T') job $SLURM_JOB_ID on $(hostname) (previous: ${PREV_JOB:-none})"
  if ! python -m training.werewolf_rl.continuous guard --run_dir "$RUN_DIR" --role "$CHAIN_ROLE" \
      --prev_job "${PREV_JOB:-}"; then
    echo "[chain:$CHAIN_ROLE] not continuing (see $RUN_DIR/chain/$CHAIN_ROLE.json)"
    exit 0
  fi
  local out="" attempt existing
  # A requeued job runs this again: never queue a second successor.
  existing="$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t PD -o '%i %E' | grep -F ":$SLURM_JOB_ID" | awk '{print $1}')"
  if [ -n "$existing" ]; then
    echo "[chain:$CHAIN_ROLE] successor already queued: $existing"
    return 0
  fi
  for attempt in 1 2 3 4 5; do
    # shellcheck disable=SC2086  # CHAIN_SBATCH_OPTS is a list of options
    if out=$(sbatch $CHAIN_SBATCH_OPTS --job-name="$SLURM_JOB_NAME" --dependency="afterany:$SLURM_JOB_ID" \
        --export="ALL,PREV_JOB=$SLURM_JOB_ID" "$CHAIN_SCRIPT" 2>&1); then
      break
    fi
    echo "[chain:$CHAIN_ROLE] sbatch failed (attempt $attempt): $out"
    sleep 60
  done
  echo "[chain:$CHAIN_ROLE] $(date '+%F %T') job $SLURM_JOB_ID queued its successor: $out" \
    | tee -a "$RUN_DIR/chain/${CHAIN_ROLE}_jobs.log"
  if [ -z "${PREV_JOB:-}" ]; then  # first job of this chain since start.sh
    chain_notify start "Werewolf RL $RUN_NAME: the $CHAIN_ROLE chain is running (job $SLURM_JOB_ID on $(hostname -s))."
  fi
}

# chain_notify EVENT MESSAGE: text an alert (no-op without ~/.config/werewolf_rl/notify.env).
chain_notify() {
  python -m training.werewolf_rl.notify send --event "$1" --run_dir "$RUN_DIR" "$2" || true
}

_chain_env() {
  if [ "${SLURM_JOB_PARTITION:-}" = "teaching" ]; then
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"  # PCIe T4 nodes: NCCL P2P hangs
  fi
}
