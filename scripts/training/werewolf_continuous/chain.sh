# Sourced by main.sbatch (every self-renewing chain job: h100, v100, v100b, gapN).
#
# Slurm copies a batch script when it is submitted, so jobs queued days ahead
# would keep an old main.sbatch. Every main.sbatch version calls chain_start
# first, and this file is read when each job starts, so fixes and the run's
# settings (RUN/launch.env, written by start.sh) reach queued jobs too.
#
# chain_start: the whole job; it never returns (older queued copies of
#   main.sbatch that run more code after it get this flow too). It reads
#   RUN/launch.env, runs the chain guard (STOP file, 3 crashes in a row), tops
#   up this chain's queue -- CHAIN_DEPTH follow-up jobs, each
#   --dependency=afterany on the one before: a follow-up queued days ahead is
#   older than anything other users submit later, so at every 24 h handover it
#   is first in line for the GPUs this job releases -- and then runs the node
#   (chain_run_node) until the time limit.
# chain_run_node: trainer.py on the first TRAIN_GPUS GPUs (default half),
#   player.py on the rest; a process that fails is restarted up to 3 times.
#
# Holding (main chains; gap chains and HOLD_ON_FAILURE=0 do not): a job never
# hands its GPUs back before its time limit unless STOP exists. A process that
# crashed 3 times or exited early, a python that cannot run torch on the GPU,
# or 3 crashed jobs in a row leave the GPUs held but idle; start.sh touches
# RUN/chain/retry and everything idle starts again within CHAIN_POLL (60) s.
# RUN/chain/holding/ lists what is idle (status.sh shows it).
#
# Pausing (every chain, gap chains too): while RUN/PAUSE exists (stop.sh --pause),
# no trainer or player runs -- they leave at once -- but the job keeps its GPUs
# and its queued follow-ups; when start.sh removes PAUSE, each GPU starts again
# with the code and settings then on disk.
#
# Needs (exported by start.sh / keeper, inherited job to job): REPO RUN_DIR
# CHAIN_ROLE CHAIN_SCRIPT; PREV_JOB is set for every follow-up job.

chain_start() {
  local hold="" hold_arg="" g ran=""
  cd "$REPO" || exit 1
  export USER="${USER:-$(id -un)}"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  if [ -f "$RUN_DIR/launch.env" ]; then
    # shellcheck disable=SC1091
    source "$RUN_DIR/launch.env"
  fi
  echo "[chain:$CHAIN_ROLE] $(date '+%F %T') job $SLURM_JOB_ID on $(hostname) (previous: ${PREV_JOB:-none})"
  mkdir -p "$RUN_DIR/chain/holding"
  rm -f "$RUN_DIR/chain/holding/${CHAIN_ROLE}_"*  # idle marks left by this chain's earlier jobs
  _chain_env
  _chain_holds && hold_arg="--hold"
  # The guard and alerts need only the standard library: they run even if no torch python works,
  # so a broken environment counts as a crash instead of looping.
  # shellcheck disable=SC2086  # hold_arg is empty or one flag
  "$CHAIN_PY3" -m training.werewolf_rl.continuous guard --run_dir "$RUN_DIR" --role "$CHAIN_ROLE" \
      --prev_job "${PREV_JOB:-}" $hold_arg
  g=$?
  if [ "$g" -eq 4 ]; then
    hold="its previous jobs crashed 3 times in a row (see $RUN_DIR/chain/$CHAIN_ROLE.json)"
  elif [ "$g" -ne 0 ] && { [ "$g" -eq 3 ] || ! _chain_holds; }; then
    echo "[chain:$CHAIN_ROLE] not continuing (see $RUN_DIR/chain/$CHAIN_ROLE.json); cancelling its queued jobs"
    scancel -u "$USER" --name="$SLURM_JOB_NAME" --state=PENDING || true
    exit 0
  elif [ "$g" -ne 0 ]; then
    hold="the chain guard failed (exit $g; see this job's log)"
  fi
  if [ -n "${CHAIN_ENV_ERROR:-}" ]; then
    echo "[chain] $CHAIN_ENV_ERROR"
    if ! _chain_holds; then
      chain_notify gave_up "Werewolf RL ${RUN_NAME:-?}: job $SLURM_JOB_ID on $(hostname -s): $CHAIN_ENV_ERROR See README 'Python on the GPU nodes'."
      exit 1
    fi
    chain_notify gave_up "Werewolf RL ${RUN_NAME:-?}: job $SLURM_JOB_ID on $(hostname -s): $CHAIN_ENV_ERROR The job keeps its GPUs, idle; fix it (README 'Python on the GPU nodes') and run start.sh ${RUN_NAME:-run1} to retry at once."
    hold="${hold:-$CHAIN_ENV_ERROR}"
  fi
  [ -n "${PREV_JOB:-}" ] && ran="$(sacct -n -X -P -j "$PREV_JOB" -o ElapsedRaw 2>/dev/null | head -n 1 | tr -d ' ')"
  _chain_top_up
  # Every job that starts got GPUs: say which, where and whether they work.
  _chain_gpu_alert "${ran:-0}" "$hold"
  if _chain_is_extra_slot; then
    _chain_yield_watch &
  fi
  _chain_main "$hold"
  exit $?
}

# Main chains keep their GPUs until the time limit whatever happens (HOLD_ON_FAILURE=0
# turns that off); gap chains hand them back, so the keeper can re-place them.
_chain_holds() {
  [ "${HOLD_ON_FAILURE:-1}" = 1 ] && [ "${CHAIN_ROLE#gap}" = "$CHAIN_ROLE" ]
}

# _chain_main [HOLD_REASON]: run the node; if it stops early (or HOLD_REASON is set),
# hold the GPUs idle until STOP or a retry request, then run it again.
_chain_main() {
  local reason="${1:-}" rc=0
  while :; do
    if [ -z "$reason" ]; then
      chain_run_node
      rc=$?
      [ -f "$RUN_DIR/STOP" ] && return 0
      _chain_holds || return "$rc"
      reason="every GPU process of this job stopped before the time limit (last status $rc)"
      chain_notify gave_up "Werewolf RL $RUN_NAME: $CHAIN_ROLE job $SLURM_JOB_ID on $(hostname -s): $reason. It keeps its GPUs, idle, until its time limit; run start.sh $RUN_NAME to retry at once."
    fi
    _chain_wait_retry job "$reason" || return 0
    reason=""
    _chain_env  # a fix may change which python works
    [ -n "${CHAIN_ENV_ERROR:-}" ] && reason="$CHAIN_ENV_ERROR"
    [ -z "$reason" ] && chain_notify gpu "Werewolf RL $RUN_NAME: $CHAIN_ROLE job $SLURM_JOB_ID on $(hostname -s) is working again: its $(chain_gpus) idle GPUs restart now (${CHAIN_PY_INFO:-python ok})." --force
  done
}

# _chain_wait_retry WHAT REASON: keep the GPU(s) idle but held. Returns 1 at STOP,
# 0 when start.sh asks for a retry (RUN/chain/retry newer than this hold); the time
# limit ends the job otherwise.
_chain_wait_retry() {
  local mark="$RUN_DIR/chain/holding/${CHAIN_ROLE}_${SLURM_JOB_ID}_$1"
  mkdir -p "$RUN_DIR/chain/holding"
  echo "$(date '+%F %T') job $SLURM_JOB_ID on $(hostname -s), $1: $2" > "$mark"
  echo "[chain:$CHAIN_ROLE] $(date '+%F %T') $1 idle, GPUs kept: $2 (until STOP, start.sh ${RUN_NAME:-}, or the time limit)"
  while :; do
    if [ -f "$RUN_DIR/STOP" ]; then
      rm -f "$mark"
      return 1
    fi
    if [ "$RUN_DIR/chain/retry" -nt "$mark" ]; then
      rm -f "$mark"
      echo "[chain:$CHAIN_ROLE] $(date '+%F %T') retry requested: restarting $1"
      return 0
    fi
    sleep "${CHAIN_POLL:-60}"
  done
}

chain_gpus() {
  if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    echo "$SLURM_GPUS_ON_NODE"
  else
    IFS=, read -ra _devs <<< "${CUDA_VISIBLE_DEVICES:-0}"
    echo "${#_devs[@]}"
  fi
}

# Sbatch resource options for this chain's follow-up jobs: the chain's target
# from launch.env (CHAIN_OPTS_<role>), else exactly what this job was given.
chain_opts() {
  local var="CHAIN_OPTS_${CHAIN_ROLE}" n
  if [ -n "${!var:-}" ]; then
    echo "${!var}"
    return
  fi
  n="$(chain_gpus)"
  echo "--partition=$SLURM_JOB_PARTITION --gres=gpu:$n --cpus-per-task=$(( n * ${CPUS_PER_GPU:-4} ))" \
       "--mem=$(( n * ${MEM_PER_GPU_GB:-48} ))G ${CHAIN_TIME_MIN:+--time-min=$CHAIN_TIME_MIN} ${CHAIN_EXTRA_OPTS:-}"
}

# Keep CHAIN_DEPTH (gap chains: 1) follow-up jobs queued behind this one.
_chain_top_up() {
  local depth="${CHAIN_DEPTH:-1}" queued="" ok=0 attempt tail count out opts
  [ "${CHAIN_ROLE#gap}" != "$CHAIN_ROLE" ] && depth=1
  for attempt in 1 2 3 4 5; do
    if queued="$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t PD -o '%i' 2>/dev/null)"; then
      ok=1
      break
    fi
    sleep 30
  done
  queued="$(echo "$queued" | grep -E '^[0-9]+$' | sort -n)"
  count=$(echo "$queued" | grep -c . || true)
  tail="$(echo "$queued" | tail -n 1)"
  tail="${tail:-$SLURM_JOB_ID}"
  if [ "$ok" -eq 0 ]; then  # cannot see the queue: keep the chain alive with one follow-up
    echo "[chain:$CHAIN_ROLE] squeue failed; queueing one follow-up without checking"
    count=$((depth - 1))
  fi
  opts="$(chain_opts)"
  while [ "$count" -lt "$depth" ]; do
    out=""
    for attempt in 1 2 3 4 5; do
      # shellcheck disable=SC2086  # opts is a list of options
      if out=$(sbatch $opts --job-name="$SLURM_JOB_NAME" --dependency="afterany:$tail" \
          --export="ALL,PREV_JOB=$tail" "$CHAIN_SCRIPT" 2>&1); then
        break
      fi
      echo "[chain:$CHAIN_ROLE] sbatch failed (attempt $attempt): $out"
      out=""
      sleep 60
    done
    [ -z "$out" ] && break
    echo "[chain:$CHAIN_ROLE] $(date '+%F %T') job $SLURM_JOB_ID queued a follow-up after $tail: $out ($opts)" \
      | tee -a "$RUN_DIR/chain/${CHAIN_ROLE}_jobs.log"
    tail="${out##* }"
    count=$((count + 1))
  done
  echo "[chain:$CHAIN_ROLE] $count follow-up job(s) queued (depth $depth)"
}

chain_run_node() {
  local devs ntrain i pids=() pid status=0
  IFS=, read -ra devs <<< "${CUDA_VISIBLE_DEVICES:-0}"
  ntrain="${TRAIN_GPUS:-$(( ${#devs[@]} / 2 ))}"
  echo "[main] GPUs ${devs[*]}: first $ntrain train, the rest play"
  for i in "${!devs[@]}"; do
    if [ "$i" -lt "$ntrain" ]; then _chain_slot train "$i" "${devs[$i]}" & else _chain_slot play "$i" "${devs[$i]}" & fi
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  return "$status"
}

_chain_slot() {  # what slot dev
  local what=$1 slot=$2 dev=$3 restarts=0 rc log
  log="$RUN_DIR/logs/${CHAIN_ROLE}_${SLURM_JOB_ID}_${what}${slot}.log"
  mkdir -p "$RUN_DIR/logs"
  while [ ! -f "$RUN_DIR/STOP" ]; do
    if [ -f "$RUN_DIR/PAUSE" ]; then
      _chain_wait_resume "$what$slot" || return 0
    fi
    if [ "$what" = train ]; then
      # shellcheck disable=SC2086  # TRAINER_ARGS is a list of extra flags
      CHAIN_SLOT="$slot" CUDA_VISIBLE_DEVICES="$dev" python scripts/training/werewolf_continuous/trainer.py \
        --run_dir "$RUN_DIR" --role main --chain_names "${CHAIN_NAMES:-}" \
        --checkpoint_dir "$CHECKPOINT_DIR" --hyperparameters "$HYPERPARAMETERS" --model_path "$MODEL_PATH" \
        --extra_partitions "${EXTRA_PARTITIONS:-}" --max_extra "${MAX_EXTRA:-0}" --extra_name "${EXTRA_NAME:-}" \
        --extra_sbatch "$REPO/scripts/training/werewolf_continuous/extra.sbatch" \
        ${TRAINER_ARGS:-} >> "$log" 2>&1
    else
      # shellcheck disable=SC2086  # PLAYER_ARGS is a list of extra flags
      CHAIN_SLOT="$slot" CUDA_VISIBLE_DEVICES="$dev" python scripts/training/werewolf_continuous/player.py \
        --run_dir "$RUN_DIR" --slot "$slot" --role main --chain_names "${CHAIN_NAMES:-}" \
        --checkpoint_dir "$CHECKPOINT_DIR" --hyperparameters "$HYPERPARAMETERS" --model_path "$MODEL_PATH" \
        ${PLAYER_ARGS:-} >> "$log" 2>&1
    fi
    rc=$?
    [ -f "$RUN_DIR/STOP" ] && return 0
    if [ -f "$RUN_DIR/PAUSE" ]; then  # paused for an update: wait at the top, then run the new code
      restarts=0
      continue
    fi
    if [ "$rc" -eq 0 ]; then
      _chain_holds || return 0
      _chain_wait_retry "$what$slot" "$what.py exited before the time limit (status 0); log $log" || return 0
      restarts=0
      continue
    fi
    restarts=$((restarts + 1))
    echo "[main] $what slot $slot exited with $rc (restart $restarts/3)"
    if [ "$restarts" -ge 3 ]; then
      if ! _chain_holds; then
        chain_notify gave_up "Werewolf RL $RUN_NAME: a $what process on $(hostname -s) (job $SLURM_JOB_ID, GPU slot $slot) crashed 3 times in a row (exit $rc); that GPU is idle until the job ends. Log: $log"
        return "$rc"
      fi
      chain_notify gave_up "Werewolf RL $RUN_NAME: a $what process on $(hostname -s) (job $SLURM_JOB_ID, GPU slot $slot) crashed 3 times in a row (exit $rc). The job keeps that GPU, idle, until its time limit; fix it and run start.sh $RUN_NAME to restart it at once. Log: $log"
      _chain_wait_retry "$what$slot" "$what.py crashed 3 times in a row (exit $rc); log $log" || return 0
      restarts=0
      continue
    fi
    sleep "${CHAIN_RESTART_DELAY:-30}"
  done
}

# _chain_wait_resume WHAT: keep the GPU held (nothing runs on it) while RUN/PAUSE exists.
# Returns 0 when start.sh removes PAUSE, 1 at STOP.
_chain_wait_resume() {
  local mark="$RUN_DIR/chain/holding/${CHAIN_ROLE}_${SLURM_JOB_ID}_$1"
  mkdir -p "$RUN_DIR/chain/holding"
  echo "$(date '+%F %T') job $SLURM_JOB_ID on $(hostname -s), $1: paused (stop.sh --pause); start.sh resumes it" > "$mark"
  echo "[chain:$CHAIN_ROLE] $(date '+%F %T') $1 paused, GPU kept (until start.sh ${RUN_NAME:-} or STOP)"
  while [ -f "$RUN_DIR/PAUSE" ]; do
    if [ -f "$RUN_DIR/STOP" ]; then
      rm -f "$mark"
      return 1
    fi
    sleep "${CHAIN_POLL:-60}"
  done
  rm -f "$mark"
  [ -f "$RUN_DIR/STOP" ] && return 1
  echo "[chain:$CHAIN_ROLE] $(date '+%F %T') resumed: starting $1"
  return 0
}

# Is this the gap slot on a partition where we also take GPUs beyond the chain targets
# (EXTRA_GPU_PARTITIONS, launch.env)? Then _chain_yield_watch runs next to the job.
_chain_is_extra_slot() {
  [ "${CHAIN_ROLE#gap}" != "$CHAIN_ROLE" ] && case ",${EXTRA_GPU_PARTITIONS:-}," in
    *",${SLURM_JOB_PARTITION:-?},"*) true ;;
    *) false ;;
  esac
}

# Extra GPUs: every YIELD_POLL (60) s ask the keeper whether another user's waiting job needs this
# job's GPUs (keeper.py yield_reason: only when the chains there hold their target without it, and
# the waiting job fits once these GPUs are free). If so, say so and cancel this job and its queued
# follow-up: the GPUs are free at once.
_chain_yield_watch() {
  local why
  while sleep "${YIELD_POLL:-60}"; do
    [ -f "$RUN_DIR/STOP" ] && return 0
    if why="$("${CHAIN_PY3:-python}" -m training.werewolf_rl.keeper yield-check --run_dir "$RUN_DIR" \
        --job "$SLURM_JOB_ID" 2>/dev/null)"; then
      chain_notify gpu "Werewolf RL ${RUN_NAME:-?}: gave back $(chain_gpus)x ${CHAIN_GPU_MODEL:-GPU} on $(hostname -s) (partition $SLURM_JOB_PARTITION, $CHAIN_ROLE job $SLURM_JOB_ID, extra GPUs): $why. Our chains keep theirs." --force
      scancel -u "$USER" --name="$SLURM_JOB_NAME" || true
      return 0
    fi
  done
}

# chain_notify EVENT MESSAGE: send an alert (no-op without ~/.config/werewolf_rl/notify.env).
chain_notify() {  # EVENT MESSAGE [--force]
  echo "[alert] $1: $2"
  "${CHAIN_PY3:-python}" -m training.werewolf_rl.notify send --event "$1" --run_dir "$RUN_DIR" ${3:-} "$2" || true
}

# _chain_gpu_alert RAN HOLD: the "got GPUs" alert every job sends when it starts -- GPU model,
# count, partition, node, chain, time limit, handover, python, and how many GPUs the run
# now holds. RAN = seconds the previous job ran (0: this chain had no GPUs before).
_chain_gpu_alert() {
  local ran="${1:-0}" hold="${2:-}" n model end held how status
  n="$(chain_gpus)"
  model="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/, / /')"
  CHAIN_GPU_MODEL="${model:-GPU}"
  end="$(squeue -h -j "$SLURM_JOB_ID" -o '%e' 2>/dev/null | head -n 1)"
  end="$(date -d "$end" '+%a %H:%M' 2>/dev/null || echo '?')"
  held="$(squeue -h -u "$USER" -t R -o '%j %b' 2>/dev/null | awk -v p="wwc-${RUN_NAME:-}-" 'index($1, p) == 1 {
    if (match($2, /[0-9]+$/)) s += substr($2, RSTART)} END {print s + 0}')"
  if [ -n "${PREV_JOB:-}" ] && [ "$ran" != 0 ]; then
    how="takes over from job $PREV_JOB"
    # A renewal -- the chain's previous job held these GPUs on this node until moments ago
    # (24 h handovers, or the short pieces --time-min allows) -- is no new GPU: log only.
    local prev_node prev_end gap
    prev_node="$(sacct -n -X -P -j "$PREV_JOB" -o NodeList 2>/dev/null | head -n 1)"
    prev_end="$(date -d "$(sacct -n -X -P -j "$PREV_JOB" -o End 2>/dev/null | head -n 1)" +%s 2>/dev/null || echo 0)"
    gap=$(( $(date +%s) - prev_end ))
    if [ "$prev_node" = "$(hostname -s)" ] && [ "$gap" -ge 0 ] && [ "$gap" -le 900 ] && [ -z "$hold" ]; then
      echo "[chain:$CHAIN_ROLE] renewed the same ${n} GPU(s) on $(hostname -s): job $PREV_JOB -> $SLURM_JOB_ID, until $end (no alert)"
      return 0
    fi
  else
    how="new for this chain"
  fi
  status="working (${CHAIN_PY_INFO:-python ok})"
  [ -n "$hold" ] && status="IDLE, GPUs kept: $hold"
  _chain_is_extra_slot && status="$status. Gap slot: GPUs beyond our chain targets are given back as soon as another user's job needs them"
  chain_notify gpu "Werewolf RL ${RUN_NAME:-?}: got ${n}x ${model:-GPU} on $(hostname -s) (partition ${SLURM_JOB_PARTITION:-?}) - $CHAIN_ROLE job $SLURM_JOB_ID, $how, runs until $end. $status. The run now holds ${held:-?} GPUs." --force
}

_chain_env() {
  if [ "${SLURM_JOB_PARTITION:-}" = "teaching" ]; then
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"  # PCIe T4 nodes: NCCL P2P hangs
  fi
  _chain_python
}

# Pick a python that runs CUDA on this node's GPUs (see README, "Python on the GPU nodes"):
#   1. the python on PATH (the repo's .venv: torch cu130; login/teaching/dgxh100 nodes);
#   2. JOB_VENV: a venv on /home with the same Python build and .venv's packages --
#      .venv/bin/python links to /usr/local/miniforge3, which the dgx/dgxh100 nodes lack;
#   3. JOB_VENV with JOB_TORCH_OVERLAY first on PYTHONPATH: torch cu126 for the V100
#      nodes, whose drivers (525 / 535) can't run cu130 builds (which also lack sm_70).
#      On drivers older than 535 the overlay's sitecustomize.py loads NVIDIA's CUDA
#      forward-compatibility driver (cuda-compat-12-2) first; see README 2.8.
_chain_python() {
  local venv="${JOB_VENV:-$HOME/.local/share/werewolf_rl/venv-nodes}"
  local overlay="${JOB_TORCH_OVERLAY:-$HOME/.local/share/werewolf_rl/torch-cu126}"
  local check='import torch, numpy, transformers
x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
torch.nn.functional.scaled_dot_product_attention(x[None, None], x[None, None], x[None, None]).sum().item()'
  local info='import os, torch; c = os.environ.get("WEREWOLF_CUDA_COMPAT_LOADED")
print(f"torch {torch.__version__}" + (f", NVIDIA CUDA compat {os.path.basename(c.split()[0])} on driver {c.split()[-1][:-1]}" if c else ""))'
  CHAIN_PY3=python CHAIN_ENV_ERROR="" CHAIN_PY_INFO=""
  if timeout 300 python -c "$check" >/dev/null 2>&1; then
    CHAIN_PY_INFO="$(timeout 120 python -c "$info" 2>/dev/null)"
    echo "[chain] $(hostname -s): python $(command -v python) ($CHAIN_PY_INFO)"
    return 0
  fi
  if [ -x "$venv/bin/python" ] && timeout 300 "$venv/bin/python" -c "$check" >/dev/null 2>&1; then
    export PATH="$venv/bin:$PATH" VIRTUAL_ENV="$venv"
  elif [ -x "$venv/bin/python" ] && [ -d "$overlay" ] && PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}" \
      timeout 300 "$venv/bin/python" -c "$check" >/dev/null 2>&1; then
    export PATH="$venv/bin:$PATH" VIRTUAL_ENV="$venv" PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}"
  else
    CHAIN_ENV_ERROR="no python here runs torch on the GPU (PATH python $(command -v python), JOB_VENV $venv)."
    for CHAIN_PY3 in "$venv/bin/python" python3 /usr/bin/python3; do
      "$CHAIN_PY3" -c 'import sys; sys.exit(sys.version_info < (3, 8))' >/dev/null 2>&1 && return 0
    done
    return 0
  fi
  CHAIN_PY_INFO="$(timeout 120 python -c "$info" 2>/dev/null)"
  echo "[chain] $(hostname -s): python $(command -v python) ($CHAIN_PY_INFO)${PYTHONPATH:+, PYTHONPATH=$PYTHONPATH}"
}
