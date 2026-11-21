#!/bin/bash
# Pause or stop the always-on Werewolf RL loop.
#
#   scripts/training/werewolf_continuous/stop.sh [RUN_NAME] [--pause | --now]
#
# --pause (for updates; keeps the GPUs and the place in the queue): writes
#   RUN/PAUSE. Players abandon their current game and trainers finish their
#   current step (it is saved), then nothing runs, but every job keeps its GPUs
#   and its queued follow-ups, and the keeper keeps the queues topped up. Make
#   the update, then start.sh removes PAUSE and every GPU starts again with the
#   new code within a minute.
# Default (graceful stop): writes RUN/STOP, cancels the run's queued (not yet
#   running) jobs and removes the keeper's cron line on this login node. The
#   training leader finishes its current step, saves and exits; helpers and
#   players finish their current task / game and exit. The run loses its place
#   in the queue: a later start.sh queues new jobs at the back of the line.
# --now: additionally cancel every running job of the run immediately (training
#   loses the step in progress -- the last checkpoint is kept -- and players their game).
# Every mode sends an ntfy status alert with the queue (section 8 of the README).
# Restart with start.sh (it removes STOP and PAUSE).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
RUN_NAME="${RUN_NAME:-run1}"
MODE=stop

for arg in "$@"; do
  case "$arg" in
    --now) MODE=now ;;
    --pause) MODE=pause ;;
    *) RUN_NAME="$arg" ;;
  esac
done

RUN_DIR="${RUN_DIR:-$REPO/checkpoints/werewolf_rl/continuous/$RUN_NAME}"
mkdir -p "$RUN_DIR"

announce() {
  if [ -f "$RUN_DIR/launch.json" ]; then
    python -m training.werewolf_rl.keeper announce --run_dir "$RUN_DIR" "$1" || true
  fi
}

if [ "$MODE" = pause ]; then
  touch "$RUN_DIR/PAUSE"
  echo "Wrote $RUN_DIR/PAUSE: players abandon their game, trainers finish their step, then every"
  echo "job keeps its GPUs idle (nothing is cancelled; the keeper keeps running)."
  echo "Resume (with whatever code is on disk then): scripts/training/werewolf_continuous/start.sh $RUN_NAME"
  announce "Werewolf RL $RUN_NAME stopped for updates: games and training are paused; every job keeps its GPUs and its place in the queue until start.sh resumes it."
  exit 0
fi

touch "$RUN_DIR/STOP"
echo "Wrote $RUN_DIR/STOP"

if [ -f "$RUN_DIR/launch.json" ]; then
  python -m training.werewolf_rl.keeper uninstall-cron --run_dir "$RUN_DIR" || true
fi

names="$(squeue -h -u "$USER" -o '%j' | grep "^wwc-$RUN_NAME-" | sort -u || true)"
for name in $names; do
  if [ "$MODE" = now ]; then
    scancel -u "$USER" --name="$name" && echo "Cancelled all jobs named $name"
  else
    scancel -u "$USER" --name="$name" --state=PENDING && echo "Cancelled queued jobs named $name"
  fi
done
if [ "$MODE" = now ]; then
  announce "Werewolf RL $RUN_NAME stopped now (stop.sh --now): every job was cancelled."
else
  echo "Graceful stop: running jobs wind down on their own."
  announce "Werewolf RL $RUN_NAME stopped (stop.sh): queued jobs cancelled, running jobs wind down."
fi
