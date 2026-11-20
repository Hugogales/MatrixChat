#!/bin/bash
# Stop the always-on Werewolf RL loop.
#
#   scripts/training/werewolf_continuous/stop.sh [RUN_NAME] [--now]
#
# Default (graceful): writes RUN/STOP. The training leader finishes its current
# step, saves and exits; helpers and players finish their current task / game
# and exit; queued successor jobs see STOP and end without running.
# --now: additionally cancel every job of the run immediately (training loses
# the step in progress -- the last checkpoint is kept -- and players their game).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_NAME="${RUN_NAME:-run1}"
NOW=0

for arg in "$@"; do
  case "$arg" in
    --now) NOW=1 ;;
    *) RUN_NAME="$arg" ;;
  esac
done

RUN_DIR="${RUN_DIR:-$REPO/checkpoints/werewolf_rl/continuous/$RUN_NAME}"
mkdir -p "$RUN_DIR"
touch "$RUN_DIR/STOP"
echo "Wrote $RUN_DIR/STOP"

if [ "$NOW" -eq 1 ]; then
  for name in "wwc-$RUN_NAME-h100" "wwc-$RUN_NAME-v100" "wwc-$RUN_NAME-extra"; do
    scancel -u "$USER" --name="$name" && echo "Cancelled jobs named $name"
  done
else
  echo "Graceful stop: running jobs wind down on their own; queued successors will not run."
fi