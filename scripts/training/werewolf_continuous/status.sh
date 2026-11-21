#!/bin/bash
# Show the state of the always-on Werewolf RL loop: GPUs held per chain, what every GPU
# is doing (game / step progress, idle reasons), training, games, keeper and alerts.
#
#   scripts/training/werewolf_continuous/status.sh [RUN_NAME] [--jobs] [--report] [--send]
#     --jobs    also list every job of the run, queued ones included (squeue)
#     --report  print the hourly ntfy GPU report instead (what the phone gets)
#     --send    send that report now
#
# The page is training/werewolf_rl/report.py (format_status); the keeper sends the
# report every hour (launch.json report_minutes, start.sh REPORT_MINUTES).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
RUN_NAME="${RUN_NAME:-run1}"
JOBS=0
MODE=status
for arg in "$@"; do
  case "$arg" in
    --jobs) JOBS=1 ;;
    --report) MODE=report ;;
    --send) MODE=send ;;
    *) RUN_NAME="$arg" ;;
  esac
done
RUN_DIR="${RUN_DIR:-$REPO/checkpoints/werewolf_rl/continuous/$RUN_NAME}"
if [ ! -d "$RUN_DIR" ]; then
  echo "no run directory $RUN_DIR"
  exit 1
fi

case "$MODE" in
  status) python -m training.werewolf_rl.report status --run_dir "$RUN_DIR" ;;
  report) python -m training.werewolf_rl.report hourly --run_dir "$RUN_DIR" ;;
  send) python -m training.werewolf_rl.report hourly --run_dir "$RUN_DIR" --send ;;
esac
if [ "$JOBS" -eq 1 ]; then
  echo
  echo "ALL JOBS"
  squeue -u "$USER" -o "%.10i %.10P %.22j %.9T %.10M %.10L %R" | awk -v p="wwc-$RUN_NAME-" 'NR==1 || index($3, p)==1'
fi
