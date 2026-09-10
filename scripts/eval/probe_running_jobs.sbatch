#!/bin/bash
# Behavioral "smoke test" for the training race: runs the fixed
# handoff/listener/repetition probe suite against each run's CURRENT
# ("last") checkpoint -- not just the best-val one -- so we can track how
# turn-taking behavior evolves hour by hour while jobs are still training.
#
# Usage:
#   RUN_NAMES="run_a run_b run_c" sbatch scripts/eval/probe_running_jobs.sbatch
#
# For each run in RUN_NAMES, writes:
#   logs/<run>/checkpoint_probe_latest.json   -- full probe output (overwritten each hour)
#   logs/<run>/checkpoint_probe_history.jsonl -- one compact summary line appended per hour
#SBATCH --account=undergrad_research
#SBATCH --job-name=probe_running
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CONTAINER="/data/containers/msoe-tf2x.sif"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
RUN_NAMES="${RUN_NAMES:?Set RUN_NAMES to a space-separated list of run ids}"

for run in $RUN_NAMES; do
  ckpt_dir="checkpoints/$run/last"
  if [[ ! -f "$ckpt_dir/model_state.pt" ]]; then
    echo "[skip] $run: no checkpoints/$run/last/model_state.pt yet"
    continue
  fi
  echo "=== probing $run (checkpoints/$run/last) ==="
  singularity exec --nv -B /data:/data "$CONTAINER" python scripts/eval/evaluate_checkpoint_suite.py \
    --checkpoint_dir "$ckpt_dir" \
    --model_path "$MODEL_PATH" \
    --output "logs/$run/checkpoint_probe_latest.json" \
    --device cuda
  singularity exec --nv -B /data:/data "$CONTAINER" python - "$run" << 'PYEOF'
import json, sys, time
import torch
run = sys.argv[1]
summary = json.load(open(f"logs/{run}/checkpoint_probe_latest.json"))
step = None
try:
    ckpt = torch.load(f"checkpoints/{run}/last/model_state.pt", map_location="cpu")
    step = (ckpt.get("metadata") or {}).get("step")
except FileNotFoundError:
    pass
row = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "step": step,
    "clean_handoff_rate": summary["clean_handoff_rate"],
    "no_listener_response_rate": summary["no_listener_response_rate"],
    "unwanted_interruption_rate": summary["unwanted_interruption_rate"],
    "overlap_rate": summary["overlap_rate"],
    "distinct_1": summary["distinct_1"],
    "distinct_2": summary["distinct_2"],
    "repeated_4gram_fraction": summary["repeated_4gram_fraction"],
    "calibration_summary": summary.get("calibration_summary"),
    "scaling_summary": summary.get("scaling_summary"),
    "secret_scenarios_summary": summary.get("secret_scenarios_summary"),
}
with open(f"logs/{run}/checkpoint_probe_history.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row, indent=2))
PYEOF
done
echo "probe_running_jobs done."
