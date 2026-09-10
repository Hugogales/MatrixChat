# Month-long rotating HPO search

The search uses three persistent SLURM jobs:

- `controller.sbatch`: CPU-only teaching-node controller and sole owner of
  `search_state.json`.
- `worker_pool.sbatch`: one H100 allocation requesting exactly five GPUs;
  five workers rotate through atomic train/eval tickets.
- `watchdog.sbatch`: CPU-only controller-lease watchdog.

All durable artifacts live under `SEARCH_DIR`; model state remains under the
normal `checkpoints/<candidate_id>/` layout and is linked from each candidate
artifact directory.

Before a production launch:

1. Run the repository tests.
2. Run `controller_smoke.sbatch` + `worker_pool_smoke.sbatch`.
3. Calibrate the remote judge with `calibrate_judge.py`. The production config
   refuses promotion when its configured calibration report is not
   `launch_ready`.

Production:

```bash
SEARCH_DIR="$PWD/searches/month_2026_08"
sbatch --export=ALL,SEARCH_DIR="$SEARCH_DIR",SEARCH_CONFIG="$PWD/scripts/search/configs/month_h100.json" \
  scripts/search/controller.sbatch
sbatch --export=ALL,SEARCH_DIR="$SEARCH_DIR" scripts/search/watchdog.sbatch
sbatch --export=ALL,SEARCH_DIR="$SEARCH_DIR" scripts/search/worker_pool.sbatch
```

The controller and worker pool self-chain before day five. Candidate training
slices stop via `main.py --max_runtime_minutes`, save optimizer/scheduler/RNG
state, and may resume on any worker/GPU.
