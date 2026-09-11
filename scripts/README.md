# scripts/

All scripts are grouped by purpose. Run every command from the repo root
(paths below are relative to it).

## `data_prep/` -- download and preprocess raw data

- `download_data.sh` -- populates `$RAW_DIR` from HF/GitHub for the sources
  named in `$SOURCES` (e.g. `SOURCES="meld ami werewolf"`).
- `distill_qwen.sbatch` -- GPU job that generates the `qwen_distill`
  self-distillation source (not downloaded, produced by running the base model).
- `preprocess_meld_ami_werewolf.sbatch` -- CPU job (`teaching` partition):
  converts raw MELD + timed AMI + Werewolf data into matrix-formatted Arrow
  shards + `manifest.json`.

## `training/` -- SLURM training entrypoints

- `train_meld_ami_werewolf.sbatch` -- current training job (all 3 sources,
  equally weighted ~1/3 each, on a V100). Most hyperparameters are
  environment-overridable. Set `INIT_FROM` + `INIT_CHECKPOINT_PREFER` for a
  fresh warm start, or `RESUME_FROM` for an exact continuation when the full
  training state is available; never set both.

## `search/` -- persistent rotating HPO

- CPU-only controller + watchdog and a five-H100 rotating worker pool for the
  month-long search.
- Atomic shared-filesystem tickets, resumable time-sliced training, broad
  evaluation, remote Llama judging, confidence-aware scoring, and ASHA-style
  promotion/pruning.
- See `scripts/search/README.md` for calibration, smoke-test, and launch steps.

## `eval/` -- checkpoint evaluation, probing, and demo tools

- `evaluate_checkpoint_suite.py` -- the fixed multi-prompt handoff/listener/
  repetition/privacy probe suite for one checkpoint (small, greedy, fast).
- `demo_handoff_variation.py` (+ `.sbatch` launcher) -- the BROAD stochastic
  sweep (prompts x agent counts x seeds x context kinds, with
  chain-continuation for 2nd/3rd handoffs). Context kinds include
  `real_prefix` -- a genuine multi-turn excerpt pulled straight from
  AMI/MELD/Werewolf (not synthetic filler) -- and `max_new_tokens` defaults to
  48 (was 16) so longer stretches of the resulting "conversation" play out.
  Always broad-sweep-verify any promising small-suite result before trusting
  it (see `SUCCESS_STORIES.md`).
- `probe_running_jobs.sbatch` -- runs `evaluate_checkpoint_suite.py` against
  each listed run's CURRENT (`last`) checkpoint, for hourly checkups on
  still-training jobs. `RUN_NAMES="run_a run_b" sbatch scripts/eval/probe_running_jobs.sbatch`.
- `turn_taking_probe.py` -- interactive/scriptable single-scenario probe
  (loads a checkpoint, builds one scenario, generates, prints the result).
- `multi_agent_tokens.py` -- watch multiple agents generate tokens side by
  side as a table (one column per agent).
- `demo_freeform_conversation.py` -- seed a free-form multi-agent
  conversation from one prompt and render the resulting transcript.
- `run_best_model_conversations.sbatch` -- Slurm wrapper around the free-form
  demo; accepts `CHECKPOINT_DIR`, `SEED_TEXT`, `NUM_AGENTS`,
  `MAX_NEW_TOKENS`, `TEMPERATURE`, `SEED`, and `OUTPUT` overrides.
- `demo_grid_render.py` -- render the exact `[Agent x Time]` grid for a
  specific (prompt, num_agents, seed) trial, for documenting an example.
- `verify_cross_agent.py` -- logit-level proof that agents genuinely
  condition on each other's context (not just generating independently).

## `analysis/` -- cross-run comparison and plotting

- `plot_metrics.py` -- exports one run's `metrics.jsonl` as PNG plots
  (content/activity/reward/total loss, activity accuracy, speak rates).
- `analyze_training_race.py` -- cross-run diagnostics (plateau/overfit/
  underfit/activity-collapse/reward-conflict detection) over the race's arms.

## `ops/` -- cluster operations/infrastructure automation

- `gpu_yield_daemon.py` (+ `.sbatch` launcher) -- always-on daemon (teaching
  partition, 0 GPUs) that codes `.cursor/rules/yield-gpus-when-others-pending.mdc`
  into an automated poll loop instead of a manual hourly-checkup action: when
  another user's job appears PENDING on `dgx`/`dgxh100`, holds our own
  pending jobs on that partition and cancels enough of our lowest-sunk-cost
  running jobs to accommodate it, then reclaims (releases holds + attempts
  to resubmit what it cancelled) once that job starts running or leaves the
  queue. **Defaults to `--dry-run`** (logs every action it would take to
  `logs/gpu_yield_daemon/decisions.jsonl` without touching real jobs) --
  read the module docstring's safety rails before passing `LIVE=1`. See the
  script header for launch commands. Also drives `job_queue.py` every tick
  (see below): computes each partition's current GPU usage (any job under
  `--our-user`, whether queue-managed or submitted directly via `sbatch` --
  manually-run jobs are auto-detected into the budget accounting) against a
  fixed cap (`--budget-dgx-gpus`/`--budget-h100-gpus`, default 16/4 = 2 DGX
  nodes + 4 H100 GPUs, the user's standing usage cap), and submits the
  highest-priority fitting `queued` entries when headroom exists. A
  queue-managed job that gets yield-cancelled re-enters the queue (status
  back to `queued`, `RESUME_FROM`/`NUM_STEPS` merged into its env) instead
  of being immediately resubmitted, so priority/pausing still apply.
- `job_queue.py` -- the manipulable priority queue itself: a small CLI
  (`add`/`list`/`show`/`set-priority`/`bump`/`pause`/`resume`/`remove`/
  `cancel`) over `logs/job_queue/queue.json`. The user keeps submitting to
  it; the agent reorders/pauses/removes entries as priorities change.
  `gpu_yield_daemon.py` (above) is the only thing that actually launches a
  `queued` entry via `sbatch`, once GPU budget allows. Jobs submitted
  directly via `sbatch`/Slurm (bypassing this queue entirely) still count
  against the same budget -- see `--our-user`'s `squeue` usage above.

## `archive/` -- historical, no-longer-reusable scripts kept for reference

- `launch_training_race.py` -- one-time launcher for the original 4-arm race
  (07-18). Superseded by ad hoc `sbatch` calls with env overrides since; kept
  for the historical methodology, not meant to be re-run as-is.

## Removed (superseded, findings preserved in `TRAINING_RUN_LOG.md`/`SUCCESS_STORIES.md`)

Single- and two-source preprocessing/training scripts from earlier stages of
the project (`train_matrix_qwen.sbatch`, `train_meld.sbatch`,
`train_meld_ami.sbatch`, `preprocess_data.sbatch`, `preprocess_meld_ami.sbatch`)
and one-off diagnostic/smoke scripts whose results are already written up
elsewhere (`diag_warmup_context.py`, `regen_single_case_grid.py` +
`regen_grid_cases.sbatch`, `smoke_demo_handoff_variation.py` + `.sbatch`,
`probe_completed_kitchensink.sbatch`) were deleted during the 2026-07-21
cleanup rather than archived, since their content is fully superseded or
duplicated.
