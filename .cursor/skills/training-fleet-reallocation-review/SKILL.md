---
name: training-fleet-reallocation-review
description: Performs a full six-hour MatrixChat training portfolio review, deciding which manual jobs to keep or cancel, how to divide the fixed GPU budget between targeted experiments and HPO, and whether to repair, rebuild, broaden, or narrow the Bayesian search. Use every six hours during the training race, or when asked to reconsider the fleet/search allocation.
---

# Training fleet reallocation review

Act autonomously. Do not ask the user to choose jobs or approve an
allocation; inspect the evidence and use best judgment. Keep the total GPU
count unchanged unless the user explicitly changes it.

Note: as of 2026-08-04 there are TWO independent resource pools, each with
its own fixed budget -- do not move GPUs between them without an explicit
user instruction. **The DGX/V100 pool is SHARED with other users on this
account** -- always check `squeue -p dgx` for other users' pending jobs
each review, not just your own job health. A job requesting close to a
full node's memory (`scontrol show job <id>` -> `ReqTRES` mem near a
node's `CfgTRES` mem) can NEVER schedule as long as every DGX node has at
least one of your jobs on it, even if individual GPUs are technically
idle -- the fix is to fully vacate at least one whole node, not just
reduce total GPU count.
1. The DGX/V100 fleet, reduced 2026-08-04 from 12 HPO + 2 targeted manual
   (14 V100s) to **7 HPO + 2 targeted manual (9 V100s)** after another
   user's jobs were blocked from scheduling (see `TRAINING_RUN_LOG.md`
   08-04 13:24 entry for the diagnosis). The freed 5-GPU worker pool
   (`dh-dgx1-3`) is intentionally left fully vacated for shared use --
   do not backfill it without an explicit user instruction to reclaim it.
2. A separate H100 phase-two search (`searches/h100_2026_08_phase2`,
   `matrix_h100p2_*` jobs on the `dgxh100` partition), launched with **4
   workers** on 2026-08-05 because another user occupied 3 of the previously
   available 7 GPUs during the restart. Scale only to genuinely idle capacity
   after checking `scontrol show node dh-dgxh100-1`; never assume the old
   7-GPU allocation is still available. It retains the wider
   batch_size/max_flat_len/lora_r H100 space, adds five controlled dataset
   presets, and first rescores imported phase-one leaders using corrected
   strict+dirty chain metrics. Health-check it the same way (job/.err
   tracebacks, canonical candidate status distribution, leaderboard,
   `srun --overlap nvidia-smi` for real memory/utilization evidence) but do
   not fold its GPU count into the DGX budget math.

## 1. Establish the real allocation and health

- Read the latest user instruction and
  `.cursor/skills/training-race-hourly-checkup/SKILL.md`.
- Inspect `squeue` and `scontrol` for every controller, watchdog, worker
  pool, verification job, and manual arm. Count allocated GPUs, not job
  names.
- Check each job's own errors and `sacct` for jobs that disappeared.
- Inspect recent candidate status/failure distributions in
  `search_state.json`. A RUNNING pool and zero top-level tracebacks do not
  prove ticket health; follow
  `.cursor/rules/search-worker-handled-failures.mdc`.

## 2. Re-evaluate every manual arm

For every manual arm:

1. Generate and view fresh `probe.png` and relevant reward plots.
2. Read the last several raw probes, not only the composite score.
3. Read recent decoded text.
4. Check broad-verification results and checkpoint availability.
5. Classify it:
   - **keep**: unresolved, high-information comparison or replication;
   - **cancel**: question answered, broadly refuted, dominated, collapsed,
     or redundant;
   - **verify**: repeated small-suite signal merits a 576-trial sweep.

Do not keep training merely because a job is healthy. Preserve useful
checkpoints before cancellation.

## 3. Audit whether HPO is learning

Inspect:

- leaderboard and score-by-rung histories;
- queue throughput and claimed/pending balance;
- recent sampled configurations and optimizer trial numbers;
- failure causes, judge health, gates, and decoded outputs;
- whether new candidates are meaningfully different or only duplicates;
- whether the search is approaching candidate limits or starving workers.

Treat infrastructure failures separately from bad hyperparameters. Recover
completed checkpoints rather than retraining them. Never tell Optuna a
false zero for an evaluation/import/service failure.

## 4. Decide the GPU split

Allocate the fixed total toward whichever side currently has higher
information value:

- Favor HPO when manual questions are answered, many candidates await
  evaluation, or a clean optimizer can efficiently explore.
- Favor manual jobs when a precise seed-replication, ablation, broad
  verification, or final-production run cannot be represented well by the
  search.

The current baseline after the 2026-08-02 review is **12 HPO GPUs + 2
targeted manual GPUs = 14 total**, but re-derive this each review. Never
silently increase the total.

## 5. Decide whether to continue or rebuild the search

Do not restart merely because the leaderboard is unchanged. Rebuild or
restart when evidence shows:

- optimizer history is contaminated by infrastructure failures;
- long-lived processes need code changes;
- the search space/objective conflicts with broad-verified behavior;
- valid results have converged enough to justify a narrower phase;
- candidate/population limits or worker counts no longer fit the budget.

When rebuilding:

1. Stop controller/watchdog/workers.
2. Back up state and optimizer storage.
3. Reconcile claimed tickets.
4. Seed the new Bayesian study only with valid scored history.
5. Recover completed checkpoints/evaluations.
6. Restart every long-lived process so code changes take effect.

Prefer a clean history-informed TPE restart over discarding valid evidence.
Do not narrow prematurely when seed sensitivity remains unresolved.

## 6. Optimize for the real goal

The goal is one robust final-production configuration, not the highest
small-suite spike. Weight broad verification, seed robustness, decoded
quality, low overlap, diversity, and chains together.

Known scoring caveat: an ungated composite can still hide unacceptable
overlap (candidate40/DH scored 5.75 despite 79% overlap and only 1/576 clean
handoffs). Always inspect raw metrics and text.

When independent broad-verified runs converge tightly enough, explicitly
shift GPUs from exploration toward seed confirmation and the final run.

## 7. Execute and verify

- Make keep/cancel/reallocate/restart decisions in the same cycle.
- Fix bugs found during the review and add a durable rule for new failure
  modes.
- After launches, verify Slurm state, traceback count, actual GPU count,
  queue claims, configuration, and live progress.
- Update `TRAINING_RUN_LOG.md`, `SUCCESS_STORIES.md`, and the hourly skill if
  the standing allocation changes.
- Re-arm the next six-hour review through the session loop mechanism using a
  unique `AGENT_LOOP_TICK_MATRIX_SIXHOUR` sentinel and a monitored-output
  regex. A plain background `sleep && echo` loop cannot wake the agent even
  when `ps` shows it alive. Before starting a replacement, verify no matching
  monitored loop already exists and kill both parent/child only when replacing
  it; duplicate permanent loops previously accumulated by cycle 4.
