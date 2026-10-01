---
name: training-race-hourly-checkup
description: Runs a full hourly (or on-demand) checkup of the MatrixChat turn-taking training race -- a personally-managed DGX/V100 fleet plus a separate month-long hyperparameter search (its own persistent controller/watchdog/worker-pool, currently on DGX/V100). Use whenever the hourly/periodic search-review timer fires, when the user asks "how is it going" / "check on the jobs" / "any updates", or whenever resuming after a monitoring gap. Covers probing every job's raw metrics (not just composite scores), applying lessons from prior experiments and the hyperparameter search, updating SUCCESS_STORIES.md with verified findings, keeping the personally-managed fleet at its current job budget, verifying every launched job's hyperparameters, fixing bugs on sight, and re-arming the next check.
---

# Training race hourly checkup

Do a genuine inspection every cycle, not a glance. Never trust a composite
score or a job's mere "RUNNING" state without reading what's actually behind
it.

**The end goal of this entire race is to converge on one optimal
hyperparameter configuration for a final, longer/bigger production training
run** -- not to find "the best model right now." Every DGX arm and search
candidate is a data point toward that config. Treat every checkup as an
opportunity to update your working model of which levers matter (see
Section 9) and to notice when confidence is high enough to propose ending
exploration and launching the final run.

As of 2026-08-05 there are two clean phase-two searches:
`searches/month_2026_08_phase2` on 7 DGX/V100 workers and
`searches/h100_2026_08_phase2` on however many H100s are genuinely free
(4 at launch because the partition is shared). Their preserved phase-one
directories remain read-only historical evidence. Phase two first rescores
21 imported strong checkpoints with v3 strict+dirty chain evaluation and
does not sample fresh candidates until every import finishes.

Monitoring reliability: a plain background `while true; sleep; echo` process
is NOT sufficient even if `ps` shows it alive -- without a monitored-output
notification its echo cannot wake the agent to perform the check. Arm the
hourly loop through the session loop mechanism with a unique
`AGENT_LOOP_TICK_MATRIX_HOURLY` sentinel and monitored-output regex, smoke
check it once, and do not create a duplicate while it remains active.

## 1. Job health

```bash
squeue -u $USER -o "%.10i %.10P %.30j %.8T %.12M" | grep -v bench_gen
```

`bench_gen` (or any job you don't recognize) may belong to someone else on
the shared account -- confirm with the user once, don't re-flag it after.

The `dgx` partition is shared with other users. Periodically check
`squeue -p dgx` for OTHER users' pending jobs, not just your own -- a job
stuck on `(Resources)`/`(Priority)` for a while can mean the fleet is
crowding them out even if your own jobs look healthy. If found, check
whether their `ReqTRES` mem is close to a full node's `CfgTRES` mem (via
`scontrol show job <id>`/`scontrol show node <name>`) -- such jobs need at
least one FULLY EMPTY node, which individual idle GPUs cannot satisfy if
every node still has some of your jobs on it. As of 2026-08-04 the DGX
budget is 7 HPO + 2 manual = 9 V100s (reduced from 14 for exactly this
reason); re-derive the current number from the user's latest instruction.

For every DGX/H100 arm and the controller, grep its own `.err` for
tracebacks -- don't assume "still RUNNING" means healthy:

```bash
grep -ci traceback logs/slurm/<job_kind>_<jobid>.err
```

If a job dropped off the queue, `sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,Elapsed -X`
to see if it COMPLETED naturally, FAILED, or was CANCELLED -- each implies a
different next step (read its final result / diagnose+fix / already handled).

If H100 `judge_quality` values look suspiciously flat/neutral across many
candidates, check `require_calibrated_judge`/`judge_calibration_path` in
`search_state.json`'s config and confirm that calibration file's
`launch_ready` is still `true`, and spot-check a candidate's score entry for
a non-null `judge_error` -- the controller silently falls back to
`needs_review`/gated scoring (not a crash) if calibration or the remote
judge endpoint is unavailable.

Also inspect the canonical candidate status/failure distribution every
cycle, not just Slurm `.err` files. Search workers catch ticket exceptions,
so a RUNNING pool with zero top-level tracebacks can still be failing every
candidate (this silently happened to 42 evaluations before 2026-08-02).
Follow `.cursor/rules/search-worker-handled-failures.mdc`.

## 2. Probe every arm's RAW numbers, not the composite score alone

For each DGX run, pull the last 1-2 real probe entries from `metrics.jsonl`
(skip entries where `"probe"` is `null`) and read `clean_handoff_rate`,
`no_listener_response_rate`, `listener_response_rate`,
`nonoverlap_listener_rate`, `overlap_rate`, `distinct_1`,
`repeated_4gram_fraction` directly -- not just `scripts/analysis/score_probe.py`'s
single number. The composite score is useful for ranking, but the raw rates
are what tell you whether something structurally changed.

For both searches, read `searches/month_2026_08_phase2/reports/latest.md`
and `searches/h100_2026_08_phase2/reports/latest.md` (top ~8-10 rows)
every cycle -- a new leader or a newly-ungated candidate is the single
highest-value thing to catch each hour. During the import phase, also track
`imported_complete` counts and confirm the corrected outputs contain
`dirty_handoff_rate`, `chain_2plus_rate_lenient`, and
`chain_3plus_rate_lenient`. H100 rungs default to
500/1500/5000 target steps with only the top ~1/3 promoted at each boundary
-- a candidate sitting at "rung_complete" is awaiting that promotion pass,
not stuck.

When a DGX arm's trajectory is the deciding factor in a keep/cancel call
(not just a routine "still progressing" check), generate and look at its
actual plot, not just the raw numbers -- trends are easy to misread from a
table of 2-3 points:

```bash
python3 scripts/analysis/plot_metrics.py --run_dir logs/<run_name>
```

then view `logs/<run_name>/plots/probe.png` (and `reward_breakdown.png` for
the p0/p_same/p_handoff/p_overlap breakdown if the raw rates alone are
ambiguous).

## 3. Apply standing lessons before reacting to any single reading

- **One non-zero small-suite reading (10 trials, `evaluate_checkpoint_suite.py`
  via `--probe_every`) is not a result.** Require a repeat within the SAME
  arm, or a full broad-sweep verification (`scripts/eval/demo_handoff_variation.sbatch`,
  which runs `demo_handoff_variation.py`, 72-576+ stochastic trials, 2-4h),
  before treating it as real. This project has repeatedly seen small-suite
  spikes (0.1, 0.3, 0.5 clean-handoff readings) regress to near-zero (e.g.
  30% -> 1.2%) under broad verification -- don't skip this step even when a
  reading looks exciting. See `SUCCESS_STORIES.md`'s "Broad-sweep baseline"
  and "Small-suite leads" sections for the running tally of how often this
  has happened.
- **Distinguish a transient dip from real collapse.** A single bad-looking
  probe (elevated `repeated_4gram_fraction`, crashed `distinct_1`) can fully
  recover next probe. Only treat it as real degeneration if it's severe
  (e.g. `repeated_4gram_fraction` > ~0.5) or persists across 2+ probes.
- **Always follow `.cursor/rules/read-decoded-text-not-just-metrics.mdc`**
  before calling any checkpoint "healthy" or "best" -- spot-check actual
  decoded trial text (echo/self-repetition/vocabulary leakage don't always
  show up in the aggregate rates).
- **A candidate's checkpoint at its peak rung can be silently overwritten**
  by its own continued training. `scripts/search/controller.py`'s
  `snapshot_checkpoint_before_promotion` protects every promotion since
  2026-07-29 -- if inspecting an older/pre-fix candidate, don't assume its
  best-scoring checkpoint's weights still exist on disk.
- **"More training" is not universally better.** Some candidates peak
  mid-training and decline (e.g. candidate17: 8.26 at step1500 -> 5.04 at
  step5000); others improve monotonically to their final step. Don't assume
  either pattern -- check each candidate's own score-by-rung history.

## 4. Keep the current 14-GPU allocation: 12 HPO + 2 targeted manual jobs

Count DGX arms (exclude the search's own `matrix_month_workers*`/
`matrix_month_controller`/`matrix_month_watchdog` infra jobs from this
budget). As of 2026-08-02, keep exactly 2 high-information manual jobs and
12 search workers (7-worker primary pool + 5-worker secondary pool), for the
same 14-GPU total used before reallocation. Do not backfill cancelled manual
arms back to 5/7: those GPUs now belong to HPO. Current manual priorities are
`DO` (DE2 paired ablation: stronger overlap maximum) and `DP` (DE2 paired
ablation: lower activity positive weight). `DN` collapsed and `DG2`
broad-verified repetition-gated, so neither is active anymore. Re-derive
this from the user's latest instruction if it changes again.

Cancel-and-replace candidates, in priority order:
1. Real, severe collapse (see 3 above) with no recovery.
2. An arm whose question is answered (its H100 reference target has since
   been pruned/superseded, or it's shown a stable, unambiguous result for
   many probes with nothing left to learn).
3. The single slowest/least-informative arm, if you need room for a
   materially better lead and are otherwise full.

When backfilling, prioritize (in this order): a new/ungated H100 leaderboard
entrant not yet replicated on DGX > a real (non-gated) H100 finalist/pruned
candidate not yet tested > a seed-robustness replicate of the current best
DGX result > continuing to fill out already-represented families. Note when
a leaderboard entrant's `optuna_trial_number` is set -- TPE-drawn candidates
are worth a little extra attention since they represent the optimizer's own
best guesses, not blind random search.

Since the search's own worker pool now runs on V100 too (`gpu_memory_gb<=40`
clamp in `sample_candidate`), its saved candidate configs already have
`batch_size=1` -- you can often copy `gradient_accumulation_steps` directly
rather than remapping it. When you do want a faster DGX iteration than the
search's own effective batch, override with `GRADIENT_ACCUMULATION_STEPS=8`
(prioritize fast wall-clock iteration over exact effective-batch fidelity --
this project is deadline-constrained). Use `PROBE_EVERY=100` (or 125) so
probe checkpoints arrive quickly.

## 5. Verify every new launch, every time

After submitting, wait ~30-60s then check:

```bash
grep -ci traceback logs/slurm/<name>_<jobid>.err   # must be 0
python3 -c "import json; d=json.load(open('hyperparameters/<run_name>.json')); print(d['activity_pos_weight'], ...)"  # spot-check the key levers landed
```

For any override containing a comma (only `DATASET_WEIGHTS` currently),
`export VAR=...` in the submitting shell FIRST, then `sbatch --export=ALL,...`
-- never inline a comma-containing value in `--export=...,VAR="a,b,c",...`
(Slurm splits on every comma regardless of quoting; see
`.cursor/rules/sbatch-env-overrides.mdc`). Confirm via the job's own stdout
(`grep -A1 "training sources/weights" logs/slurm/<name>_<jobid>.out`) --
the hyperparameters JSON can still round-trip a truncated value and look
plausible at a glance.

## 6. Fix bugs on sight, don't just note them

If a traceback, wrong hyperparameter, or unexpected behavior turns up,
diagnose and fix it in the same cycle -- cancel/resubmit if needed, and add
a `.cursor/rules/*.mdc` entry if it's a new, non-obvious failure mode (see
existing rules: `sbatch-env-overrides.mdc`, `stale-file-cache-null-bytes.mdc`,
`first-step-validation-is-slow.mdc`, `background-loop-monitoring.mdc`).
Never leave a known-broken job running "to see what happens."

## 7. Log everything

Follow `.cursor/rules/training-run-log.mdc`: every submit/cancel gets a row
in `TRAINING_RUN_LOG.md` in the same turn, one concise sentence per cell.
Also log genuine findings (a new leader, a confirmed/refuted small-suite
lead, a collapse) even when no job action resulted, so the next checkup
(by you or anyone else) has continuity.

## 8. Update SUCCESS_STORIES.md with VERIFIED findings only

`SUCCESS_STORIES.md` is the durable record of confirmed wins, methodological
warnings, and matrix-rendered evidence (clean handoffs, chains) -- read it
at the start of a session for context on what's already known, and add to
it whenever a finding clears the bar in Section 3 (repeated within an arm,
or broad-verified) -- not on every promising-looking small-suite reading.
When adding a win, embed the actual evidence (matrix cells / decoded
transcript excerpt), not just a metric number, and note exactly how it was
verified (repeat count, broad-sweep trial count, judge score if used). If a
past "story" turns out to have been a fluke once broad-verified (this has
happened more than once -- a strong 10-trial small-suite reading regressing
to near-zero under a full broad sweep), correct it in place rather than
leaving a stale claim standing.

Also keep the file's own "Still to check" section current: prune items that
reference jobs/candidates no longer relevant, and add newly-opened questions
(e.g. `chain_2plus_rate` -- whether 2+ sequential handoffs EVER happen on a
verified checkpoint -- has been an open, unresolved question since early in
the race; keep tracking it explicitly rather than losing it in raw numbers).

## 9. Synthesize across DGX + H100 toward the final config

Periodically (at least every few cycles) step back from individual arms and
ask: **across every candidate that has scored well AND held up under
verification, what do they have in common?** Concrete levers this project
has already converged evidence on, as of the last synthesis -- re-derive
this yourself each time rather than trusting it blindly, it will keep
shifting as more data comes in:
- `sampling_strategy=balanced_cycle` appears in essentially every top real
  H100 leader (candidate17/7/40/50/64) -- strong, consistent signal.
- `lambda_activity=0.5` + `lambda_reward=0.2` is the dominant winning pair.
- `context_lookback_columns` >= 64 all work; no single value dominates.
- `activity_pos_weight` in the ~0.78-0.89 band; below ~0.75 tends toward
  silence, above ~0.90 toward overlap/babble collapse.
- Small-suite (10-trial) DGX spikes have a poor track record of holding up
  under broad verification -- weight DGX small-suite readings much less
  than H100's own broad-swept rung scores when they conflict.

Keep a running mental note of which of these are still holding up vs. which
have been contradicted by newer data. When you have several independent,
broad-verified confirmations converging on a tight hyperparameter region
with no outstanding open questions, say so explicitly to the user and
propose transitioning from exploration to planning the final production
run (longer schedule, the converged-on config, real validation) rather than
continuing to spin up more exploratory arms indefinitely.

## 10. Re-arm the next check

```bash
sleep 3600 && echo MONTH_SEARCH_WAKE
```
with `notify_on_output` on `^MONTH_SEARCH_WAKE$` on that same call. Shorten
the interval (e.g. 1800s) after a fast-moving development or after catching
a monitoring gap, to reduce the worst-case blind window.

**The timer WILL occasionally die silently** (underlying session resets kill
even a correctly-armed loop with no error/notification -- see
`.cursor/rules/background-loop-monitoring.mdc`). There is no fully reliable
fix available; the mitigation is to always do a real, fresh check (not trust
elapsed time) the moment you're back, and to re-arm every single cycle
without being asked.
