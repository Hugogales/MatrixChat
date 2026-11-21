# Always-on Werewolf RL (`werewolf_continuous`)

This loop keeps a set of GPUs on Rosie busy improving the Werewolf model until
you stop it. Some GPUs **play games** with the current policy, the others
**train** on those games (PPO, as described in `WEREWOLF_RL_PLAN.md`), and new
weights flow back to the players after every training step. It runs as Slurm
jobs that renew themselves, so nothing has to stay open on your side.

The game, rewards, critic and PPO math are the same code the single-job
trainer (`scripts/training/train_werewolf_rl.py`) and the eval script
(`scripts/eval/werewolf_game.py`) use. This folder only adds the machinery to
run them continuously across machines.

For a shorter overview of how the pieces fit together, read
[docs/WEREWOLF_CONTINUOUS_TRAINING.md](../../../docs/WEREWOLF_CONTINUOUS_TRAINING.md) first.

---

## 1. Quick start

Run these on a **login node** from the repo root, with the repo's `.venv`
active (jobs inherit `PATH`; on GPU nodes where that python doesn't work,
`chain.sh` switches to the node venv, see 2.8). `start.sh` also installs the
keeper's cron line on that login node.

```bash
# Start (or resume) a run called run1
scripts/training/werewolf_continuous/start.sh run1

# See what it is doing
scripts/training/werewolf_continuous/status.sh run1

# Pause for an update: nothing runs, but every job keeps its GPUs and its place
# in the queue; start.sh then restarts everything with the new code (2.5)
scripts/training/werewolf_continuous/stop.sh run1 --pause
scripts/training/werewolf_continuous/start.sh run1

# Stop gracefully (current step finishes and saves, then everything winds down;
# the queued jobs are cancelled, so the run gives up its place in line)
scripts/training/werewolf_continuous/stop.sh run1

# Stop immediately (cancels every job of the run; the last checkpoint is kept)
scripts/training/werewolf_continuous/stop.sh run1 --now
```

`start.sh` and `stop.sh` send an ntfy alert (`status`) saying what happened,
with a short summary of the queue (section 8).

Running `start.sh run1` again later **resumes** the run: same game pool,
same published weights, newest learner checkpoint. It is also safe while the
run is going: it only adds what is missing. A different `RUN_NAME` starts a
separate run in its own directory.

### What `start.sh` sets up

| Job name | Default resources | What runs on it |
|---|---|---|
| `wwc-<RUN>-h100` | 4× H100 on `dgxh100` (chain) | GPUs 0–1: trainers, GPUs 2–3: game players |
| `wwc-<RUN>-v100` | 8× V100 on `dgx` (chain) | GPUs 0–3: trainers, GPUs 4–7: game players |
| `wwc-<RUN>-v100b` | 4× V100 on `dgx` (chain) | GPUs 0–1: trainers, GPUs 2–3: game players |
| `wwc-<RUN>-gap1` | whatever is free (keeper) | half trainers, half players (a 1-GPU gap job plays) |

Each chain is one running job plus `CHAIN_DEPTH` (14) follow-up jobs already in
the queue (2.3), and a chain job never gives its GPUs back before its time
limit, even when something breaks (2.3, "Holding"). Rosie's interactive QOS runs at most **4 jobs per user** at
once, so 3 chains leave one slot, which the keeper uses for a gap job while
the chains are short of their GPUs (2.4).

### Smoke test on the teaching T4s

Small games and short jobs, handy after changing code. With 45-minute jobs you
also get to see a time-limit handoff and a successor job taking over.

```bash
CHAINS="a:teaching:2 b:teaching:2" CHAIN_DEPTH=1 GAP_PARTITIONS="" KEEPER=0 \
EXTRA_SBATCH="--time=00:45:00" \
TRAINER_ARGS="--games_per_step 4 --save_every 1 --keep_last 2" \
GAME_ARGS="--max_discussion_columns 30 --max_days 2 --reason_tokens 16 --reason_sentences 1" \
scripts/training/werewolf_continuous/start.sh smoke
```

---

## 2. How it works

```
             +---------------------------- shared run directory on /home ---------------------------+
             |                                                                                        |
 players --> |  pool/incoming/  (finished games)  --claim-->  leader  --tasks-->  trainers (all GPUs)   |
   ^         |                                                  |  <--gradients--                     |
   |         |                                                  v                                     |
   +-------- |  weights/policy_vN.pt  <--publish-- (sum gradients, optimizer step, checkpoint)         |
             +----------------------------------------------------------------------------------------+
```

Every process on every machine coordinates **only through files** in the run
directory. NCCL is not used: it needs a fixed set of GPUs on a fast network,
while here trainers sit on different machines and GPU types and gap jobs
come and go.

### 2.1 Players (`player.py`)

Each player GPU loops:

1. stop if `RUN/STOP` exists (or, for a borrowed GPU, if someone is waiting for it);
2. load the newest published policy if there is one (`weights/LATEST`);
3. pause if the pool already holds `--max_pool` (64) fresh games, since more would only go stale;
4. check that another game fits in the job's remaining time (see 2.6);
5. play one recorded game and write it to `pool/incoming/`.

A game file name records who played it:
`v000019_dh-node10_t4j302952s1_194239287261107.pt` means policy version 19,
node dh-node10, a T4 in job 302952 slot 1, game seed 194239287261107.

Recording means every sampled action (speak or yield, each word, each vote)
is saved with the log-probability it was sampled with, so the trainers can
recompute exactly the same distribution.

### 2.2 Trainers and the leader (`trainer.py`, `training/werewolf_rl/distributed.py`)

All training GPUs run the same program. Whichever one holds the **lease**
(`train/leader.json`) is the leader. It is also a trainer and works on its own
share. One step:

| Phase | Who | What happens |
|---|---|---|
| claim | leader | Takes `--games_per_step` (16) games from the pool, oldest first, that were played by the current or previous policy version (`--max_game_lag 1`). Older games are archived unused. |
| prep | every trainer | Each game goes to one trainer, faster GPUs getting more (split by each GPU type's measured per-game time). That trainer runs the old-policy pass, the KL-reference pass and the critic features. |
| advantages | leader | Critic values, then turn-level GAE for V_round and V_game, then per-role EMA normalization, then the 50/50 mix. Writes one advantage file per game. |
| PPO epochs (×2) | every trainer | Each trainer computes gradients for its games and writes them to a file. The leader sums them, updates the critic, clips, and takes the optimizer step. |
| publish | leader | Publishes the new weights as `weights/policy_v(N+1).pt`, archives the games, saves a checkpoint (`learner/step_*`), and appends a line to `learner/metrics.jsonl`. |

Details worth knowing:

* **Same game, same GPU.** A game stays on the trainer that did its prep for
  all PPO epochs, so its PPO ratios never mix bf16 (H100) and fp16 (V100) numbers.
* **Lost trainers.** Every process rewrites a heartbeat file every 20 s. A
  trainer whose heartbeat is older than 120 s, or which left on SIGTERM, is
  treated as gone, and its unfinished games are reassigned. The step still
  trains on all its games.
* **Leader changes.** If the leader's job is about to end, it hands the lease
  over. If the leader dies, its lease goes stale after 180 s. Either way another
  trainer takes over from the newest checkpoint. A checkpoint is saved every
  step, so at most the step in progress is lost, and its games go back into the pool.
* **What trains.** LoRA adapters plus the matrix interface modules (about
  16.8M parameters). The full-rank layers the SFT run unfroze stay at their SFT
  values. The KL reference is the frozen SFT checkpoint.

### 2.3 The job chains (24 h limit, first in line)

Rosie's scheduler gives every job the same priority, so GPUs go strictly
**first come, first served**: when a job ends, its GPUs go to the oldest
waiting job that fits. Jobs may run at most 24 hours. So each chain keeps
`CHAIN_DEPTH` (14) **follow-up jobs queued ahead**, each
`--dependency=afterany` on the one before. The follow-up that takes over at
each 24 h handover was submitted two weeks earlier, which puts it ahead of
anyone who started waiting since, and the chain keeps its GPUs. (Pending jobs
cost nothing and Rosie has no limit on them; only running jobs are capped.)

When a chain job starts (`chain.sh`, which every job reads at start, so
fixes reach jobs queued long ago):

1. it reads `RUN/launch.env` (chain sizes, depth, trainer/player flags),
   picks a python that runs CUDA on the node (2.8), and runs the **chain
   guard**. If `RUN/STOP` exists, the job exits and cancels the chain's queued
   jobs. The guard also counts crashed jobs (`FAILED`, `OUT_OF_MEMORY`,
   `NODE_FAIL`, `BOOT_FAIL`, or `COMPLETED` after less than 10 minutes, so a
   broken setup can't burn through the queue); a normal end (`TIMEOUT`,
   cancelled, a long `COMPLETED`) resets the count. At 3 in a row, a main chain's
   job **holds** (below); a gap chain stops and stays stopped until `start.sh`.
2. it **tops up** the chain's queue to `CHAIN_DEPTH` follow-ups (sized from
   `CHAIN_OPTS_<role>` in `launch.env`), appending after the last one.
3. it sends the `start` alert if this is the chain's first job (or the job
   before it never ran).

Never cancel a single queued follow-up: with `afterany`, the one behind it
would start at once, next to the running job. Cancel a whole chain by name
(`stop.sh` does).

**Holding.** A main-chain job keeps its GPUs until its time limit, whatever
goes wrong, so a failure never hands them to someone else:

* a trainer or player process that crashed 3 times in a row (each failure is
  restarted after 30 s), or one that exits early, leaves its GPU **held but
  idle**, with a `gave_up` alert;
* a node where no python runs torch on the GPU (2.8), or a chain whose last 3
  jobs crashed, leaves the whole job held but idle (`gave_up` / `stopped` alert).
  Its follow-ups stay queued; when the holding job reaches its time limit, the
  next one starts fresh and tries again;
* if `chain.sh` itself can't be read or is broken mid-edit, `main.sbatch`
  keeps the GPUs and tries it again every minute.

Fix the cause, then rerun `start.sh <RUN>`: it touches `RUN/chain/retry`, and
every idle GPU restarts in place within a minute, without being released.
`status.sh` lists idle GPUs as `IDLE (GPUs kept)`. Only `RUN/STOP` (`stop.sh`)
or the time limit ends a job early. `HOLD_ON_FAILURE=0` restores the old
behavior (failed processes leave the job, 3 crashed jobs stop the chain).

**`--time-min` (short pieces when the scheduler holds a node free).** Every chain and gap job is
submitted with `--time-min=00:30:00` (`TIME_MIN`). Normally a job gets its full 24 h. But
when an older job's booking makes Slurm keep a whole node free for a later start, a job
with `--time-min` may start at once with its limit cut down, so that it ends before that
booking. Slurm cuts it to exactly the minimum (30 min).

This is how the H100 chain runs while such a booking lasts:
* As of November 2026, freunds's 2-node job needs a GPU on dh-dgxh100-2. angelosm's
  ~1,500-task backlog, which is older than everyone's jobs, keeps that node full, so
  Slurm keeps all of dh-dgxh100-1 free for freunds's start, which never comes.
* Our 24 h jobs could not start there, so the H100 chain runs back to back in 30-minute
  pieces on dh-dgxh100-1. Each piece reloads the model (a minute) and plays games;
  trainer steps that don't fit in a piece are left to the V100 trainers.
* When the booking goes away, the pieces become 24 h jobs again on their own.
* A piece that takes over the same node is a renewal, not new GPUs, so it only writes
  a log line and sends no `gpu` alert.
* The queue depth counts jobs, not days: with 30-minute pieces, the 14 queued follow-ups
  are about 7 h old when they start, not 14 days.

**Extra GPUs beyond the targets** (`EXTRA_GPU_PARTITIONS`, default `dgxh100,dgx`: H100 and
V100). When every chain holds its target, the keeper puts the free job slot (Rosie allows 4
running jobs per user; the 3 chains use 3) to work as an **extra** job:
* it takes all free GPUs on one node of an extra partition, but not on a partition where a
  waiting job of another user could use the free GPUs;
* there is one slot, so when both partitions have free GPUs it takes the hole that does the
  most work: an H100 counts as 2.3 V100s (`GPU_SPEED` in keeper.py, measured on run1), so
  3 free H100s beat 5 free V100s and 8 free V100s beat 2 free H100s. A running extra job is
  not swapped for a better hole (that would restart its players);
* every minute, the extra job asks `keeper yield-check` whether another user's waiting job
  (one waiting for resources, not one held back by its own limits or dependencies) would
  fit once these GPUs are free and doesn't fit now. If so, it sends a `gpu` alert ("gave
  back ...") and cancels itself and its queued follow-up, so the GPUs are free at once. The
  keeper checks the same every 5 minutes;
* a gap job that covers a chain's missing GPUs (because that chain is between jobs) is
  part of the minimum, so it never yields;
* a job that can't start even with our extras released (such as freunds's 2-node job
  above) is no reason to give them back.

The chains' own GPUs (4 H100 + 12 V100 by default) are never given back.

**What can still take a handover.** At each handover Slurm gives the freed
GPUs to the oldest *eligible* waiting job that fits. Our follow-up is two weeks
old, so only jobs that have waited longer can beat it, plus jobs submitted
before this run started that are still waiting (for example another user's
follow-up queued before ours, or an array held back by that user's own job
limit). If one of those takes some GPUs at a handover, the chain's next job
waits for that job to finish, and the keeper meanwhile puts a gap job on what
is left (2.4). The only way to rule this out completely is an admin
reservation (`scontrol create reservation`, admins only) for the run's nodes.

### 2.4 The keeper and gap jobs

`training/werewolf_rl/keeper.py` runs from cron on the login node every
5 minutes (`RUN/keeper/keeper.log`). Each tick it:

1. re-creates a chain that has no jobs left and tops every chain's queue up to
   1 + `CHAIN_DEPTH` jobs (not for a stopped chain, nor after `STOP`);
2. while the chains hold fewer GPUs than their target (their job still waits),
   starts a **gap job** in the free job slot (`wwc-<RUN>-gap1`), sized to the
   largest free hole on `GAP_PARTITIONS` (at most the shortfall). It runs
   trainers and players like a chain job and renews itself (one follow-up);
3. ends the gap job when the chains hold their target (unless it is an **extra**
   job, below), when a chain job is blocked by the 4-jobs-per-user limit, or when
   the gap job holds GPUs a waiting chain job needs;
3b. when every chain holds its target, starts an **extra** job in that slot on
   `EXTRA_GPU_PARTITIONS` (H100 or V100: all free GPUs on one node, not on a partition
   where another user's waiting job could use them; the hole with the most work wins),
   and ends it as soon as another user's waiting job would fit
   in its GPUs (2.3, "Extra GPUs"). The extra job checks this itself every minute;
4. cancels a gap job still waiting after 10 minutes (that hole closed) and
   doesn't try that size on that partition again for an hour;
5. swaps a running gap job for one at least 2 GPUs bigger when such a hole opens.

`sbatch --test-only` is not used to judge holes: on this queue it predicts
days for jobs that start in seconds (it simulates the whole backlog, including
jobs the per-user limit holds back).

The old "extras" (1-GPU jobs the leader borrowed) are off: they would eat the
job slots the chains need at each handover. The keeper ends any it finds.

### 2.5 Stopping

* `stop.sh RUN --pause` (**for updates**) writes `RUN/PAUSE` and nothing else:
  players abandon their current game (an unfinished game is not saved),
  the leader finishes its current step and saves, helpers finish their task.
  Then each GPU waits, still held by its job. Queued follow-ups stay queued
  and the keeper keeps topping them up, so the run loses neither GPUs nor its
  place in line. `status.sh` shows `PAUSE=yes` and each waiting GPU as
  `PAUSED (GPUs kept)`. Change code or settings, then run `start.sh RUN`: it
  writes the settings, removes `PAUSE`, and within a minute every GPU starts
  again with the code on disk. A job that starts while paused waits the same way.
* `stop.sh RUN` writes `RUN/STOP`, cancels the run's **queued** jobs and
  removes the keeper's cron line (on that login node). The leader finishes its
  current step and saves. Players finish their current game. Holding jobs
  notice within a minute. The run gives up its place in line: a later
  `start.sh` queues new jobs behind everyone who queued in the meantime.
* `stop.sh RUN --now` also cancels the **running** jobs.
* `start.sh RUN` removes STOP (and PAUSE), resets the crash counters, queues
  missing chains and follow-ups and reinstalls the keeper.
* Each of these sends an ntfy `status` alert: "stopped for updates", "stopped",
  "running again", each with the queue (which jobs run where, which wait and why).

### 2.6 End-of-job timing

Before each new game (players) or step (leader), the process compares the
job's remaining time (from `squeue`) with the **average duration so far** of
that item on that GPU type × 1.05 (+ the average save time for the leader).
If it would not fit:

* the leader hands leadership to another trainer;
* the process then **holds its GPU idle until Slurm's time limit**, still
  finishing any work already assigned to it, then the successor job starts.

Averages are kept in `RUN/timing/*.jsonl` over the whole run.

### 2.7 Why files and not timestamps

Rosie's node clocks disagree by a minute or more. All liveness and lease checks
therefore compare **file modification times**, which the NFS server stamps with
its own clock, against a probe file the reader has just written, never one
machine's clock against another's.

### 2.8 Python on the GPU nodes

The repo's `.venv` works on the login and teaching nodes but not on the DGX nodes:

* `.venv/bin/python` is a link to `/usr/local/miniforge3/bin/python`, which the
  `dgx` and `dgxh100` nodes don't have (there `python` silently becomes another
  interpreter: Python 2.7, or one that picks up `~/.local`);
* its torch is `2.14.0+cu130`, which needs NVIDIA driver 580+. The `dgxh100`
  nodes have 595 (fine), but the `dgx` (V100) nodes have older drivers, and
  cu130 builds have no V100 (`sm_70`) kernels anyway:
  * dh-dgx1-3 has **535 (CUDA 12.2)**;
  * dh-dgx1-1 and dh-dgx1-2 have **525.105.17 (CUDA 12.0)**.

So each chain job tries, in order, the first python that runs a small fp16
attention on its GPU:

1. the python on `PATH` (`.venv`);
2. `JOB_VENV` = `~/.local/share/werewolf_rl/venv-nodes`: a venv whose
   interpreter is a copy of the same miniforge Python 3.12.11 on `/home`
   (`~/.local/share/werewolf_rl/miniforge3`) and whose `site-packages` is a link
   to `.venv`'s, so packages installed into `.venv` show up there too;
3. the same with `JOB_TORCH_OVERLAY` = `~/.local/share/werewolf_rl/torch-cu126`
   first on `PYTHONPATH`: links to `~/.local`'s `torch 2.14.1+cu126` and its CUDA 12
   libraries (V100 nodes).

**NVIDIA's CUDA compatibility library (driver 525 nodes).**
* The problem: on driver 525, torch's CUDA 12.6 runtime initializes, but it cannot
  open a GPU context ("CUDA-capable device(s) is/are busy or unavailable"). A plain
  CUDA context made without that runtime works.
* The fix: the overlay's `sitecustomize.py`, which Python imports at startup. When
  the node's kernel driver (`/proc/driver/nvidia/version`) is older than 535, it
  loads NVIDIA's forward-compatibility user-mode driver first, so torch uses that.
  It is the official `cuda-compat-12-2` package (535.183.06, the same user-mode
  version as dh-dgx1-3's working driver), unpacked in
  `~/.local/share/werewolf_rl/cuda-compat-12-2`. The 12.3 and 12.4 packages also work.
  The 12.6 package (560.x) is refused by driver 525 (error 803).
* Nodes with driver 535 or newer are left alone. If the Rosie admins update the
  drivers, the library stops being used without any change on our side.
* The job log shows `NVIDIA CUDA compat cuda-compat-12-2 on driver 525` when it is in use.
* `WEREWOLF_CUDA_COMPAT=off` disables it, and `WEREWOLF_CUDA_COMPAT=<dir>` picks
  another unpacked package.
* To rebuild it: download `cuda-compat-12-2_535.183.06-1_amd64.deb` from NVIDIA's
  `ubuntu2004/x86_64` CUDA repository (the V100 nodes run Ubuntu 20.04),
  `dpkg-deb -x` it into an empty directory, and move its
  `usr/local/cuda-12.2/compat` to `~/.local/share/werewolf_rl/cuda-compat-12-2`.

The choice is printed at the top of each job's `logs/slurm/*.out`. If none works,
the job sends a `gave_up` alert and keeps its GPUs idle (2.3, "Holding"); after
fixing the venv, `start.sh` restarts it in place. To rebuild the node venv:

```bash
D=~/.local/share/werewolf_rl
rsync -a --exclude pkgs /usr/local/miniforge/miniforge3/ $D/miniforge3/
mkdir -p $D/venv-nodes/bin $D/venv-nodes/lib/python3.12
ln -sfn $D/miniforge3/bin/python3.12 $D/venv-nodes/bin/python
ln -sfn ~/MatrixChat/.venv/lib/python3.12/site-packages $D/venv-nodes/lib/python3.12/site-packages
printf 'home = %s\ninclude-system-site-packages = false\nversion = 3.12.11\n' $D/miniforge3/bin > $D/venv-nodes/pyvenv.cfg
```

---

## 3. The run directory

`checkpoints/werewolf_rl/continuous/<RUN_NAME>/`

| Path | Contents |
|---|---|
| `game_config.json` | Game rules and sampling settings shared by every player and trainer (written by `start.sh` from `GAME_ARGS` on first start). |
| `STOP` | Present means stop. Created by `stop.sh`, removed by `start.sh`. |
| `PAUSE` | Present means paused for an update: nothing runs, every job keeps its GPUs (2.5). Created by `stop.sh --pause`, removed by `start.sh`. |
| `launch.json`, `launch.env` | The run's chains, depth, gap settings and job flags (written by `start.sh`). Every chain job sources `launch.env` when it starts. |
| `keeper/keeper.log`, `keeper/state.json` | The keeper's decisions; its last tick, waiting times and backoffs. |
| `weights/LATEST`, `weights/policy_v*.pt` | Newest published policy version and its trainable weights (last 3 kept). Version 0 is the SFT checkpoint itself, with no file. |
| `pool/incoming/v*.pt` | Finished games waiting to be trained on (full recordings). |
| `pool/claimed/step_*/` | Games the leader took for a step in progress. |
| `pool/archive/v*.json` | Transcripts of trained-on or stale games (recording tensors dropped; `--keep_rollouts` keeps them). |
| `learner/step_*/` | Checkpoints (last 3 kept): `model_state.pt` + `lora_adapters/` load with `werewolf_game.py`; `rl_state.pt` holds the optimizer, critic and normalizer for resuming. |
| `learner/metrics.jsonl` | One line per training step (see section 6). |
| `train/` | Live coordination: `leader.json` lease, `steps/<S>/` tasks and results, `epoch_weights/`. |
| `heartbeats/` | One file per live trainer and player. |
| `timing/*.jsonl` | Durations of every game, prep/gradient task, step and save. |
| `chain/<role>.json`, `chain/*_jobs.log` | Crash counter (and holding / stopped flag) and job history of each chain (`h100`, `v100`, `v100b`, `gap1`). |
| `chain/holding/` | One file per GPU (or whole job) a running chain job keeps idle, with the reason (2.3, "Holding"). |
| `chain/retry` | Touched by `start.sh`; idle GPUs restart when it is newer than their hold. |
| `extras/` | Last role of each borrowed GPU (old extras, off by default). |
| `logs/` | One log per process, e.g. `h100_<job>_train0.log`, `v100_<job>_play6.log`, `gap1_<job>_play0.log`. |

Slurm's own job logs go to `logs/slurm/wwc-<RUN>-*_<jobid>.out/.err` (python
choice, chain guard and follow-up lines, crashes).

---

## 4. Code reference

### Shell scripts (this folder)

| File | Purpose |
|---|---|
| `start.sh [RUN]` | Writes `game_config.json`, removes `STOP`, resets crash counters, touches `chain/retry` (idle GPUs restart), writes `launch.json`/`launch.env`, runs one keeper tick (queues missing chains, follow-ups, a gap job) and installs the keeper's cron line. Safe to rerun. |
| `stop.sh [RUN] [--pause \| --now]` | `--pause`: `PAUSE` only (GPUs and queue kept, start.sh resumes). Default: `STOP`, cancel queued jobs, remove the cron line; `--now` cancels running jobs too. Sends a `status` alert. |
| `status.sh [RUN] [--jobs\|--report\|--send]` | The status page (section 6): GPUs held per chain, what every GPU is doing, training, games, keeper, alerts. `--jobs` adds every queued job; `--report` prints the hourly phone report, `--send` sends it now. Built by `training/werewolf_rl/report.py`. |
| `main.sbatch` | One chain job; calls `chain_start` from `chain.sh` (Slurm freezes this file at submit time, `chain.sh` is read at start). If `chain.sh` can't be read, it keeps the GPUs and retries every minute. |
| `chain.sh` | `chain_start` (the whole job, never returns): `launch.env`, python choice (2.8), guard, queue top-up, start alert, then `chain_run_node`: `TRAIN_GPUS` (default half) trainers and the rest players, each with up to 3 restarts; then holding (`_chain_wait_retry`) instead of exiting. Also turns off NCCL P2P on `teaching`. |
| `notify.env.example` | Template for the private text-alert settings (section 8). |
| `extra.sbatch` | One borrowed GPU (old extras, off by default): `continuous extra-role` picks train or play, then runs `trainer.py` or `player.py` with `--role extra`. |
| `trainer.py` | Loads the SFT model, builds a `TrainerWorker` and runs it until STOP, yield or the time limit. Leaves immediately on SIGTERM. |
| `player.py` | Loads the SFT model and plays recorded games forever (see 2.1). Leaves immediately on SIGTERM. |

### `training/werewolf_rl/continuous.py`: run directory, pool, Slurm

| Function | What it does |
|---|---|
| `run_paths`, `ensure_run_dir`, `stop_requested` | Paths inside the run directory, create them, check `STOP`. |
| `write_game_config`, `apply_game_config` | Save / load the shared game and sampling settings. |
| `publish_weights`, `load_weights`, `latest_version`, `weights_path` | Write the trainable parameters as version N and point `LATEST` at it; copy a version into a model. |
| `write_game`, `pool_games`, `game_version`, `fresh_count` | Add a finished game to the pool (atomic rename); list it oldest first; read a game's version from its name. |
| `claim_games`, `release_claim`, `archive_game` | Take N fresh games for a step (archiving stale ones); afterwards archive them as JSON transcripts. |
| `record_time`, `average_time`, `fits` | Log durations; average so far; whether another item fits in the remaining time. |
| `remaining_seconds`, `parse_slurm_duration`, `job_info`, `job_state`, `job_elapsed` | Slurm time left (`squeue %L`); final state and run time (`sacct`). |
| `waiting_jobs`, `node_resources`, `idle_gpus`, `my_jobs` | Who is waiting for resources on a partition; free GPUs/CPUs/memory per node; our jobs by name prefix. |
| `maintain_extras` | Borrow idle GPUs politely (see 2.4). |
| `gpu_identity`, `choose_extra_role` | Identify a borrowed GPU; pick its role (alternating). |
| `chain_check` / `chain_guard` | What a new chain job does: run, hold (main chains after 3 crashed jobs in a row; short `COMPLETED` runs count) or stop (STOP, or a gap chain's sticky stop). |
| `idle_until_killed` | Hold until the time limit, or until a release reason (STOP, yield) appears. |
| CLI | `python -m training.werewolf_rl.continuous guard / write-config / extra-role ...` (used by the scripts). |

### `training/werewolf_rl/distributed.py`: the file-coordinated trainer

| Name | What it does |
|---|---|
| `TrainerWorker` | One training GPU. Sets up trainable parameters and the KL reference, runs the heartbeat and lease threads, executes `prep` and `grad` tasks (`do_task`), and loops in `run_forever` (lead, help, or hold). |
| `TrainerWorker.try_lead`, `release_lease`, `lease_held_by_other` | Acquire, release or check the leader lease (age from file mtimes). |
| `TrainerWorker.ensure_weights` | Load the weights a task asks for (published version or between-epoch weights). |
| `Leader` | The leader's side: `resume` (newest checkpoint, orphaned games back to the pool), `run` (claim, step, repeat; hand over when time is short), `train_step` (the phases in 2.2), `_plan` (split games by GPU speed), `_run_tasks` (wait, reassign from dead workers, work on its own tasks). |
| `Heartbeat`, `alive_heartbeats`, `server_now`, `file_age` | Liveness that is safe against clock skew (see 2.7). |

### `training/werewolf_rl/keeper.py`: chains and gap jobs (2.4)

| Name | What it does |
|---|---|
| `parse_chains`, `write_launch`, `read_launch`, `detect_job_cap` | `"role:partition:gpus ..."`; write/read `launch.json` + `launch.env`; the QOS's `MaxJobsPU`. |
| `Slurm` | `squeue`/`scontrol`/`sbatch`/`scancel` access (a fake in the tests). |
| `plan` | The decisions of one tick (no I/O): chain jobs to queue, gap jobs to end and why, gap slots to fill. |
| `shortfall`, `holes`, `fit` | GPUs the chains lack; free holes, best first (CPU/memory-limited, backoffs skipped). |
| `tick`, `locked_tick` | One pass: queue, cancel, wait for released GPUs, submit a gap job. |
| `install_cron`, `keeper_status` | The cron line (one per run, on the login node); the last tick for `status.sh`. |
| `queue_summary`, `announce` | The one-line queue in stop.sh / start.sh alerts (`IDLE` marks jobs holding unused GPUs). |
| CLI | `python -m training.werewolf_rl.keeper tick [--dry_run] / write-launch / install-cron / uninstall-cron / show / announce --run_dir RUN`. |

### `training/werewolf_rl/report.py`: status page and hourly report (section 6)

| Piece | Purpose |
|---|---|
| `snapshot` | Everything in one dict: Slurm jobs and what a waiting chain is behind, per-GPU rows from heartbeats and idle marks, training and game stats, keeper and alerts. |
| `gpu_rows`, `describe_player`, `describe_trainer` | One row per GPU of a running job (slot, train/play, state, detail). |
| `format_status`, `format_report` | The status.sh page; the phone version (at most ~3900 characters, no other users' names). |
| `maybe_send_report` | Called by every keeper tick; sends the report once per `report_minutes`, on the clock. |
| CLI | `python -m training.werewolf_rl.report status / hourly [--send] --run_dir RUN`. |

### Shared pieces used by both trainers

| Module | Contents |
|---|---|
| `training/werewolf_rl/core.py` | `add_rl_args`, `set_trainable`, `reference_state`, `SwapParams`, `optimizer_groups`, `pick_dtype`, `disable_dropout`, `save_checkpoint`, `load_rl_state`, `turn_advantages`, `mix_advantages`, `epoch_summary`. |
| `training/werewolf_rl/game.py` | The game (rules, recording, sampling filters, model loading). |
| `training/werewolf_rl/ppo.py` | `prepare_game`, `evaluate_game`, `game_losses`, `vote_advantages`. |
| `training/werewolf_rl/rewards.py` | Pillar 1 floor and text penalties, Pillar 2 round rewards, Pillar 3 win reward, GAE. |
| `training/werewolf_rl/critic.py` | Two-head critic and per-role EMA normalizer. |
| `training/werewolf_rl/notify.py` | Text alerts: `load_config`, `send_text` (Textbelt or Twilio), `notify` (event filter, repeat skip, log); CLI `test` / `send`. |

---

## 5. Configuration

### Environment variables for `start.sh`

| Variable | Default | Meaning |
|---|---|---|
| `RUN_NAME` (or 1st argument) | `run1` | Name of the run; its jobs are `wwc-<RUN_NAME>-*`. |
| `RUN_DIR` | `checkpoints/werewolf_rl/continuous/<RUN_NAME>` | Run directory. |
| `CHECKPOINT_DIR` | `checkpoints/checkpoint` | SFT checkpoint the policy starts from and the KL reference. |
| `HYPERPARAMETERS` | `hyperparameters/adaptcleanv3_cand_00003.json` | Matrix-config overrides for loading that checkpoint. |
| `MODEL_PATH` | `models/Qwen3-4B-Instruct-2507` | Base model. |
| `CHAINS` | `h100:dgxh100:4 v100:dgx:8 v100b:dgx:4` | The chains, `role:partition:GPUs` (one node each, so at most 8). Jobs already queued keep their size; `stop.sh --now` + `start.sh` resizes at once. |
| `CHAIN_DEPTH` | `14` | Follow-up jobs kept queued per chain (days ahead). |
| `HOLD_ON_FAILURE` | `1` | `1` = a main-chain job keeps its GPUs (idle) when something fails, until its time limit or a retry (2.3); `0` = it exits, and 3 crashed jobs stop the chain. |
| `GAP_PARTITIONS` | `dgx,dgxh100` | Where gap jobs may go (`""` = none). |
| `CPUS_PER_GPU`, `MEM_PER_GPU_GB` | `4`, `48` | Per-GPU CPUs and memory of every job. |
| `EXTRA_SBATCH` | empty | Extra `sbatch` options for every job (e.g. `--exclude=dh-dgx1-2`, `--time=00:45:00`). |
| `TRAIN_GPUS` | half of each job's GPUs | How many of each job's GPUs train (the rest play). |
| `KEEPER` | `1` | `0` = don't install the keeper's cron line. |
| `REPORT_MINUTES` | `60` | The keeper's ntfy report of every GPU, sent this often on the clock (`0` = off). |
| `TIME_MIN` | `00:30:00` | `--time-min` of every chain and gap job: start at once with a shorter limit when Slurm holds a node free for an older job (2.3). `""` = off. |
| `EXTRA_GPU_PARTITIONS` | `dgxh100,dgx` | Where the gap slot also grabs GPUs beyond the chain targets (H100 and V100), given back as soon as another user's job needs them (2.3). `""` = off. |
| `TRAINER_ARGS` | `--games_per_step 16 --save_every 1 --keep_last 3` | Extra `trainer.py` flags (below). |
| `PLAYER_ARGS` | empty | Extra `player.py` flags (below). |
| `GAME_ARGS` | empty | Game and sampling flags from `game.add_game_args` (e.g. `--seer`, `--num_players 7`). Applied when the run is first created or whenever given. |

Change game rules by starting a **new** run rather than restarting an old one
with different `GAME_ARGS`, so its pool never mixes games played under two rule sets.

### Useful `trainer.py` flags (`TRAINER_ARGS`)

| Flag | Default | Meaning |
|---|---|---|
| `--games_per_step` | 16 | Games per training step. |
| `--max_game_lag` | 1 | Train only on games played by the current or this many earlier versions. |
| `--ppo_epochs` | 2 | PPO passes over each step's games. |
| `--time_margin` | 0.05 | Safety margin on the average step time (2.6). |
| `--save_every`, `--keep_last` | 1, 3 | Checkpoint frequency and how many to keep. |
| `--lora_lr`, `--interface_lr`, `--critic_lr` | 1e-5, 1e-5, 1e-4 | Learning rates. |
| `--lambda_conv`, `--lambda_ppo`, `--beta_kl` | 1.0, 0.25, 0.02 | Loss balance from the plan. |
| `--kl_reference` | `sft` | `qwen` anchors the KL to plain Qwen instead (words and names only). |
| `--keep_rollouts` | off | Keep full game recordings in the archive (large). |

Reward and penalty weights (e.g. `--overlap_grace`, `--share_cap`, `--lambda_rep`)
are flags too; see `RewardConfig` in `training/werewolf_rl/rewards.py`.

### Useful `player.py` flags (`PLAYER_ARGS`)

| Flag | Default | Meaning |
|---|---|---|
| `--max_pool` | 64 | Pause while this many fresh games are already waiting. |
| `--max_lag` | 1 | What counts as fresh for that check. |
| `--time_margin` | 0.05 | Safety margin on the average game time. |

---

## 6. Reading the results

### The status page and the hourly report

`status.sh run1` prints one page (`training/werewolf_rl/report.py`):

* **GPUS**: GPUs held against the target per kind (e.g. `12/16 GPUs held (H100 0/4,
  V100 12/12): 7 working, 5 idle`). Then one line per chain:
  * its running job, node, time run and time left, and a count of GPU states;
  * or, while waiting, its pending reason and the older job the scheduler is keeping
    nodes free for, with that job's planned start.
* **EACH GPU**: one line per GPU of each running job.
  * A player shows the game it is in (`game 13: day 2 vote, 3 min in`), games done,
    average minutes, the last result, its policy version and GPU memory.
  * A trainer shows whether it leads (step and phase) or helps, and its tasks done.
  * A GPU can also show `waiting` (pool full), `loading` (new policy), `holding`
    (the job is about to end), `IDLE` / `paused` (with the reason), `starting`, or
    `down` (no heartbeat: see its log).
  * A job whose GPUs are all idle for the same reason gets one line.
* **TRAINING**: the policy version, the number of steps (and how many in the last hour),
  the leader's current step, and the last step's metrics.
* **GAMES**: the pool, games in the last hour, and the last 50 games under the current
  rules: Village win rate, days per game, minutes per game.
* **KEEPER / ALERTS**: the keeper's last tick and actions, the alert provider, and when
  the hourly report last went out.

The data comes from the heartbeats that every trainer and player rewrites every 20 s,
chain.sh's idle marks, `timing/*.jsonl` (each game's winner, days and length) and
`learner/metrics.jsonl`.

**Hourly report.** The keeper (cron, every 5 minutes) sends a shorter version of the
same page to ntfy once an hour, on the clock: one line per chain and one per working
GPU (`g4 play playing: game 13: day 2 vote, ...`). A job whose GPUs are all idle gets
one line. Other users' names are left out, because the push goes through ntfy.sh.
`REPORT_MINUTES` (start.sh) changes the interval and `0` turns it off.
`NTFY_PRIORITY_REPORT=low` in `notify.env` makes it arrive silently.
`status.sh run1 --report` shows what the phone gets, and `--send` sends it now.

### Metrics

For details, each line of
`learner/metrics.jsonl` is one step:

| Field | Meaning |
|---|---|
| `step`, `policy_version` | Step number and the version the step trained from (it publishes `policy_version + 1`). |
| `game_versions`, `game_players` | Policy versions and players (GPU, job, slot) of the step's games. |
| `trainers`, `reassigned_tasks` | Which trainers took part; how many tasks had to be moved off a trainer that disappeared. |
| `village_win_rate`, `mean_days`, `mean_columns` | Game outcomes and length. |
| `mean_speak_rate`, `mean_discussion_columns_per_game` | How much the players talk. Watch these for discussion collapsing. |
| `logp_mismatch_max/mean` | Gap between the log-probs recorded during play and recomputed in training. Small for current-version games; larger for one-version-old games, as expected. |
| `phase_seconds` | Where the step's time went (load, prep, advantages, each epoch's gradients and update, publish/save). |
| `epochs[]` | Per PPO epoch: `grad_norm`, `ppo_clip_frac`, `approx_kl_old`, `kl_ref` (KL to the SFT policy), `floor_penalty`, `self_rep`, `echo`, `critic_loss`. |

### Analyzing the games

`scripts/eval/analyze_werewolf_games.py` reads every game the run has played
(`pool/incoming`, `claimed`, `archive`) and explains how long games last and why:

```bash
python scripts/eval/analyze_werewolf_games.py                 # all games of run1
python scripts/eval/analyze_werewolf_games.py --last 100      # newest 100 (or --version 12)
python scripts/eval/analyze_werewolf_games.py --show 5        # full transcript of game #5 (or a seed)
python scripts/eval/analyze_werewolf_games.py --json logs/werewolf_games/run1_report.json
```

It prints:

* **why games end**: the longest game the rules allow (read from each game). With
  the default rule the Werewolves win only when every Villager is dead, so 6 players /
  1 Werewolf allow 3 votes: vote, kill, vote, kill, then a 1-vs-1 vote that is always
  a tie decided by lot. Games recorded before `--werewolves_win` used parity (2 votes).
  Then how each game ended, and the Village's win rate next to what random voting gives;
* **votes**: per day, the share of Villager votes on the Werewolf vs chance, mean
  P(vote Werewolf), ties, and whether the voted-out player was the most talkative,
  the most mentioned, or had said "I'm a werewolf";
* **discussion**: per day, columns, how many were cut off by `max_discussion_columns`
  vs ended by the long-silence limit, utterances and their length;
* **claims**: "I'm a werewolf" / "I'm a villager" by real role; a per-version table
  (to watch training change the play) and one line per game.

Play a game with any saved checkpoint:

```bash
python scripts/eval/werewolf_game.py \
    --checkpoint_dir checkpoints/werewolf_rl/continuous/run1/learner/step_000120 \
    --game_seed 7 --output logs/werewolf_games/run1_step120_seed7.json
```

---

## 7. Troubleshooting

| Symptom | Likely cause / what to do |
|---|---|
| Jobs stay `PENDING (Priority/Resources)` | Someone older in the queue gets the GPUs first (everyone has priority 1). The chain starts at the next handover it is first in line for; `squeue -u $USER --start` estimates ignore jobs that aren't eligible yet, so they are often too early. Meanwhile the keeper runs a gap job if any GPU is free. |
| `status.sh` says the keeper is not running | Its cron line lives on one login node (`crontab -l` there). Rerun `start.sh` on a login node, or `python -m training.werewolf_rl.keeper install-cron --run_dir RUN`. |
| A job log says `no python here runs torch on the GPU` | See 2.8; rebuild the node venv. |
| `BadConstraints` after `scontrol update` of a queued job | Changing `NumCPUs` leaves CPUs-per-task inconsistent. Cancel the job instead (if it heads a chain, the follow-up behind it takes over). |
| `status.sh` shows `IDLE (GPUs kept)` | Something failed and the job is holding those GPUs (2.3). The line names the log; fix the cause, then `start.sh <RUN>` restarts them in place within a minute. |
| `status.sh` shows `STOPPED: 3 consecutive crashes` (gap chains, or `HOLD_ON_FAILURE=0`) | Read the newest `logs/slurm/wwc-<RUN>-*.err` and `RUN/logs/*.log` for the error, fix it, then `start.sh <RUN>` again (it resets the counter). |
| A process log says `CUDA unavailable ... refusing to run on CPU` | The `python` on your `PATH` at `start.sh` time has a torch build the node's driver can't run. Start from a shell with a compatible environment. |
| Exit code 135 / `Bus error` while loading the model | That node's NFS reads are failing (seen on `dh-node9`). Set `EXTRA_SBATCH="--exclude=<node>"` and rerun `start.sh`. |
| Pool fills up, steps don't happen | No trainer is running (chains queued or crashed, only a 1-GPU gap job playing). Players pause at `--max_pool` fresh games. Check `status.sh`. |
| Step times grow | Look at `phase_seconds`: slow `prep`/`epoch*_grads` means slow or few trainers. Steps on the leader's own CPU (`advantages`, `epoch*_update`) are normally a few seconds. |
| Accidentally deleted something under `/home` | NetApp snapshots: `<dir>/.snapshot/hourly.*/` holds hourly copies; restore with `cp -p`. |
| Disk usage | Checkpoints are about 1 GB each (they embed the SFT base layers so they load on their own; the last `--keep_last` are kept), published weights about 70 MB each (last 3), and archive transcripts are small. `--keep_rollouts` makes the archive large. |

---

## 8. Phone alerts (ntfy)

The loop sends a push notification to your phone when something needs
attention, through [ntfy](https://ntfy.sh). It's free, unlimited and needs no
account. A job sends an HTTPS request to `https://ntfy.sh/<your topic>`, and
the ntfy app on your phone shows it. The compute nodes allow HTTPS, so jobs
send alerts directly; nothing else runs on Rosie. If ntfy can't be reached,
the alert goes to your number through textbelt.com instead (free key: one text
per day).

Your topic name works like a password: anyone who knows it can read your
alerts and post to it. Keep the long random one and don't share it.

### Setup

1. Install the **ntfy** app (iPhone App Store or Google Play).
2. In the app tap **+** and subscribe to the topic in `NTFY_TOPIC` from
   `~/.config/werewolf_rl/notify.env` (server: `ntfy.sh`).
3. Test from any terminal on Rosie:

   ```bash
   python -m training.werewolf_rl.notify test
   ```

`~/.config/werewolf_rl/notify.env` is private (outside the repo, readable only
by you). If it's missing, copy `scripts/training/werewolf_continuous/notify.env.example`
there, `chmod 600` it, and put a long random topic in `NTFY_TOPIC`. Jobs re-read
it for every alert, so edits apply without restarting the run.

### Alerts

| Alert (`NOTIFY_EVENTS`) | Priority | When |
|---|---|---|
| `start` | default | (Old; replaced by `gpu`, which every job sends when it starts.) |
| `crash` | high | A job ended by crashing (`FAILED`, `OUT_OF_MEMORY`, `NODE_FAIL`, `BOOT_FAIL`); the next job takes over. Sent when that next job starts. |
| `gave_up` | high | A trainer or player process crashed 3 times in a row (or a node has no working python): the job keeps that GPU idle until its time limit. Fix it, then `start.sh` restarts it in place. |
| `stopped` | urgent | A chain's last 3 jobs crashed: its next job keeps the GPUs idle until its time limit, then tries again (a gap chain stops for good). Fix the cause, then `start.sh` again. |
| `status` | default | `stop.sh --pause` ("stopped for updates"), `stop.sh` / `--now` ("stopped"), `start.sh` ("running again"), each with the queue. |
| `gpu` | default | Every time a job of the run starts and gets GPUs: how many and which (model, memory), node, partition, chain, job, whether it took over from the chain's previous job, when it ends, whether it works (and which python / CUDA compat), and how many GPUs the run now holds. Also when a held job's idle GPUs start working again. Sent even if `NOTIFY_EVENTS` leaves it out. |
| `report` | default | Every hour from the keeper: GPUs held, training and game totals, one line per chain and per working GPU (section 6). Override the priority with `NTFY_PRIORITY_REPORT` (e.g. `low`). |

`status` (default priority) comes from `stop.sh` and `start.sh` ("stopped for
updates", "stopped", "running again", plus the queue). It is sent even when
`NOTIFY_EVENTS` leaves it out, because someone just ran the script.

An identical alert within 30 minutes is skipped. Every attempt is logged in
`RUN/chain/notify.log`. A failed alert never affects training. `status.sh`
shows whether alerts are on.

### Other options

`NOTIFY_PROVIDER` can also be:

* `textbelt`: SMS through textbelt.com (free key: one per day).
* `twilio`: SMS through a Twilio account.
* `gateway`: email-to-text through your carrier. This does **not** work for
  Verizon any more: in November 2026 both `vtext.com` and `vzwpix.com` bounced
  with "address not found". AT&T's and T-Mobile's gateways are shut down too.
  Using it on Rosie also needs the login-node cron relay
  (`python -m training.werewolf_rl.notify install-cron`), because compute
  nodes block email.
