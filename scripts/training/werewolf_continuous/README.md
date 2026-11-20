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

---

## 1. Quick start

Run these from the repo root, in the shell whose `python` you want the jobs to
use (jobs inherit `PATH` from where you run `start.sh`).

```bash
# Start (or resume) a run called run1
scripts/training/werewolf_continuous/start.sh run1

# See what it is doing
scripts/training/werewolf_continuous/status.sh run1

# Stop gracefully (current step finishes and saves, then everything winds down)
scripts/training/werewolf_continuous/stop.sh run1

# Stop immediately (cancels every job of the run; the last checkpoint is kept)
scripts/training/werewolf_continuous/stop.sh run1 --now
```

Running `start.sh run1` again later **resumes** the run: same game pool,
same published weights, newest learner checkpoint. A different `RUN_NAME`
starts a separate run in its own directory.

### What `start.sh` submits

| Job name | Default resources | What runs on it |
|---|---|---|
| `wwc-<RUN>-h100` | 4× H100 on `dgxh100` | GPUs 0–1: trainers, GPUs 2–3: game players |
| `wwc-<RUN>-v100` | 4× V100 on `dgx` | GPUs 0–1: trainers, GPUs 2–3: game players |
| `wwc-<RUN>-extra` | 1 borrowed GPU each, submitted by the leader | either a trainer or a player (alternates per GPU) |

### Smoke test on the teaching T4s

Small games and short jobs, handy after changing code. With 45-minute jobs you
also get to see a time-limit handoff and a successor job taking over.

```bash
RUN_NAME=smoke \
H100_SBATCH_OPTS="--partition=teaching --gres=gpu:t4:2 --cpus-per-task=8 --mem=64G --time=00:45:00" \
V100_SBATCH_OPTS="--partition=teaching --gres=gpu:t4:2 --cpus-per-task=8 --mem=64G --time=00:45:00" \
EXTRA_PARTITIONS="" \
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
while here trainers sit on different machines and GPU types and borrowed GPUs
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

### 2.3 The job chain (24 h limit)

Your QOS allows at most 24 hours per job, so each main job, as soon as it
starts (`chain.sh`):

1. runs the **chain guard**. It stops if `RUN/STOP` exists, or if the previous
   job in this chain was the 3rd crash in a row (`FAILED`, `OUT_OF_MEMORY`,
   `NODE_FAIL`, `BOOT_FAIL`). A normal end (`TIMEOUT`, `COMPLETED`, cancelled)
   resets the count.
2. queues its **successor** with `--dependency=afterany:<this job>`, so the next
   job is waiting, and gaining queue priority, the moment this one ends. A job
   never queues a second successor, and main jobs use `--no-requeue`.

Inside a job, each trainer or player process that fails is restarted up to
3 times in a row before its GPU is left idle for the rest of that job.

### 2.4 Borrowed GPUs (extras)

Every 2 minutes the leader looks at each partition in `EXTRA_PARTITIONS`
(`dgx,dgxh100`). It submits one 1-GPU `wwc-<RUN>-extra` job per idle GPU,
but only when **nobody else is waiting** on that partition and none of our
extras is already pending there. There is no cap (`MAX_EXTRA=0`).

* **Role.** When an extra starts it picks train or play. A GPU we have borrowed
  before does the opposite of last time. A GPU we have never borrowed takes
  whichever role currently has fewer live extras (ties go to playing).
  Recorded per physical GPU in `RUN/extras/`.
* **Giving it back.** Every 60 s an extra checks the queue. As soon as another
  user's job, or one of our own main-chain jobs, is waiting for resources on its
  partition, it abandons its current game or training share and exits. A
  trainer's share is reassigned, and a player's unfinished game is simply dropped.
* Extras only lead training if no main trainer is alive.

### 2.5 Stopping

* `stop.sh RUN` writes `RUN/STOP`. The leader finishes its current step and
  saves. Players finish their current game. Holding jobs notice within a minute.
  Queued successors see STOP and end without running.
* `stop.sh RUN --now` also `scancel`s every job named `wwc-RUN-*`.
* `start.sh RUN` removes the STOP file and resets the crash counters.

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

---

## 3. The run directory

`checkpoints/werewolf_rl/continuous/<RUN_NAME>/`

| Path | Contents |
|---|---|
| `game_config.json` | Game rules and sampling settings shared by every player and trainer (written by `start.sh` from `GAME_ARGS` on first start). |
| `STOP` | Present means stop. Created by `stop.sh`, removed by `start.sh`. |
| `weights/LATEST`, `weights/policy_v*.pt` | Newest published policy version and its trainable weights (last 3 kept). Version 0 is the SFT checkpoint itself, with no file. |
| `pool/incoming/v*.pt` | Finished games waiting to be trained on (full recordings). |
| `pool/claimed/step_*/` | Games the leader took for a step in progress. |
| `pool/archive/v*.json` | Transcripts of trained-on or stale games (recording tensors dropped; `--keep_rollouts` keeps them). |
| `learner/step_*/` | Checkpoints (last 3 kept): `model_state.pt` + `lora_adapters/` load with `werewolf_game.py`; `rl_state.pt` holds the optimizer, critic and normalizer for resuming. |
| `learner/metrics.jsonl` | One line per training step (see section 6). |
| `train/` | Live coordination: `leader.json` lease, `steps/<S>/` tasks and results, `epoch_weights/`. |
| `heartbeats/` | One file per live trainer and player. |
| `timing/*.jsonl` | Durations of every game, prep/gradient task, step and save. |
| `chain/<h100/v100>.json`, `chain/*_jobs.log` | Crash counter and job history of each chain. |
| `extras/` | Last role of each borrowed GPU. |
| `logs/` | One log per process, e.g. `h100_<job>_train0.log`, `v100_<job>_play2.log`, `extra_<job>_train.log`. |

Slurm's own job logs go to `logs/slurm/wwc-<RUN>-*_<jobid>.out/.err` (chain
guard and successor lines, crashes).

---

## 4. Code reference

### Shell scripts (this folder)

| File | Purpose |
|---|---|
| `start.sh [RUN]` | Refuses if the run already has queued jobs. Writes `game_config.json`, removes `STOP`, resets crash counters, exports the settings and submits the two main jobs. |
| `stop.sh [RUN] [--now]` | Graceful stop via `STOP`, optionally `scancel` too. |
| `status.sh [RUN]` | Jobs, policy version, pool size, average times, live trainers and players, current leader, chain state, last step's metrics. |
| `main.sbatch` | One node job: chain guard and successor, then `TRAIN_GPUS` (default half) trainer processes and the rest player processes, each with up to 3 restarts. |
| `chain.sh` | `chain_start`: guard, then queue exactly one successor. Also turns off NCCL P2P on `teaching`. |
| `notify.env.example` | Template for the private text-alert settings (section 8). |
| `extra.sbatch` | One borrowed GPU: `continuous extra-role` picks train or play, then runs `trainer.py` or `player.py` with `--role extra`. |
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
| `remaining_seconds`, `parse_slurm_duration`, `job_state` | Slurm time left (`squeue %L`) and final job state (`sacct`). |
| `waiting_jobs`, `idle_gpus`, `my_jobs` | Who is waiting for resources on a partition; free GPUs per node; our jobs by name prefix. |
| `maintain_extras` | Borrow idle GPUs politely (see 2.4). |
| `gpu_identity`, `choose_extra_role` | Identify a borrowed GPU; pick its role (alternating). |
| `chain_guard` | STOP and 3-crashes-in-a-row rule for a new chain job. |
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
| `H100_SBATCH_OPTS` | `--partition=dgxh100 --gres=gpu:h100:4 --cpus-per-task=16 --mem=192G` | Resources of the first main job (any `sbatch` options, e.g. `--exclude=node`). |
| `V100_SBATCH_OPTS` | `--partition=dgx --gres=gpu:v100:4 --cpus-per-task=16 --mem=192G` | Resources of the second main job. |
| `TRAIN_GPUS` | half of each job's GPUs | How many of each main job's GPUs train (the rest play). |
| `EXTRA_PARTITIONS` | `dgx,dgxh100` | Where to borrow idle GPUs (`""` turns borrowing off). |
| `MAX_EXTRA` | `0` | Cap on borrowed GPUs (0 = no cap). |
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

`status.sh run1` gives a summary. For details, each line of
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
| Jobs stay `PENDING (Priority)` | The partitions are busy; they start on their own when the scheduler allows. `squeue -u $USER --start` shows estimates when Slurm has one. |
| `status.sh` shows `STOPPED: 3 consecutive crashes` | Read the newest `logs/slurm/wwc-<RUN>-*.err` and `RUN/logs/*.log` for the error, fix it, then `start.sh <RUN>` again (it resets the counter). |
| A process log says `CUDA unavailable ... refusing to run on CPU` | The `python` on your `PATH` at `start.sh` time has a torch build the node's driver can't run. Start from a shell with a compatible environment. |
| Exit code 135 / `Bus error` while loading the model | That node's NFS reads are failing (seen on `dh-node9`). Add `--exclude=<node>` to that job's `*_SBATCH_OPTS` and restart. |
| No extras ever appear | Expected while other users have jobs waiting on `dgx`/`dgxh100`. Extras only borrow GPUs nobody else is waiting for. |
| Pool fills up, steps don't happen | No trainer is running (both main jobs queued or crashed). Players pause at `--max_pool` fresh games. Check `status.sh`. |
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
| `start` | default | A chain's first job starts running after `start.sh` (jobs can wait hours in the queue). |
| `crash` | high | A job ended by crashing (`FAILED`, `OUT_OF_MEMORY`, `NODE_FAIL`, `BOOT_FAIL`); the next job takes over. Sent when that next job starts. |
| `gave_up` | high | A trainer or player process crashed 3 times in a row inside a job; its GPU is idle until the job ends. |
| `stopped` | urgent | A chain stopped for good after 3 crashes in a row. Fix the cause, then `start.sh` again. |

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
