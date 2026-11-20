"""Elastic, file-coordinated PPO trainers for the always-on Werewolf loop.

Every training GPU runs one :class:`TrainerWorker`. The workers live on
different machines (an H100 node, a V100 node, borrowed GPUs), can join and
leave at any time, and share nothing but the run directory, so instead of
NCCL they coordinate through files:

    RUN/train/leader.json            lease: which worker leads, renewed every few seconds
    RUN/heartbeats/trainer_<id>.json each worker's liveness, GPU type, remaining job time
    RUN/train/steps/<S>/games.json   the games claimed for step S
    RUN/train/steps/<S>/tasks/*.json work the leader assigned (prep or grad, games, weights)
    RUN/train/steps/<S>/prep_<i>.pt  old/ref log-probs + critic features of game i
    RUN/train/steps/<S>/adv_<i>.pt   normalized advantages of game i
    RUN/train/steps/<S>/grad_<t>.pt  summed gradients of task t
    RUN/train/epoch_weights/         weights between the PPO epochs of a step

One step (the leader drives it; it also works on its own share):

1. claim ``games_per_step`` fresh games from the pool;
2. prep: each game goes to one worker, which runs the old and reference
   passes and the critic features (the game stays with that worker for the
   whole step, so its PPO ratios never mix GPU types/precisions);
3. the leader computes advantages (critic values, GAE, per-role EMA, mix);
4. ``ppo_epochs`` times: workers compute gradients for their games, the
   leader sums them, updates the critic, clips, and steps the optimizer;
5. the leader publishes the new weights for players and saves a checkpoint.

A worker whose heartbeat stops (crashed, or a borrowed GPU given back) has
its unfinished games reassigned to the others, so every step still trains on
all its games. Any worker can take over leadership from the last checkpoint.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import shutil
import socket
import threading
import time
import uuid

import torch
import torch.nn.functional as F

from . import continuous as cont
from .core import (
    SwapParams, disable_dropout, epoch_summary, load_rl_state, mix_advantages, optimizer_groups, reference_state,
    save_checkpoint, set_trainable, turn_advantages,
)
from .critic import MultiLayerResidualCritic, RoleConditionedEMA, default_tap_layers
from .ppo import LossConfig, SamplingConfig, evaluate_game, game_losses, prepare_game, vote_advantages
from .rewards import RewardConfig, exempt_token_ids


class LeaseLost(Exception):
    pass


class Yield(Exception):
    pass


def train_paths(run_dir):
    root = os.path.join(run_dir, "train")
    return {"root": root, "lease": os.path.join(root, "leader.json"), "steps": os.path.join(root, "steps"),
            "epoch_weights": os.path.join(root, "epoch_weights"), "heartbeats": os.path.join(run_dir, "heartbeats")}


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


_PROBE = {"t": 0.0, "now": None}


def server_now(run_dir):
    """The shared filesystem's current time, read off a freshly written probe file.

    Machines' clocks differ (a minute or more on Rosie), but NFS stamps file
    modification times with the file server's clock, so comparing file mtimes
    with this value is skew-proof. Cached for a second.
    """
    if _PROBE["now"] is not None and time.monotonic() - _PROBE["t"] < 1.0:
        return _PROBE["now"] + (time.monotonic() - _PROBE["t"])
    folder = train_paths(run_dir)["heartbeats"]
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f".probe_{socket.gethostname().split('.')[0]}_{os.getpid()}")
    with open(path, "w") as f:
        f.write(str(time.time()))
        f.flush()
        os.fsync(f.fileno())
    _PROBE.update(t=time.monotonic(), now=file_mtime(path))
    return _PROBE["now"]


def file_mtime(path):
    """mtime of ``path`` as the file server sees it now (open + fstat revalidates NFS attributes)."""
    try:
        with open(path) as f:
            return os.fstat(f.fileno()).st_mtime
    except FileNotFoundError:
        return None


def file_age(run_dir, path):
    mtime = file_mtime(path)
    return None if mtime is None else server_now(run_dir) - mtime


def alive_heartbeats(run_dir, prefix, dead_after):
    """Heartbeat records ``prefix*.json`` rewritten within ``dead_after`` seconds (server clock)."""
    out = []
    for path in glob.glob(os.path.join(train_paths(run_dir)["heartbeats"], f"{prefix}*.json")):
        age = file_age(run_dir, path)
        hb = _read_json(path)
        if hb and age is not None and age <= dead_after:
            out.append(hb)
    return out


class Heartbeat:
    """Writes ``RUN/heartbeats/<name>.json`` every ``interval`` seconds from a daemon thread."""

    def __init__(self, run_dir, name, info, interval=20.0, extra=None):
        self.path = os.path.join(train_paths(run_dir)["heartbeats"], f"{name}.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.info, self.interval, self.extra = dict(info), interval, extra
        self.running = True
        self.beat()
        threading.Thread(target=self._loop, daemon=True).start()

    def beat(self):
        record = dict(self.info, t=time.time())
        if self.extra is not None:
            record.update(self.extra())
        cont.atomic_write_json(self.path, record)

    def _loop(self):
        while self.running:
            time.sleep(self.interval)
            if self.running:
                try:
                    self.beat()
                except OSError:
                    pass

    def stop(self):
        self.running = False
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


class TrainerWorker:
    """One training GPU. Leads when it holds the lease, otherwise works on assigned tasks."""

    def __init__(self, run_dir, model, tokenizer, cfg, saved, args, device, dtype, *, role="main",
                 partition="", chain_names=(), log=print, heartbeat=20.0, dead_after=120.0, lease=180.0,
                 poll=2.0):
        self.run, self.model, self.tokenizer, self.args = run_dir, model, tokenizer, args
        self.device, self.dtype, self.role, self.partition = device, dtype, role, partition
        self.chain_names, self.log = tuple(chain_names), log
        self.saved, self.saved_cfg = saved, saved.get("matrix_config", {})
        self.dead_after, self.lease_seconds, self.poll = dead_after, lease, poll
        self.paths = train_paths(run_dir)
        for key in ("steps", "epoch_weights", "heartbeats"):
            os.makedirs(self.paths[key], exist_ok=True)
        cont.ensure_run_dir(run_dir)
        self.kind = _gpu_kind(device)
        self.wid = f"{self.kind}-{socket.gethostname().split('.')[0]}-{os.environ.get('SLURM_JOB_ID', 'local')}-" \
                   f"{uuid.uuid4().hex[:6]}"

        # Policy setup (identical on every worker, so parameter names line up).
        self.base_keys = set_trainable(model, saved, train_base_layers=False)
        disable_dropout(model)
        model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.ref_state = reference_state(model, args.kl_reference, self.base_keys)  # before loading any RL weights
        model.pretrained_unfrozen = {}
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.sft_state = {n: p.detach().cpu().clone() for n, p in self.params.items()}  # policy v0
        self.tap_layers = default_tap_layers(model.base_model.config.num_hidden_layers)
        self.max_agents = cfg.get("max_agents", 8)
        self.reward_cfg = RewardConfig(**{k: getattr(args, k) for k in vars(RewardConfig())})
        self.loss_cfg = LossConfig(args.lambda_conv, args.lambda_ppo, args.beta_kl, args.clip_eps,
                                   kl_on_speak=args.kl_reference == "sft")
        try:
            vocab_limit = len(tokenizer)
        except TypeError:
            vocab_limit = None
        self.scfg = SamplingConfig(args.temperature, args.repetition_penalty, vocab_limit,
                                   args.activity_temperature, args.vote_temperature)
        self.loaded = ("v", 0)  # which weights are in the model
        self.games, self.preps = {}, {}  # (step, i) -> GameBatch / prep result, kept for affinity
        self.done_tasks = set()
        self.is_leader = False
        self.lease_info = {}
        self.lock = threading.Lock()
        self.yield_reason = None
        self.hb = Heartbeat(run_dir, f"trainer_{self.wid}", {"wid": self.wid, "kind": self.kind, "role": role,
                                                             "host": socket.gethostname().split(".")[0],
                                                             "job": os.environ.get("SLURM_JOB_ID", "local")},
                            interval=heartbeat, extra=self._heartbeat_extra)
        threading.Thread(target=self._lease_loop, args=(max(1.0, heartbeat / 2),), daemon=True).start()
        if role == "extra" and partition:
            threading.Thread(target=self._watch_queue, daemon=True).start()

    # ------------------------------------------------------------------ liveness
    def _heartbeat_extra(self):
        remaining = cont.remaining_seconds()
        return {"remaining": remaining, "accepting": self._can_take_step(remaining), "leader": self.is_leader}

    def _can_take_step(self, remaining=None):
        if remaining is None and os.environ.get("SLURM_JOB_ID"):
            return False  # inside a job but its time is unknown (e.g. it is being killed): take nothing new
        avg = cont.average_time(self.run, "learner_step")
        return self.yield_reason is None and cont.fits(remaining, avg, self.args.time_margin)

    def leave(self):
        """Stop advertising: remove the heartbeat and release the lease (on SIGTERM / exit)."""
        self.hb.stop()
        self.release_lease()

    def _watch_queue(self):
        while self.yield_reason is None:
            waiting = cont.waiting_jobs(self.partition, self.chain_names)
            if waiting:
                self.yield_reason = f"{len(waiting)} job(s) waiting on {self.partition}"
                return
            time.sleep(60)

    def _check_yield(self):
        if self.yield_reason:
            raise Yield(self.yield_reason)

    def alive_workers(self):
        return {hb["wid"]: hb for hb in alive_heartbeats(self.run, "trainer_", self.dead_after)}

    # ------------------------------------------------------------------ lease
    def _write_lease(self):
        """Rewrite the lease (its mtime is the renewal time); caller holds ``self.lock``."""
        info = dict(self.lease_info, owner=self.wid, released=False,
                    expires=time.time() + self.lease_seconds)  # 'expires' only for older readers
        cont.atomic_write_json(self.paths["lease"], info)

    def lease_held_by_other(self):
        current = _read_json(self.paths["lease"])
        if not current or current.get("owner") == self.wid:
            return False
        if current.get("released") or current.get("expires", 1) == 0:
            return False
        age = file_age(self.run, self.paths["lease"])
        return age is not None and age <= self.lease_seconds

    def _lease_loop(self, interval):
        while True:
            time.sleep(interval)
            if self.is_leader:
                with self.lock:
                    current = _read_json(self.paths["lease"])
                    if current and current.get("owner") != self.wid:
                        self.is_leader = False  # someone else took over; the main thread notices
                        continue
                    self._write_lease()

    def try_lead(self):
        """Take the lease if it is free or expired. Extras only lead when no main trainer is alive."""
        if not self._can_take_step(cont.remaining_seconds()):
            return False
        if self.role == "extra" and any(hb.get("role") == "main" for hb in self.alive_workers().values()):
            return False
        if self.lease_held_by_other():
            return False
        with self.lock:
            self.lease_info = {"owner": self.wid, "step": None}
            self._write_lease()
        time.sleep(min(3.0, self.poll))
        current = _read_json(self.paths["lease"])
        self.is_leader = bool(current and current.get("owner") == self.wid)
        return self.is_leader

    def release_lease(self):
        with self.lock:
            current = _read_json(self.paths["lease"])
            if current and current.get("owner") == self.wid:
                cont.atomic_write_json(self.paths["lease"], dict(current, expires=0, released=True))
            self.is_leader = False

    def _set_lease(self, **info):
        """Update the lease now, so helpers see a new step/phase without waiting for a renewal."""
        with self.lock:
            self.lease_info.update(info)
            if self.is_leader:
                self._write_lease()

    def _check_lease(self):
        if not self.is_leader:
            raise LeaseLost()

    # ------------------------------------------------------------------ weights
    def ensure_weights(self, spec):
        key = ("v", spec["version"]) if spec["kind"] == "v" else ("e", spec["id"])
        if key == self.loaded:
            return
        path = cont.weights_path(self.run, spec["version"]) if spec["kind"] == "v" else spec["path"]
        if spec["kind"] == "v" and spec["version"] == 0:
            state = self.sft_state
        else:
            state = torch.load(path, map_location="cpu")
        with torch.no_grad():
            for n, t in state.items():
                self.params[n].data.copy_(t.to(self.params[n].device, self.params[n].dtype))
        self.loaded = key

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.params.items()}

    # ------------------------------------------------------------------ games
    def step_dir(self, step):
        return os.path.join(self.paths["steps"], f"{step:06d}")

    def game(self, step, i):
        key = (step, i)
        if key not in self.games:
            paths = _read_json(os.path.join(self.step_dir(step), "games.json"))
            result = torch.load(paths[i], map_location="cpu", weights_only=False)
            exempt = exempt_token_ids(self.tokenizer, result["names"])
            self.games[key] = prepare_game(result, exempt, self.reward_cfg, self.max_agents,
                                           self.args.num_players).to(self.device)
        return self.games[key]

    def autocast(self):
        if self.dtype == torch.float16:
            return torch.autocast("cuda", dtype=torch.float16)
        return torch.autocast(self.device.type, enabled=False)

    def _forget_old_steps(self, keep_step):
        for d in (self.games, self.preps):
            for key in [k for k in d if k[0] != keep_step]:
                del d[key]

    # ------------------------------------------------------------------ tasks
    def do_task(self, step, task):
        """Run one assigned task and write its result files."""
        self._forget_old_steps(step)
        t_start = time.time()
        self.ensure_weights(task["weights"])
        t0 = time.time()
        self.timing = {"weights": t0 - t_start, "games": 0.0, "save": 0.0}
        self.model.train()
        if task["kind"] == "prep":
            for i in task["games"]:
                self._check_yield()
                out_path = os.path.join(self.step_dir(step), f"prep_{i}.pt")
                if os.path.exists(out_path):
                    continue
                tg = time.time()
                g = self.game(step, i)
                self.timing["games"] += time.time() - tg
                with torch.no_grad(), self.autocast():
                    old = evaluate_game(self.model, g, self.scfg, tap_layers=self.tap_layers,
                                        chunk=self.args.token_chunk)
                    with SwapParams(self.model, self.ref_state):
                        ref = evaluate_game(self.model, g, self.scfg, chunk=self.args.token_chunk)
                mism = [(old[k] - ro).abs() for k, ro in (("tok_logp", g.tok_logp_rollout),
                                                          ("spk_logp", g.spk_logp_rollout),
                                                          ("ch_logp", g.ch_logp_rollout)) if ro.numel()]
                mism = torch.cat(mism) if mism else torch.zeros(1, device=self.device)
                prep = {"worker": self.wid,
                        "old": {k: old[k].float().cpu() for k in ("tok_logp", "spk_logp", "ch_logp", "cand_logp")},
                        "ref": {k: ref[k].float().cpu() for k in ("tok_logp", "spk_logp", "ch_logp")},
                        "hidden": old["hidden"][0].to(torch.float16).cpu(),
                        "mismatch": [float(mism.max()), float(mism.sum()), mism.numel()]}
                ts = time.time()
                torch.save(prep, out_path + ".tmp")
                os.replace(out_path + ".tmp", out_path)
                self.timing["save"] += time.time() - ts
                self.preps[(step, i)] = prep
            cont.record_time(self.run, f"task_prep_{self.kind}", (time.time() - t0) / max(len(task["games"]), 1))
        else:
            self._grad_task(step, task)
            cont.record_time(self.run, f"task_grad_{self.kind}", (time.time() - t0) / max(len(task["games"]), 1))
        total = time.time() - t_start
        other = self.timing["weights"] + self.timing["games"] + self.timing["save"]
        self.log(f"[trainer] step {step} {task['kind']} task ({len(task['games'])} game(s)) {total:.1f}s: "
                 f"weights {self.timing['weights']:.1f}s, games {self.timing['games']:.1f}s, "
                 f"compute {total - other:.1f}s, save {self.timing['save']:.1f}s")
        self.done_tasks.add(task["tid"])

    def _prep(self, step, i):
        if (step, i) not in self.preps:
            self.preps[(step, i)] = torch.load(os.path.join(self.step_dir(step), f"prep_{i}.pt"),
                                               map_location="cpu", weights_only=False)
        return self.preps[(step, i)]

    def _grad_task(self, step, task):
        scale = 1024.0 if self.dtype == torch.float16 else 1.0
        for _attempt in range(6):
            for p in self.params.values():
                p.grad = None
            stats_sum = {}
            for i in task["games"]:
                self._check_yield()
                tg = time.time()
                g = self.game(step, i)
                prep = self._prep(step, i)
                adv = torch.load(os.path.join(self.step_dir(step), f"adv_{i}.pt"), map_location="cpu")
                self.timing["games"] += time.time() - tg
                dev = lambda d: {k: v.to(self.device) for k, v in d.items()}  # noqa: E731
                with self.autocast():
                    cur = evaluate_game(self.model, g, self.scfg, chunk=self.args.token_chunk)
                    loss, stats = game_losses(g, cur, dev(prep["old"]), dev(prep["ref"]), dev(adv),
                                              self.reward_cfg, self.loss_cfg, task["norm"])
                (loss * scale).backward()
                for k, v in stats.items():
                    stats_sum[k] = stats_sum.get(k, 0.0) + v
                del cur, loss
            grads = {n: (p.grad.float() / scale).cpu() if p.grad is not None else torch.zeros(p.shape)
                     for n, p in self.params.items()}
            if all(torch.isfinite(t).all() for t in grads.values()):
                break
            scale /= 8.0  # fp16 overflow: retry with a smaller loss scale
            self.log(f"[trainer] non-finite gradients; retrying task {task['tid']} with loss scale {scale:g}")
        else:
            raise RuntimeError(f"task {task['tid']}: gradients stayed non-finite")
        for p in self.params.values():
            p.grad = None
        ts = time.time()
        out = os.path.join(self.step_dir(step), f"grad_{task['tid']}.pt")
        torch.save({"grads": grads, "stats": stats_sum, "games": task["games"], "worker": self.wid}, out + ".tmp")
        os.replace(out + ".tmp", out)
        self.timing["save"] += time.time() - ts

    def my_open_tasks(self, step):
        tasks = []
        for path in sorted(glob.glob(os.path.join(self.step_dir(step), "tasks", "*.json"))):
            task = _read_json(path)
            if task and task["worker"] == self.wid and task["tid"] not in self.done_tasks and not self._task_done(step, task):
                tasks.append(task)
        return tasks

    def _task_done(self, step, task):
        if task["kind"] == "prep":
            return all(os.path.exists(os.path.join(self.step_dir(step), f"prep_{i}.pt")) for i in task["games"])
        return os.path.exists(os.path.join(self.step_dir(step), f"grad_{task['tid']}.pt"))

    # ------------------------------------------------------------------ main loop
    def _safe_task(self, step, task):
        """Run a helper task; a vanished step (reassigned while we were slow) is not an error."""
        try:
            self.do_task(step, task)
            self.failures = 0
        except Yield:
            raise
        except FileNotFoundError:
            self.log(f"[trainer] step {step} moved on without task {task['tid']}; skipping it")
            self.done_tasks.add(task["tid"])
        except Exception as exc:  # noqa: BLE001 - one bad task must not end the worker
            import traceback

            self.failures = getattr(self, "failures", 0) + 1
            self.done_tasks.add(task["tid"])
            self.log(f"[trainer] task {task['tid']} failed ({self.failures}/3): {exc}\n{traceback.format_exc()}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if self.failures >= 3:
                raise

    def run_forever(self):
        """Lead or help until STOP, a yield, or the job's time limit. Returns an exit reason."""
        try:
            while True:
                if cont.stop_requested(self.run):
                    return "STOP file"
                if self.yield_reason:
                    return self.yield_reason
                if self.try_lead():
                    outcome = Leader(self).run()
                    if outcome in ("stop",):
                        return "STOP file"
                    continue
                if not self._can_take_step(cont.remaining_seconds()):
                    self.log("[trainer] a full step no longer fits in this job; holding the GPU")
                    while not cont.stop_requested(self.run) and not self.yield_reason:
                        lease = _read_json(self.paths["lease"]) or {}
                        step = lease.get("step")
                        if step:  # finish anything already assigned to us
                            for task in self.my_open_tasks(step):
                                self._safe_task(step, task)
                        time.sleep(self.poll)
                    return self.yield_reason or "STOP file"
                lease = _read_json(self.paths["lease"]) or {}
                step = lease.get("step")
                tasks = self.my_open_tasks(step) if step else []
                if tasks:
                    self._safe_task(step, tasks[0])
                else:
                    time.sleep(self.poll)
        except Yield as exc:
            return str(exc)
        finally:
            self.release_lease()
            self.hb.stop()


class Leader:
    """The leader's side of :class:`TrainerWorker`: drives steps while it holds the lease."""

    def __init__(self, worker):
        self.w = worker
        self.args, self.run_dir, self.log = worker.args, worker.run, worker.log
        model, args = worker.model, worker.args
        self.critic = MultiLayerResidualCritic(model.hidden_size, len(worker.tap_layers),
                                               max_agents=worker.max_agents).to(worker.device)
        self.ema = RoleConditionedEMA().to(worker.device)
        self.optimizer = torch.optim.AdamW(optimizer_groups(model, self.critic, args), weight_decay=args.weight_decay)
        self.step, self.version = 0, 0
        self.tids = itertools.count()
        self.last_extras = 0.0
        self.learner_dir = cont.run_paths(self.run_dir)["learner"]

    # ------------------------------------------------------------------ setup
    def resume(self):
        found = sorted(glob.glob(os.path.join(self.learner_dir, "step_*", "rl_state.pt")))
        if found:
            self.step, self.version = load_rl_state(found[-1], self.w.model, self.critic, self.ema, self.optimizer)
            self.log(f"[leader] {self.w.wid} resumed from {os.path.dirname(found[-1])} "
                     f"(step {self.step}, v{self.version})")
        else:
            self.w.ensure_weights({"kind": "v", "version": 0})
            self.log(f"[leader] {self.w.wid} starting from the SFT checkpoint")
        published = cont.latest_version(self.run_dir)
        if published > self.version:  # weights published after the last checkpoint were lost with their leader
            self.version = published + 1
            cont.publish_weights(self.run_dir, self.w.model, self.version)
            self.log(f"[leader] republished the checkpoint weights as v{self.version}")
        elif published < self.version:
            cont.publish_weights(self.run_dir, self.w.model, self.version)
        self.w.loaded = ("v", self.version)
        # Games an earlier leader claimed but never finished go back into the pool.
        incoming = cont.run_paths(self.run_dir)["incoming"]
        for orphan in glob.glob(os.path.join(cont.run_paths(self.run_dir)["claimed"], "step_*", "*.pt")):
            os.replace(orphan, os.path.join(incoming, os.path.basename(orphan)))
        shutil.rmtree(self.w.paths["steps"], ignore_errors=True)
        os.makedirs(self.w.paths["steps"], exist_ok=True)

    # ------------------------------------------------------------------ loop
    def run(self):
        try:
            self.resume()
            while True:
                self.w._check_lease()
                if cont.stop_requested(self.run_dir):
                    return "stop"
                if not self._step_fits():
                    self.log("[leader] another step would not fit in this job; handing leadership over")
                    return "handoff"
                self._tick_extras()
                paths = cont.claim_games(self.run_dir, self.step + 1, self.version, self.args.games_per_step,
                                         self.args.max_game_lag, self.args.keep_rollouts)
                if paths is None:
                    self._help_or_sleep()
                    continue
                self.train_step(paths)
        except LeaseLost:
            self.log("[leader] lost the lease; going back to helping")
            return "lost"
        finally:
            self.w.release_lease()

    def _step_fits(self):
        return cont.fits(cont.remaining_seconds(), cont.average_time(self.run_dir, "learner_step"), self.args.time_margin,
                         extra=cont.average_time(self.run_dir, "learner_save") or 60.0)

    def _tick_extras(self):
        parts = [x for x in self.args.extra_partitions.split(",") if x]
        if parts and time.time() - self.last_extras >= self.args.extras_every:
            self.last_extras = time.time()
            cont.maintain_extras(self.run_dir, parts, self.args.extra_name, self.args.extra_sbatch, self.w.chain_names,
                                 self.args.max_extra, log=self.log)

    def _help_or_sleep(self):
        time.sleep(self.w.poll)

    # ------------------------------------------------------------------ tasks
    def _workers(self):
        alive = self.w.alive_workers()
        ok = [wid for wid, hb in alive.items() if hb.get("accepting", True)]
        if self.w.wid not in ok:
            ok.append(self.w.wid)
        return sorted(ok, key=lambda x: (x != self.w.wid, x))

    def _game_cost(self, kind):
        """Measured seconds one game costs a ``kind`` GPU over a whole step (None if unknown)."""
        prep = cont.average_time(self.run_dir, f"task_prep_{kind}")
        grad = cont.average_time(self.run_dir, f"task_grad_{kind}")
        return None if prep is None or grad is None else prep + self.args.ppo_epochs * grad

    def _plan(self, workers, n):
        """Split ``n`` games so every worker finishes at about the same time (by GPU speed)."""
        kinds = {wid: hb.get("kind", "gpu") for wid, hb in self.w.alive_workers().items()}
        kinds[self.w.wid] = self.w.kind
        known = [c for c in (self._game_cost(kinds.get(wid, "gpu")) for wid in workers) if c]
        default = sum(known) / len(known) if known else 1.0
        cost = {wid: self._game_cost(kinds.get(wid, "gpu")) or default for wid in workers}
        plan = {wid: [] for wid in workers}
        for i in range(n):  # greedy: next game to whoever would finish it soonest
            wid = min(workers, key=lambda x: ((len(plan[x]) + 1) * cost[x], x != self.w.wid))
            plan[wid].append(i)
        return plan

    def _new_task(self, step, kind, games, worker, weights, epoch=0, norm=None):
        tid = f"{kind}{epoch}_{next(self.tids):04d}_{uuid.uuid4().hex[:4]}"
        task = {"tid": tid, "kind": kind, "games": list(games), "worker": worker, "weights": weights,
                "epoch": epoch, "norm": norm}
        tdir = os.path.join(self.w.step_dir(step), "tasks")
        os.makedirs(tdir, exist_ok=True)
        cont.atomic_write_json(os.path.join(tdir, f"{tid}.json"), task)
        return task

    def _run_tasks(self, step, tasks):
        """Wait for ``tasks``; reassign those of dead workers; work on our own meanwhile."""
        open_tasks = {t["tid"]: t for t in tasks}
        done, reassigned = [], 0
        while open_tasks:
            self.w._check_lease()
            alive = self.w.alive_workers()
            for tid, task in list(open_tasks.items()):
                if self.w._task_done(step, task):
                    done.append(task)
                    del open_tasks[tid]
                elif task["worker"] != self.w.wid and task["worker"] not in alive:
                    load = {}
                    for t in open_tasks.values():
                        load[t["worker"]] = load.get(t["worker"], 0) + len(t["games"])
                    candidates = [x for x in self._workers() if x != task["worker"]]
                    target = min(candidates, key=lambda x: (load.get(x, 0), x != self.w.wid))
                    games = [i for i in task["games"]
                             if task["kind"] == "grad"
                             or not os.path.exists(os.path.join(self.w.step_dir(step), f"prep_{i}.pt"))]
                    new = self._new_task(step, task["kind"], games, target, task["weights"], task["epoch"],
                                         task["norm"])
                    self.log(f"[leader] {task['worker']} is gone; its {len(games)} game(s) go to {target}")
                    del open_tasks[tid]
                    open_tasks[new["tid"]] = new
                    reassigned += 1
            mine = [t for t in open_tasks.values() if t["worker"] == self.w.wid]
            if mine:
                self.w.do_task(step, mine[0])
            elif open_tasks:
                time.sleep(min(self.w.poll, 1.0))
        return done, reassigned

    # ------------------------------------------------------------------ one step
    def train_step(self, paths):
        args, w = self.args, self.w
        t0 = time.time()
        step = self.step + 1
        sdir = w.step_dir(step)
        os.makedirs(sdir, exist_ok=True)
        cont.atomic_write_json(os.path.join(sdir, "games.json"), paths)
        w._set_lease(step=step, version=self.version, phase="prep")
        n = len(paths)

        # Leader-side view of every game (CPU), for advantages, counts and logging.
        results = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
        games = [prepare_game(r, exempt_token_ids(w.tokenizer, r["names"]), w.reward_cfg, w.max_agents,
                              args.num_players) for r in results]

        phases = {"load": time.time() - t0}
        mark = time.time()
        # 1. prep: old/ref passes + critic features, one worker per game; faster GPUs get more games.
        workers = self._workers()
        weights = {"kind": "v", "version": self.version}
        plan = self._plan(workers, n)
        tasks = [self._new_task(step, "prep", gs, wid, weights) for wid, gs in plan.items() if gs]
        _, reassigned = self._run_tasks(step, tasks)
        preps = [torch.load(os.path.join(sdir, f"prep_{i}.pt"), map_location="cpu", weights_only=False)
                 for i in range(n)]
        phases["prep"], mark = time.time() - mark, time.time()

        # 2. advantages (critic values -> GAE -> per-role EMA -> mix) and the global counts.
        self.critic.eval()
        hidden = [p["hidden"].to(w.device).float() for p in preps]
        feats = [g.turn_features.to(w.device) for g in games]
        raw, targets, roles = [], [], []
        with torch.no_grad():
            for g, h, f in zip(games, hidden, feats):
                v_round, v_game = self.critic(h, f)
                adv, tgt = turn_advantages(g, v_round.cpu().tolist(), v_game.cpu().tolist(), args)
                raw.append(adv)
                targets.append(tgt.to(w.device))
                roles.append(g.turn_role)
        normed = self.ema.normalize_and_update(torch.cat(raw).to(w.device), torch.cat(roles).to(w.device)).cpu()
        offset = 0
        for i, (g, prep) in enumerate(zip(games, preps)):
            u = raw[i].shape[0]
            adv = {"turn": mix_advantages(normed[offset:offset + u], args), "vote": vote_advantages(g, prep["old"])}
            offset += u
            torch.save(adv, os.path.join(sdir, f"adv_{i}.pt"))
        norm = {"actions": float(sum(g.tok_id.numel() + g.spk_act.numel() + g.ch_row.numel() for g in games)),
                "speaks": float(sum(g.spk_act.numel() for g in games)),
                "tokens": float(sum(g.tok_id.numel() for g in games)),
                "turns": float(sum(g.turn_agent.numel() for g in games))}

        phases["advantages"], mark = time.time() - mark, time.time()
        # 3. PPO epochs: workers send gradients, the leader sums and steps.
        owner = {i: preps[i]["worker"] for i in range(n)}
        epoch_stats = []
        for epoch in range(args.ppo_epochs):
            w._set_lease(phase=f"epoch{epoch}")
            alive = set(self._workers())
            by_worker = {}
            for i in range(n):
                by_worker.setdefault(owner[i] if owner[i] in alive else w.wid, []).append(i)
            tasks = [self._new_task(step, "grad", gs, wid, weights, epoch, norm) for wid, gs in by_worker.items()]
            done, extra_reassigned = self._run_tasks(step, tasks)
            reassigned += extra_reassigned
            phases[f"epoch{epoch}_grads"], mark = time.time() - mark, time.time()
            self.optimizer.zero_grad(set_to_none=True)
            totals = {}
            for task in done:
                res = torch.load(os.path.join(sdir, f"grad_{task['tid']}.pt"), map_location="cpu")
                for name, grad in res["grads"].items():
                    p = w.params[name]
                    g_dev = grad.to(p.device, p.dtype)
                    p.grad = g_dev if p.grad is None else p.grad + g_dev
                for k, v in res["stats"].items():
                    totals[k] = totals.get(k, 0.0) + v
                for i in task["games"]:
                    owner[i] = task["worker"]
            self.critic.train()
            critic_loss = sum(
                F.huber_loss(vr, tgt[:, 0], reduction="sum", delta=args.huber_delta)
                + F.huber_loss(vg, tgt[:, 1], reduction="sum", delta=args.huber_delta)
                for (vr, vg), tgt in ((self.critic(h, f), tgt) for h, f, tgt in zip(hidden, feats, targets))
            ) / norm["turns"]
            critic_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(w.params.values()), args.max_grad_norm))
            torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), args.max_grad_norm)
            self.optimizer.step()
            epoch_stats.append({"epoch": epoch, "grad_norm": grad_norm,
                                **epoch_summary(totals, norm, w.loss_cfg.kl_on_speak),
                                "critic_loss": float(critic_loss.detach())})
            if epoch + 1 < args.ppo_epochs:
                path = os.path.join(w.paths["epoch_weights"], f"{step:06d}_e{epoch + 1}.pt")
                torch.save(w.trainable_state(), path + ".tmp")
                os.replace(path + ".tmp", path)
                weights = {"kind": "e", "id": f"{step}_{epoch + 1}", "path": path}
                w.loaded = ("e", weights["id"])
            phases[f"epoch{epoch}_update"], mark = time.time() - mark, time.time()

        # 4. publish, archive the games, checkpoint, log.
        self.version += 1
        self.step = step
        cont.publish_weights(self.run_dir, w.model, self.version)
        w.loaded = ("v", self.version)
        cont.release_claim(self.run_dir, paths, args.keep_rollouts)
        if step % args.save_every == 0:
            seconds = save_checkpoint(os.path.join(self.learner_dir, f"step_{step:06d}"), w.model, self.critic,
                                      self.ema, self.optimizer, None, w.saved_cfg, w.base_keys, step, args,
                                      version=self.version, keep_last=args.keep_last)
            cont.record_time(self.run_dir, "learner_save", seconds)
        phases["publish_save"] = time.time() - mark
        cont.record_time(self.run_dir, "learner_step", time.time() - t0, step=step)
        mism = [p["mismatch"] for p in preps]
        record = {
            "step": step, "policy_version": self.version - 1, "games": n,
            "game_versions": sorted({r.get("pool_meta", {}).get("version", -1) for r in results}),
            "game_players": sorted({r.get("pool_meta", {}).get("gpu", "?") for r in results}),
            "trainers": sorted({t for t in owner.values()}), "reassigned_tasks": reassigned,
            "village_win_rate": sum(g.info["winner"] == "Village" for g in games) / n,
            "mean_days": sum(g.info["days"] for g in games) / n,
            "mean_columns": sum(g.info["columns"] for g in games) / n,
            "mean_discussion_columns_per_game": sum(
                sum(p.get("discussion_columns", 0) for p in r["phases"] if p["phase"] == "day") for r in results) / n,
            "mean_speak_rate": sum(float(g.spk_act.mean()) if g.spk_act.numel() else 0.0 for g in games) / n,
            "mean_round_reward_per_turn": sum(float(g.turn_r_round.sum()) for g in games) / max(norm["turns"], 1),
            "actions": norm["actions"], "speak_decisions": norm["speaks"], "tokens": norm["tokens"],
            "turns": norm["turns"],
            "logp_mismatch_max": max(m[0] for m in mism),
            "logp_mismatch_mean": sum(m[1] for m in mism) / max(sum(m[2] for m in mism), 1),
            "step_seconds": time.time() - t0, "phase_seconds": {k: round(v, 1) for k, v in phases.items()},
            "epochs": epoch_stats,
        }
        with open(os.path.join(self.learner_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        self.log(json.dumps({k: v for k, v in record.items() if k != "epochs"}))
        shutil.rmtree(sdir, ignore_errors=True)
        for old in glob.glob(os.path.join(w.paths["epoch_weights"], f"{step:06d}_e*.pt")):
            os.remove(old)
        w._forget_old_steps(step + 1)


def _gpu_kind(device):
    if device.type != "cuda":
        return "cpu"
    name = torch.cuda.get_device_name(device).lower()
    return next((k for k in ("h100", "a100", "v100", "t4") if k in name), "gpu")
