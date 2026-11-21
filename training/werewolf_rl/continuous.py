"""Always-on Werewolf RL: game pool, weight publishing, timing, Slurm chaining.

Everything is coordinated through files in one run directory on the shared
filesystem, so no process has to stay attached to a terminal:

    RUN/
      game_config.json        game rules + sampling, read by players AND the learner
      STOP                    touch to stop: no new jobs/games/steps are started
      PAUSE                   stop.sh --pause: players abandon their game, trainers finish
                              the current step, then every job keeps its GPUs idle (and its
                              queued follow-ups) until start.sh removes it
      weights/LATEST          newest published policy version (int)
      weights/policy_vNNNNNN.pt   trainable parameters of that version
      pool/incoming/*.pt      finished games waiting to be trained on
      pool/claimed/step_N/    games the learner took for step N
      pool/archive/*.json     transcripts of used/stale games (rollout tensors dropped)
      timing/*.jsonl          durations of every game / learner step / save
      chain/<role>.json       consecutive-crash counter per job chain
      learner/                learner checkpoints (step_*), metrics.jsonl
      logs/                   player logs

Game file names start with the policy version that played them
(``v000012_<host>_<gpu>_<seed>.pt``), so freshness is read from the name.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import socket
import subprocess
import time

CRASH_STATES = ("FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL")
WAITING_REASONS = ("Resources", "Priority")

# ----------------------------------------------------------------------------
# Run directory
# ----------------------------------------------------------------------------
def run_paths(run_dir):
    return {
        "config": os.path.join(run_dir, "game_config.json"),
        "stop": os.path.join(run_dir, "STOP"),
        "pause": os.path.join(run_dir, "PAUSE"),
        "weights": os.path.join(run_dir, "weights"),
        "incoming": os.path.join(run_dir, "pool", "incoming"),
        "claimed": os.path.join(run_dir, "pool", "claimed"),
        "archive": os.path.join(run_dir, "pool", "archive"),
        "timing": os.path.join(run_dir, "timing"),
        "chain": os.path.join(run_dir, "chain"),
        "learner": os.path.join(run_dir, "learner"),
        "logs": os.path.join(run_dir, "logs"),
    }

def ensure_run_dir(run_dir):
    paths = run_paths(run_dir)
    for key in ("weights", "incoming", "claimed", "archive", "timing", "chain", "learner", "logs"):
        os.makedirs(paths[key], exist_ok=True)
    return paths

def stop_requested(run_dir):
    return os.path.exists(run_paths(run_dir)["stop"])

def pause_requested(run_dir):
    return os.path.exists(run_paths(run_dir)["pause"])

def halt_reason(run_dir):
    """Why trainers and players should leave now: ``"STOP file"``, ``"PAUSE file"`` or ``None``.

    STOP ends the run (jobs wind down, queued jobs are cancelled); PAUSE only
    stops the processes: chain.sh keeps the GPUs and restarts them when start.sh
    removes it.
    """
    if stop_requested(run_dir):
        return "STOP file"
    if pause_requested(run_dir):
        return "PAUSE file"
    return None

def atomic_write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def write_game_config(run_dir, args, keys):
    """Store the game/sampling settings every player and the learner must share."""
    atomic_write_json(run_paths(run_dir)["config"], {k: getattr(args, k) for k in keys})

def apply_game_config(run_dir, args):
    """Overwrite ``args`` fields with the run's shared game config (if present)."""
    path = run_paths(run_dir)["config"]
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        for k, v in json.load(f).items():
            setattr(args, k, v)
    return True

# ----------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------
def latest_version(run_dir):
    path = os.path.join(run_paths(run_dir)["weights"], "LATEST")
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0

def weights_path(run_dir, version):
    return os.path.join(run_paths(run_dir)["weights"], f"policy_v{version:06d}.pt")

def publish_weights(run_dir, model, version, keep=3):
    """Write the trainable parameters as policy ``version`` and point LATEST at it."""
    import torch

    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    path = weights_path(run_dir, version)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    latest = os.path.join(run_paths(run_dir)["weights"], "LATEST")
    with open(latest + ".tmp", "w") as f:
        f.write(str(version))

    os.replace(latest + ".tmp", latest)
    old = sorted(glob.glob(os.path.join(run_paths(run_dir)["weights"], "policy_v*.pt")))
    for p in old[:-keep]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

def load_weights(run_dir, model, version):
    """Copy published policy ``version`` into ``model`` (dtype/device converted)."""
    import torch

    state = torch.load(weights_path(run_dir, version), map_location="cpu")
    named = dict(model.named_parameters())
    missing = [n for n in state if n not in named]
    if missing:
        raise KeyError(f"published weights have {len(missing)} unknown parameter(s), e.g. {missing[:3]}")
    with torch.no_grad():
        for n, t in state.items():
            named[n].data.copy_(t.to(device=named[n].device, dtype=named[n].dtype))

# ----------------------------------------------------------------------------
# Game pool
# ----------------------------------------------------------------------------
_GAME_RE = re.compile(r"^v(\d+)_")

def game_version(path):
    m = _GAME_RE.match(os.path.basename(path))
    return int(m.group(1)) if m else -1

def write_game(run_dir, result, version, gpu_label, seed, seconds):
    """Save a finished game into the pool (atomic rename, so readers never see half a file)."""
    import torch

    host = socket.gethostname().split(".")[0]
    name = f"v{version:06d}_{host}_{gpu_label}_{seed}.pt"
    path = os.path.join(run_paths(run_dir)["incoming"], name)
    result = dict(result, pool_meta={"version": version, "gpu": gpu_label, "host": host, "seed": seed,
                                     "seconds": seconds, "finished": time.time()})
    torch.save(result, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path

def pool_games(run_dir):
    """Finished games waiting in the pool, oldest version first, then oldest file."""
    files = glob.glob(os.path.join(run_paths(run_dir)["incoming"], "v*.pt"))
    return sorted(files, key=lambda p: (game_version(p), os.path.getmtime(p) if os.path.exists(p) else 0))

def fresh_count(run_dir, min_version):
    return sum(game_version(p) >= min_version for p in pool_games(run_dir))

def archive_game(run_dir, path, keep_rollout=False, reason="used"):
    """Move a game out of the pool; by default keep only its JSON transcript."""
    import torch

    archive = run_paths(run_dir)["archive"]
    base = os.path.splitext(os.path.basename(path))[0]
    if keep_rollout:
        shutil.move(path, os.path.join(archive, os.path.basename(path)))
        return
    try:
        result = torch.load(path, map_location="cpu", weights_only=False)
        result = {k: v for k, v in result.items() if k != "rollout"}
        result["pool_meta"] = dict(result.get("pool_meta", {}), archived_as=reason)
        with open(os.path.join(archive, base + ".json"), "w") as f:
            json.dump(result, f)
    except Exception as exc:  # a corrupt file must not stop training
        print(f"[warn] could not archive {path}: {exc}", flush=True)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def claim_games(run_dir, step, version, count, max_lag=1, keep_rollout=False):
    """Take ``count`` fresh games (version >= version - max_lag) for learner step ``step``.

    Older games are archived as stale. Returns the claimed paths (moved into
    ``pool/claimed/step_N``) or ``None`` if fewer than ``count`` are fresh.
    """
    min_version = version - max_lag
    fresh = []
    for p in pool_games(run_dir):
        if game_version(p) < min_version:
            archive_game(run_dir, p, keep_rollout, reason="stale")
        else:
            fresh.append(p)
    if len(fresh) < count:
        return None
    dest = os.path.join(run_paths(run_dir)["claimed"], f"step_{step:06d}")
    os.makedirs(dest, exist_ok=True)
    claimed = []
    for p in fresh[:count]:
        target = os.path.join(dest, os.path.basename(p))
        os.replace(p, target)
        claimed.append(target)
    return claimed

def release_claim(run_dir, paths, keep_rollout=False):
    for p in paths:
        archive_game(run_dir, p, keep_rollout, reason="used")
    for d in {os.path.dirname(p) for p in paths}:
        try:
            os.rmdir(d)
        except OSError:
            pass

# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
def record_time(run_dir, kind, seconds, **extra):
    path = os.path.join(run_paths(run_dir)["timing"], f"{kind}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps({"t": time.time(), "seconds": seconds, **extra}) + "\n")

def average_time(run_dir, kind):
    """Mean duration of every recorded ``kind`` so far (None if nothing recorded yet)."""
    path = os.path.join(run_paths(run_dir)["timing"], f"{kind}.jsonl")
    total = n = 0
    try:
        with open(path) as f:
            for line in f:
                try:
                    total += float(json.loads(line)["seconds"])
                    n += 1
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        return None
    return total / n if n else None

def fits(remaining, average, margin=0.05, extra=0.0):
    """Whether another item of ``average`` seconds (+``margin``) fits in ``remaining`` seconds."""
    if remaining is None or average is None:
        return True
    return remaining > average * (1.0 + margin) + extra

# ----------------------------------------------------------------------------
# Slurm
# ----------------------------------------------------------------------------
def _run(cmd, timeout=60):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""

def parse_slurm_duration(text):
    """``[D-]HH:MM:SS`` / ``MM:SS`` / ``SS`` -> seconds (None for UNLIMITED/INVALID/empty)."""
    text = (text or "").strip()
    if not text or not text[0].isdigit():
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(x) for x in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s

def remaining_seconds(job_id=None):
    """Seconds left before this Slurm job's time limit (None outside Slurm)."""
    job_id = job_id or os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    return parse_slurm_duration(_run(["squeue", "-h", "-j", str(job_id), "-o", "%L"]).strip())

def job_info(job_id):
    """``(state, elapsed_seconds)`` of a job from sacct (``("", None)`` if unknown)."""
    lines = _run(["sacct", "-n", "-X", "-P", "-j", str(job_id), "-o", "State,ElapsedRaw"]).splitlines()
    if not lines:
        return "", None
    state, _, elapsed = lines[0].partition("|")
    state = state.split()[0] if state.strip() else ""
    try:
        return state, int(elapsed)
    except ValueError:
        return state, None

def job_state(job_id):
    return job_info(job_id)[0]

def job_elapsed(job_id):
    return job_info(job_id)[1]

def me():
    return os.environ.get("USER") or _run(["whoami"]).strip()

def waiting_jobs(partition, own_names=()):
    """Jobs waiting for resources on ``partition``: any other user's, or our own main-chain ones.

    Our extra game players yield to both: to other people, and to our own
    learner/players chain so it can get back onto its GPUs.
    """
    user = me()
    out = _run(["squeue", "-h", "-p", partition, "-t", "PD", "-o", "%i|%u|%j|%r"])
    waiting = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        job, owner, name, reason = parts
        if reason not in WAITING_REASONS:
            continue
        if owner.split("@")[0] != user.split("@")[0] or name in own_names:
            waiting.append({"job": job, "user": owner, "name": name, "reason": reason})
    return waiting

_BAD_NODE_STATES = ("DRAIN", "DOWN", "RESERVED", "MAINT", "FAIL", "NOT_RESPONDING")

def node_resources(partition):
    """Free resources per usable node of ``partition``: ``{node: {"gpus", "cpus", "mem_mb"}}``."""
    nodes = set(_run(["sinfo", "-h", "-p", partition, "-N", "-o", "%N"]).split())
    free = {}
    for line in _run(["scontrol", "show", "node", "-o"]).splitlines():
        fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
        node = fields.get("NodeName")
        if node not in nodes:
            continue
        if any(bad in fields.get("State", "") for bad in _BAD_NODE_STATES):
            continue
        total = sum(int(m.group(1)) for m in re.finditer(r"gpu(?::[\w-]+)?:(\d+)", fields.get("Gres", "")))
        used = sum(int(m.group(1)) for m in re.finditer(r"gres/gpu(?::[\w-]+)?=(\d+)", fields.get("AllocTRES", "")))
        # AllocTRES may list both gres/gpu=N and gres/gpu:type=N for the same GPUs.
        typed = [int(m.group(1)) for m in re.finditer(r"gres/gpu:[\w-]+=(\d+)", fields.get("AllocTRES", ""))]
        if typed:
            used -= sum(typed)

        def num(key):
            try:
                return int(fields.get(key, "0"))
            except ValueError:
                return 0

        free[node] = {"gpus": max(total - used, 0), "cpus": max(num("CPUTot") - num("CPUAlloc"), 0),
                      "mem_mb": max(num("RealMemory") - num("AllocMem"), 0)}
    return free

def idle_gpus(partition):
    """Free GPUs per usable node of ``partition``: ``{node: count}``."""
    return {node: r["gpus"] for node, r in node_resources(partition).items() if r["gpus"] > 0}

def my_jobs(name_prefix):
    """Our jobs whose name starts with ``name_prefix``: list of ``{job, name, state, partition}``."""
    out = _run(["squeue", "-h", "-u", me(), "-o", "%i|%j|%T|%P"])
    jobs = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 4 and parts[1].startswith(name_prefix):
            jobs.append({"job": parts[0], "name": parts[1], "state": parts[2], "partition": parts[3]})
    return jobs

def maintain_extras(run_dir, partitions, extra_name, sbatch_path, own_names=(), max_extra=0,
                    log=print, submit=None):
    """Borrow idle GPUs for extra game players, politely.

    For each partition: request one 1-GPU extra job per idle GPU, but only if
    nobody (another user, or our own main chain) is waiting there and none of
    our extras is already pending there. ``max_extra`` caps our running +
    pending extras in total (0 = no cap). Returns the job ids submitted.
    """
    if stop_requested(run_dir):
        return []
    submit = submit or _sbatch
    ours = my_jobs(extra_name)
    submitted = []
    for partition in partitions:
        if waiting_jobs(partition, own_names):
            continue
        if any(j["partition"] == partition and j["state"] == "PENDING" for j in ours):
            continue
        free = sum(idle_gpus(partition).values())
        if max_extra > 0:
            free = min(free, max_extra - len(ours) - len(submitted))
        for _ in range(max(free, 0)):
            job = submit(["--partition", partition, "--gres", "gpu:1", "--job-name", extra_name,
                          "--export", f"ALL,EXTRA_PARTITION={partition}", sbatch_path])
            if job:
                submitted.append(job)
                log(f"[extras] borrowed a GPU on {partition}: job {job}")
    return submitted

def _sbatch(argv):
    out = _run(["sbatch", *argv])
    m = re.search(r"Submitted batch job (\d+)", out)
    return m.group(1) if m else None

# ----------------------------------------------------------------------------
# Job chain guard
# ----------------------------------------------------------------------------
MIN_RUN_SECONDS = 600  # a chain job that "completes" sooner than this counts as a crash

def chain_check(run_dir, role, prev_job, max_crashes=3, state_of=job_state, alert=None,
                elapsed_of=None, min_run=MIN_RUN_SECONDS, hold=False):
    """Decide what a newly started chain job does; returns ``(action, message)``.

    ``action`` is ``"run"``, ``"stop"`` (exit and cancel the chain's queued
    jobs) or ``"hold"`` (keep the GPUs, idle; only with ``hold=True``).

    STOP present -> ``"stop"``. Otherwise the previous job's end is counted: a
    job that ended normally (reached its time limit, was cancelled, or
    completed after at least ``min_run`` seconds -- ``elapsed_of(job)`` tells
    how long it ran) resets the crash count; a crash, or a completion sooner
    than ``min_run``, adds one, so a broken setup cannot burn through the
    queued follow-up jobs. At ``max_crashes`` in a row:

    * ``hold=False`` (gap chains): the chain stops and stays stopped (its
      queued follow-ups do not revive it) until start.sh removes RUN/chain/<role>.json.
    * ``hold=True`` (the main chains): this job keeps its GPUs idle instead, and
      the follow-ups stay queued, so nobody else gets the GPUs. The holding job
      ends at its time limit, which resets the count: the next job tries again.

    Crashes, a stop and the start of a hold are sent via ``notify`` (``alert`` overrides it).
    """
    if alert is None:
        from .notify import notify as alert
    paths = ensure_run_dir(run_dir)
    if stop_requested(run_dir):
        return "stop", "STOP file present"
    path = os.path.join(paths["chain"], f"{role}.json")
    try:
        with open(path) as f:
            status = json.load(f)
    except (FileNotFoundError, ValueError):
        status = {"consecutive_crashes": 0, "history": []}
    run_name = os.path.basename(os.path.normpath(run_dir))
    if status.get("stopped"):
        return "stop", f"chain stopped earlier ({status.get('reason', '')}); start.sh resets it"
    if prev_job:
        state = state_of(prev_job) or ""
        crashed = any(state.startswith(s) for s in CRASH_STATES)
        elapsed = elapsed_of(prev_job) if elapsed_of is not None else None
        if not crashed and state.startswith("COMPLETED") and elapsed is not None and elapsed < min_run:
            crashed, state = True, f"{state} after only {elapsed}s"
        status["consecutive_crashes"] = status.get("consecutive_crashes", 0) + 1 if crashed else 0
        if crashed and status["consecutive_crashes"] < max_crashes:
            alert("crash", f"Werewolf RL {run_name}: {role} job {prev_job} crashed ({state}), "
                           f"{status['consecutive_crashes']} in a row. The next job is taking over.", run_dir)
        status["history"] = (status.get("history", []) + [{"job": prev_job, "state": state, "t": time.time()}])[-50:]
    limit = status.get("consecutive_crashes", 0) >= max_crashes
    status["stopped"] = limit and not hold
    status["holding"] = limit and hold
    status["reason"] = f"{status['consecutive_crashes']} consecutive crashes" if limit else ""
    atomic_write_json(path, status)
    if limit and hold:
        alert("stopped", f"Werewolf RL {run_name}: the {role} chain crashed {status['consecutive_crashes']} times "
                         f"in a row (last: job {prev_job}). Its next job keeps the GPUs but leaves them idle until "
                         f"its time limit, then tries again. Fix it and run start.sh {run_name} to retry at once.",
              run_dir)
        return "hold", f"previous {status['consecutive_crashes']} jobs crashed (see {path}); holding the GPUs idle"
    if limit:
        alert("stopped", f"Werewolf RL {run_name}: the {role} chain STOPPED after {max_crashes} crashes in a "
                         f"row (last: job {prev_job}). Check status.sh {run_name}, fix, then start.sh {run_name}.",
              run_dir)
        return "stop", f"previous {max_crashes} jobs crashed (see {path}); chain stopped"
    return "run", f"ok (consecutive crashes: {status.get('consecutive_crashes', 0)})"


def chain_guard(run_dir, role, prev_job, max_crashes=3, state_of=job_state, alert=None,
                elapsed_of=None, min_run=MIN_RUN_SECONDS):
    """``chain_check`` without holding: ``(ok, message)``, ok = the job should run."""
    action, msg = chain_check(run_dir, role, prev_job, max_crashes, state_of, alert, elapsed_of, min_run)
    return action == "run", msg

def idle_until_killed(should_release=None, poll=60, log=print):
    """Hold the allocation until Slurm's time limit ends the job.

    ``should_release()`` (optional) is polled; when it returns a reason the
    function returns it so the caller can exit and free the GPUs.
    """
    log("[info] next item would not fit in the remaining time; holding until the time limit")
    while True:
        if should_release is not None:
            reason = should_release()
            if reason:
                return reason
        time.sleep(poll)

def gpu_identity():
    """``(host, uuid)`` of this job's first visible GPU."""
    host = socket.gethostname().split(".")[0]
    out = _run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]).split()
    return host, (out[0] if out else os.environ.get("CUDA_VISIBLE_DEVICES", "gpu"))

def choose_extra_role(run_dir, identity=None, alive=None):
    """Role ("train"/"play") of a newly borrowed GPU.

    A GPU borrowed before takes the opposite of its last role; a GPU never
    borrowed takes whichever role has fewer live extras (ties: play). The
    choice is recorded per physical GPU in ``RUN/extras/``.
    """
    host, gpu = identity or gpu_identity()
    folder = os.path.join(run_dir, "extras")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, re.sub(r"[^\w.-]", "_", f"gpu_{host}_{gpu}") + ".json")
    try:
        with open(path) as f:
            last = json.load(f).get("last_role")
    except (FileNotFoundError, ValueError):
        last = None
    if last in ("train", "play"):
        role = "play" if last == "train" else "train"
    else:
        if alive is None:
            from .distributed import alive_heartbeats

            alive = {"train": sum(hb.get("role") == "extra" for hb in alive_heartbeats(run_dir, "trainer_", 120)),
                     "play": sum(hb.get("role") == "extra" for hb in alive_heartbeats(run_dir, "player_", 120))}
        role = "train" if alive["train"] < alive["play"] else "play"
    atomic_write_json(path, {"last_role": role, "t": time.time(), "host": host, "gpu": gpu,
                             "job": os.environ.get("SLURM_JOB_ID")})
    return role

def main(argv=None):
    """CLI for the sbatch scripts: ``guard``, ``write-config`` and ``extra-role``."""
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("guard")
    g.add_argument("--run_dir", required=True)
    g.add_argument("--role", required=True)
    g.add_argument("--prev_job", default="")
    g.add_argument("--max_crashes", type=int, default=3)
    g.add_argument("--hold", action="store_true",
                   help="at max_crashes keep the GPUs idle (exit 4) instead of stopping the chain (exit 3)")
    w = sub.add_parser("write-config", help="Write RUN/game_config.json from game flags.")
    w.add_argument("--run_dir", required=True)
    e = sub.add_parser("extra-role", help="Print 'train' or 'play' for this borrowed GPU.")
    e.add_argument("--run_dir", required=True)
    a, rest = p.parse_known_args(argv)
    if a.cmd == "extra-role":
        print(choose_extra_role(a.run_dir))
        return
    if a.cmd == "guard":
        action, msg = chain_check(a.run_dir, a.role, a.prev_job, a.max_crashes, elapsed_of=job_elapsed,
                                  hold=a.hold)
        print(f"[chain:{a.role}] {msg}", flush=True)
        raise SystemExit({"run": 0, "stop": 3, "hold": 4}[action])
    if a.cmd == "write-config":
        from .game import add_game_args

        gp = add_game_args(argparse.ArgumentParser())
        game_args = gp.parse_args(rest)
        ensure_run_dir(a.run_dir)
        write_game_config(a.run_dir, game_args, sorted(vars(game_args)))
        print(json.dumps(vars(game_args), indent=2))

if __name__ == "__main__":
    main()