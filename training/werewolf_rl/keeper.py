"""Login-node keeper for the always-on Werewolf RL loop (cron, every 5 minutes).

Rosie hands out GPUs first come, first served, and its interactive QOS runs
at most ``job_cap`` (4) jobs per user at once. A run is a few *chains*
(start.sh ``CHAINS``, e.g. 4xH100 + 8xV100 + 4xV100): self-renewing jobs that
each keep ``depth`` follow-up jobs queued (chain.sh), so every 24 h handover
goes to a job that has been waiting for days and is first in line.

Each ``tick`` the keeper

1. re-creates a chain that has no jobs left, and tops every chain's queue up
   to 1 + ``depth`` jobs (not for a chain stopped after 3 crashes, nor after
   RUN/STOP);
2. fills the job slots the chains leave (``gapN`` jobs) to cover the chains'
   *shortfall* -- GPUs they should hold but don't yet (their job is still
   waiting): a gap job is sized to the largest free hole on ``gap_partitions``
   (``test_only``: that ``sbatch --test-only`` says starts now). It holds its GPUs and
   renews itself like a chain (one follow-up), until the keeper ends it;
3. ends gap jobs that are no longer needed: when the chains hold their target,
   when one of our chain jobs is blocked by the per-user job limit, or when a
   gap job holds GPUs a waiting chain job needs to start;
4. cancels a gap job still waiting after ``patience_minutes`` (that hole
   closed) and does not try that size on that partition again for
   ``backoff_minutes``;
5. swaps a running gap job for a bigger one when a hole at least
   ``upgrade_by`` GPUs bigger opens (counting the GPUs it holds itself).

Settings live in RUN/launch.json (``write-launch``, called by start.sh), which
also writes RUN/launch.env for the jobs. Decisions are logged to
RUN/keeper/keeper.log; RUN/keeper/state.json remembers waiting times/backoffs.

    python -m training.werewolf_rl.keeper tick --run_dir RUN [--dry_run]
    python -m training.werewolf_rl.keeper install-cron --run_dir RUN   (start.sh does this)
    python -m training.werewolf_rl.keeper uninstall-cron --run_dir RUN (stop.sh does this)
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time

from . import continuous as cont

DEFAULTS = {
    "depth": 14,
    "report_minutes": 60,   # hourly ntfy GPU report (report.py; 0 = off)
    "time_min": "00:30:00",  # sbatch --time-min for chain and gap jobs ("" = off): see README 2.3
    # where the gap slot grabs GPUs beyond the targets (and gives them back when another user needs them)
    "extra_partitions": ["dgxh100", "dgx"],
    "job_cap": 4,
    "gap_partitions": ["dgx", "dgxh100"],
    "gap_max_gpus": 8,
    "patience_minutes": 10,
    "backoff_minutes": 60,
    "upgrade_by": 2,
    "start_tolerance_minutes": 5,
    "test_only": False,
    "cpus_per_gpu": 4,
    "mem_per_gpu_gb": 48,
}
WAITING = ("Resources", "Priority", "None")
# Work per GPU, relative to a V100: picks between extra holes on different partitions (the
# gap slot can hold only one). Measured on run1 (2026-11-20): games 21.3 vs 9.2 columns/s per
# GPU on H100 vs V100, trainer tasks ~4.5x faster; so 3 free H100 beat 5 free V100, 8 V100 beat 2 H100.
GPU_SPEED = {"dgxh100": 2.3, "dgx": 1.0}
CRON_TAG = "# werewolf_rl keeper"
LIVE = ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED")


# ----------------------------------------------------------------------------
# Launch settings
# ----------------------------------------------------------------------------
def paths(run_dir):
    k = os.path.join(run_dir, "keeper")
    return {"json": os.path.join(run_dir, "launch.json"), "env": os.path.join(run_dir, "launch.env"),
            "dir": k, "state": os.path.join(k, "state.json"), "log": os.path.join(k, "keeper.log"),
            "lock": os.path.join(k, "lock")}


def parse_chains(text):
    """``"h100:dgxh100:4 v100:dgx:8"`` -> ``[{"role", "partition", "gpus"}, ...]``."""
    chains = []
    for item in text.replace(",", " ").split():
        try:
            role, partition, gpus = item.split(":")
            gpus = int(gpus)
        except ValueError:
            raise ValueError(f"bad chain {item!r}: want role:partition:gpus") from None
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", role) or role.startswith(("gap", "extra")):
            raise ValueError(f"bad chain role {role!r}: letters/digits, not gap*/extra*")
        if gpus < 1:
            raise ValueError(f"chain {role}: needs at least 1 GPU")
        chains.append({"role": role, "partition": partition, "gpus": gpus})
    if len({c["role"] for c in chains}) != len(chains):
        raise ValueError("chain roles must be unique")
    return chains


def sbatch_opts(spec, partition, gpus):
    """Resource options for a ``gpus``-GPU job (per-GPU CPUs and memory from ``spec``)."""
    # --time-min: if an older job's reservation keeps the whole node free for later, Slurm may start
    # this job now with its time limit cut down to time_min, instead of leaving the GPUs empty.
    return [f"--partition={partition}", f"--gres=gpu:{gpus}", f"--cpus-per-task={gpus * spec['cpus_per_gpu']}",
            f"--mem={gpus * spec['mem_per_gpu_gb']}G",
            *([f"--time-min={spec['time_min']}"] if spec.get("time_min") else []),
            *shlex.split(spec.get("extra_sbatch", ""))]


def job_name(spec, role):
    return f"wwc-{spec['run_name']}-{role}"


def gap_roles(spec):
    n = spec.get("gap_slots")
    if n is None:
        n = max(0, spec["job_cap"] - len(spec["chains"])) if spec["job_cap"] > 0 else 1
    return [f"gap{k}" for k in range(1, n + 1)]


def complete_spec(spec):
    full = dict(DEFAULTS)
    full.update({k: v for k, v in spec.items() if v is not None})
    for key in ("gap_partitions", "extra_partitions"):
        if isinstance(full[key], str):
            full[key] = [p for p in full[key].split(",") if p]
    return full


def write_launch(run_dir, spec):
    """Write RUN/launch.json (keeper) and RUN/launch.env (sourced by every chain job at start)."""
    spec = complete_spec(spec)
    p = paths(run_dir)
    os.makedirs(p["dir"], exist_ok=True)
    cont.atomic_write_json(p["json"], spec)
    env = {"CHAIN_DEPTH": spec["depth"], "CPUS_PER_GPU": spec["cpus_per_gpu"],
           "MEM_PER_GPU_GB": spec["mem_per_gpu_gb"],
           "CHAIN_NAMES": ",".join(job_name(spec, c["role"]) for c in spec["chains"]),
           "CHAIN_EXTRA_OPTS": spec.get("extra_sbatch", ""), "CHAIN_TIME_MIN": spec.get("time_min") or "",
           "EXTRA_GPU_PARTITIONS": ",".join(spec.get("extra_partitions") or [])}
    for c in spec["chains"]:
        env[f"CHAIN_OPTS_{c['role']}"] = " ".join(sbatch_opts(spec, c["partition"], c["gpus"]))
    env.update(spec.get("env", {}))
    lines = ["# Written by start.sh (training/werewolf_rl/keeper.py write-launch). Every chain job",
             "# reads this when it starts (chain.sh), so changes reach jobs already queued.",
             *(f"export {k}={shlex.quote(str(v))}" for k, v in env.items())]
    tmp = p["env"] + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, p["env"])
    return spec


def read_launch(run_dir):
    with open(paths(run_dir)["json"]) as f:
        return complete_spec(json.load(f))


def detect_job_cap(default=4):
    """Running-job limit per user of this account's QOS (MaxJobsPU), or ``default``."""
    user = cont.me()
    qos = cont._run(["sacctmgr", "-nP", "show", "assoc", f"user={user}", "format=QOS"]).split()
    caps = []
    for name in {q for line in qos for q in line.split(",") if q}:
        value = cont._run(["sacctmgr", "-nP", "show", "qos", name, "format=MaxJobsPU"]).strip()
        if value.isdigit():
            caps.append(int(value))
    return min(caps) if caps else default


# ----------------------------------------------------------------------------
# Slurm access (replaced by a fake in the tests)
# ----------------------------------------------------------------------------
def _gpus_of(tres):
    m = re.search(r"gpu(?::[\w-]+)?:(\d+)", tres or "")
    return int(m.group(1)) if m else 0


class Slurm:
    def __init__(self, user=None):
        self.user = user or cont.me()

    def jobs(self, prefix):
        out = cont._run(["squeue", "-h", "-u", self.user, "-o", "%i|%j|%T|%P|%b|%N|%r|%E|%M|%L"])
        jobs = []
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) != 10 or not parts[1].startswith(prefix):
                continue
            jid, name, state, part, tres, node, reason, dep, elapsed, left = parts
            jobs.append({"id": jid, "name": name, "state": state, "partition": part, "gpus": _gpus_of(tres),
                         "node": node, "reason": reason, "dependency": "" if dep == "(null)" else dep,
                         "elapsed": cont.parse_slurm_duration(elapsed), "time_left": cont.parse_slurm_duration(left)})
        return jobs

    def waiting_requests(self, partition, limit=25):
        """Other users' jobs on ``partition`` waiting for resources, with their per-node needs
        (``parse_request``); jobs held back by their own limits or dependencies are left out."""
        out = cont._run(["squeue", "-h", "-p", partition, "-t", "PD", "--sort=i", "-o", "%i|%u|%r"])
        reqs = []
        for line in out.splitlines():
            jid, user, reason = (line.split("|") + ["", "", ""])[:3]
            if user == self.user or reason not in WAITING_FOR_RESOURCES or len(reqs) >= limit:
                continue
            fields = dict(f.split("=", 1) for f in cont._run(["scontrol", "show", "job", "-o", jid]).split()
                          if "=" in f)
            req = parse_request(fields, hostnames=_hostnames)
            if req["gpus"] > 0:
                reqs.append(dict(req, job=jid, user=user.split("@")[0], reason=reason))
        return reqs

    def queue_blocker(self, partition):
        """The oldest job of another user waiting for resources on ``partition`` (the one the
        scheduler is keeping nodes free for): ``{"job", "user", "start", "planned_nodes"}`` or None."""
        out = cont._run(["squeue", "-h", "-p", partition, "-t", "PD", "--sort=i", "-o", "%i|%u|%r"])
        for line in out.splitlines():
            jid, user, reason = (line.split("|") + ["", "", ""])[:3]
            if user != self.user and reason == "Resources":
                start = cont._run(["squeue", "-h", "--start", "-j", jid, "-o", "%S"]).strip()
                nodes = cont._run(["sinfo", "-h", "-p", partition, "-N", "-o", "%N|%T"])
                planned = sorted({n for n, _, st in (x.partition("|") for x in nodes.splitlines())
                                  if "plan" in st.lower()})
                return {"job": jid, "user": user.split("@")[0],
                        "start": None if start in ("", "N/A", "Unknown") else start, "planned_nodes": planned}
        return None

    def nodes(self, partitions):
        return {p: cont.node_resources(p) for p in partitions}

    def will_start(self, argv):
        """Epoch time ``sbatch --test-only`` predicts the job would start (None if it can't tell)."""
        try:
            r = subprocess.run(["sbatch", "--test-only", *argv, "--wrap", "true"], capture_output=True, text=True,
                               timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        m = re.search(r"to start at (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", r.stderr + r.stdout)
        return time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) if m else None

    def submit(self, argv, env):
        r = subprocess.run(["sbatch", *argv], capture_output=True, text=True, env=env, timeout=120, check=False)
        m = re.search(r"Submitted batch job (\d+)", r.stdout)
        if not m:
            raise RuntimeError(f"sbatch failed: {r.stderr.strip() or r.stdout.strip()}")
        return m.group(1)

    def cancel_name(self, name):
        subprocess.run(["scancel", "-u", self.user, f"--name={name}"], capture_output=True, timeout=60, check=False)


# ----------------------------------------------------------------------------
# Decisions
# ----------------------------------------------------------------------------
def _role_of(spec, name):
    prefix = f"wwc-{spec['run_name']}-"
    return name[len(prefix):] if name.startswith(prefix) else None


def _jid(job):
    return int(re.match(r"\d+", job["id"]).group(0))


def by_role(spec, jobs):
    roles = {}
    for j in jobs:
        role = _role_of(spec, j["name"])
        if role and j["state"] in LIVE:
            roles.setdefault(role, []).append(j)
    for js in roles.values():
        js.sort(key=_jid)
    return roles


def stopped_chains(run_dir, spec):
    """Chain and gap roles stopped after 3 crashes in a row (start.sh resets them)."""
    stopped = set()
    for role in [c["role"] for c in spec["chains"]] + gap_roles(spec):
        try:
            with open(os.path.join(run_dir, "chain", f"{role}.json")) as f:
                if json.load(f).get("stopped"):
                    stopped.add(role)
        except (FileNotFoundError, ValueError):
            pass
    return stopped


def shortfall(spec, roles):
    """GPUs the chains should hold but don't (their running jobs vs their targets), per partition."""
    short = {}
    for c in spec["chains"]:
        held = sum(j["gpus"] for j in roles.get(c["role"], []) if j["state"] == "RUNNING")
        short[c["partition"]] = short.get(c["partition"], 0) + max(c["gpus"] - held, 0)
    return short


# ----------------------------------------------------------------------------
# Extra GPUs: beyond the chains' targets, held by the gap slot and given back
# ----------------------------------------------------------------------------
WAITING_FOR_RESOURCES = ("Resources", "Priority", "ReqNodeNotAvail")


def _mem_mb(text):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)", (text or "").strip())
    if not m:
        return 0
    return int(float(m.group(1)) * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[m.group(2) or "M"])


def _hostnames(hostlist):
    return cont._run(["scontrol", "show", "hostnames", hostlist]).split()


def parse_request(fields, hostnames=None):
    """Per-node needs of a pending job from ``scontrol show job -o`` fields:
    ``{"nodes", "gpus", "cpus", "mem_mb", "partitions", "req_nodes", "exc_nodes"}`` (per node)."""
    nodes = int(re.match(r"\d+", fields.get("NumNodes", "1")).group(0) or 1)
    gpus = _gpus_of(fields.get("TresPerNode", ""))
    if not gpus:
        total = _gpus_of(fields.get("TresPerJob", "")) or int(
            (re.search(r"gres/gpu=(\d+)", fields.get("TRES", "")) or [0, 0])[1])
        gpus = -(-total // nodes) if total else 0
    try:
        cpus = -(-int(fields.get("NumCPUs", "1")) // nodes)
    except ValueError:
        cpus = 1
    if fields.get("MinMemoryNode"):
        mem = _mem_mb(fields["MinMemoryNode"])
    elif fields.get("MinMemoryCPU"):
        mem = _mem_mb(fields["MinMemoryCPU"]) * cpus
    else:
        mem = 0

    def node_set(key):
        value = fields.get(key, "(null)")
        if value in ("", "(null)"):
            return set()
        return set(hostnames(value)) if hostnames else {value}

    return {"nodes": nodes, "gpus": gpus, "cpus": cpus, "mem_mb": mem,
            "partitions": [p for p in fields.get("Partition", "").split(",") if p],
            "req_nodes": node_set("ReqNodeList"), "exc_nodes": node_set("ExcNodeList")}


def request_fits(req, free_nodes, give=None):
    """Whether pending job ``req`` could start on ``free_nodes`` (``{node: free}``), counting
    ``give`` = ``{node: (gpus, cpus, mem_mb)}`` (what we would release) as free too."""
    ok = 0
    for node, free in free_nodes.items():
        if (req["req_nodes"] and node not in req["req_nodes"]) or node in req["exc_nodes"]:
            continue
        g, c, m = free["gpus"], free["cpus"], free["mem_mb"]
        if give and node in give:
            g, c, m = g + give[node][0], c + give[node][1], m + give[node][2]
        if g >= req["gpus"] and c >= req["cpus"] and m >= req["mem_mb"]:
            ok += 1
    return ok >= req["nodes"]


def _free_nodes(nodes, partitions):
    merged = {}
    for p in partitions:
        merged.update(nodes.get(p, {}))
    return merged


def is_extra(spec, roles, job):
    """A gap job is *extra* when it sits on an extra partition and the chains there already
    hold their target without it (then it is given back when someone else needs the GPUs)."""
    part = job["partition"]
    if part not in spec.get("extra_partitions", []):
        return False
    target = sum(c["gpus"] for c in spec["chains"] if c["partition"] == part)
    held = sum(j["gpus"] for c in spec["chains"] if c["partition"] == part
               for j in roles.get(c["role"], []) if j["state"] == "RUNNING")
    return held >= target


def yield_reason(spec, roles, job, requests, nodes):
    """Why running gap job ``job`` should give its GPUs back now (None: keep them): it is extra,
    and another user's waiting job would fit once these GPUs are free (and doesn't fit without)."""
    if not job or job["state"] != "RUNNING" or not is_extra(spec, roles, job):
        return None
    give = {job["node"]: (job["gpus"], job["gpus"] * spec["cpus_per_gpu"],
                          job["gpus"] * spec["mem_per_gpu_gb"] * 1024)}
    for r in requests or []:
        if job["partition"] not in r["partitions"]:
            continue
        free = _free_nodes(nodes, r["partitions"])
        if not request_fits(r, free) and request_fits(r, free, give):
            return (f"another user's job {r['job']} is waiting for {r['nodes']}x{r['gpus']} GPU and fits "
                    f"once these {job['gpus']} are free")
    return None


def fit(spec, free):
    """Largest job (GPUs) a node with ``free`` resources fits."""
    return max(0, min(free["gpus"], free["cpus"] // spec["cpus_per_gpu"],
                      free["mem_mb"] // (spec["mem_per_gpu_gb"] * 1024), spec["gap_max_gpus"]))


def holes(spec, nodes, limit, short=None, backoff=None, now=0.0, own=None, parts=None):
    """Candidate gap jobs ``(partition, node, gpus)``, best first.

    ``own`` = ``{node: (gpus, cpus, mem_mb)}`` held by a gap job we would give
    back, counted as free. Sizes are capped at ``limit``; sizes in backoff are skipped.
    """
    short, backoff, own = short or {}, backoff or {}, own or {}
    order = parts or spec["gap_partitions"]
    cands = []
    for part in order:
        for node, free in nodes.get(part, {}).items():
            free = dict(free)
            if node in own:
                g, c, m = own[node]
                free = {"gpus": free["gpus"] + g, "cpus": free["cpus"] + c, "mem_mb": free["mem_mb"] + m}
            n = min(fit(spec, free), limit)
            while n > 0 and backoff.get(f"{part}:{n}", 0) > now:
                n -= 1
            if n > 0:
                cands.append((part, node, n))
    cands.sort(key=lambda c: (-c[2], 0 if short.get(c[0], 0) > 0 else 1, order.index(c[0]), c[1]))
    return cands


def plan(spec, roles, nodes, state, now, stopped=(), requests=None):
    """This tick's maintenance actions (no I/O).

    Returns ``(queue, cancel, fill)``: ``queue`` = ``[(role, after_job_or_None, count)]``
    chain jobs to add, ``cancel`` = ``[(role, why)]`` gap chains to end, and
    ``fill`` = gap roles that should get a new job (sized later from live holes).
    Updates ``state`` (waiting times, backoffs).
    """
    queue, cancel = [], []
    for c in spec["chains"]:
        if c["role"] in stopped:
            continue
        js = roles.get(c["role"], [])
        need = 1 + spec["depth"] - len(js)
        if need > 0:
            queue.append((c["role"], js[-1]["id"] if js else None, need))

    gaps = gap_roles(spec)
    short = shortfall(spec, roles)
    total_short = sum(short.values())
    chain_jobs = [j for c in spec["chains"] for j in roles.get(c["role"], [])]
    capped = [j for j in chain_jobs if j["reason"].startswith("QOSMaxJobsPerUser")]
    live_gaps = {g: roles.get(g, []) for g in gaps if roles.get(g)}
    running_gap = {g: next((j for j in js if j["state"] == "RUNNING"), None) for g, js in live_gaps.items()}
    held_by_gaps = sum(j["gpus"] for j in running_gap.values() if j)
    pending_since = state.setdefault("pending_since", {})
    backoff = state.setdefault("backoff", {})
    for role in [r for r in roles if r.startswith("extra")]:  # borrowed extras would eat job slots
        cancel.append((role, "extras are replaced by gap jobs"))

    def end(role, why):
        if role not in [r for r, _ in cancel]:
            cancel.append((role, why))

    for g, js in live_gaps.items():
        head = running_gap.get(g) or js[0]
        extra = is_extra(spec, roles, head)
        why = yield_reason(spec, roles, running_gap.get(g), requests, nodes)
        if why:
            end(g, f"giving back extra GPUs: {why}")
        elif capped:
            end(g, f"chain job {capped[0]['id']} is blocked by the per-user job limit")
        elif extra:
            continue  # extra GPUs beyond the targets: kept until someone else needs them
        elif total_short <= 0:
            end(g, "the chains hold their target")
        elif held_by_gaps > total_short:
            end(g, f"gap jobs hold {held_by_gaps} GPUs but the chains only lack {total_short}")
    # A waiting chain job that would start if a gap job gave its GPUs back.
    for j in chain_jobs:
        if j["state"] != "PENDING" or j["dependency"] or j["reason"] not in WAITING:
            continue
        for g, rj in running_gap.items():
            if not rj or rj["partition"] != j["partition"]:
                continue
            free = nodes.get(rj["partition"], {}).get(rj["node"])
            if free is None:
                continue
            alone = min(free["gpus"], free["cpus"] // spec["cpus_per_gpu"])
            together = min(free["gpus"] + rj["gpus"], (free["cpus"] + rj["gpus"] * spec["cpus_per_gpu"])
                           // spec["cpus_per_gpu"])
            if alone < j["gpus"] <= together:
                end(g, f"chain job {j['id']} needs the GPUs on {rj['node']}")
    # Gap jobs that have waited too long; bigger holes for running ones.
    seen = set()
    for g, js in live_gaps.items():
        for j in js:
            seen.add(j["id"])
            if j["state"] == "PENDING" and not j["dependency"]:
                since = pending_since.setdefault(j["id"], now)
                if now - since > spec["patience_minutes"] * 60:
                    backoff[f"{j['partition']}:{j['gpus']}"] = now + spec["backoff_minutes"] * 60
                    end(g, f"job {j['id']} waited {int((now - since) / 60)} min for {j['gpus']} GPUs")
        rj = running_gap.get(g)
        if rj and g not in [r for r, _ in cancel]:
            own = {rj["node"]: (rj["gpus"], rj["gpus"] * spec["cpus_per_gpu"],
                                rj["gpus"] * spec["mem_per_gpu_gb"] * 1024)}
            best = holes(spec, nodes, total_short, short, backoff, now, own=own)
            if best and best[0][2] >= rj["gpus"] + spec["upgrade_by"]:
                end(g, f"a {best[0][2]}-GPU hole opened ({best[0][1]}); job {rj['id']} has {rj['gpus']}")
    for jid in list(pending_since):
        if jid not in seen:
            del pending_since[jid]
    for key in [k for k, until in backoff.items() if until <= now]:
        del backoff[key]

    cancelled = {r for r, _ in cancel}
    extras_ok = total_short <= 0 and spec.get("extra_partitions") and not any(
        why.startswith("giving back") for _, why in cancel)
    fill = [g for g in gaps if (g not in live_gaps or g in cancelled) and g not in stopped] \
        if (total_short > 0 or extras_ok) and not capped else []
    return queue, cancel, fill


# ----------------------------------------------------------------------------
# One tick
# ----------------------------------------------------------------------------
def job_env(spec, role, prev_job=None):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in spec.get("env", {}).items()})
    env.update({"REPO": spec["repo"], "RUN_NAME": spec["run_name"], "RUN_DIR": spec["run_dir"],
                "CHAIN_SCRIPT": spec["sbatch"], "CHAIN_ROLE": role})
    user = cont.me()  # cron may not set USER; chain.sh needs it for squeue/scancel
    env.setdefault("USER", user)
    env.setdefault("LOGNAME", user)
    bindir = spec.get("python_bin")
    if bindir and bindir not in env.get("PATH", "").split(":"):
        env["PATH"] = f"{bindir}:{env.get('PATH', '/usr/bin:/bin')}"
    env.pop("PREV_JOB", None)
    env.pop("TRAIN_GPUS", None)
    if prev_job:
        env["PREV_JOB"] = str(prev_job)
    return env


def submit_chain_jobs(spec, slurm, role, after, count, log):
    c = next(c for c in spec["chains"] if c["role"] == role)
    ids = []
    for _ in range(count):
        argv = [*sbatch_opts(spec, c["partition"], c["gpus"]), f"--job-name={job_name(spec, role)}", "--export=ALL"]
        if after:
            argv.append(f"--dependency=afterany:{after}")
        argv.append(spec["sbatch"])
        after = slurm.submit(argv, job_env(spec, role, after))
        ids.append(after)
        with open(os.path.join(spec["run_dir"], "chain", f"{role}_jobs.log"), "a") as f:
            f.write(f"[keeper] {time.strftime('%F %T')} queued {role} job {after} ({c['gpus']} GPUs on "
                    f"{c['partition']})\n")
    log(f"queued {role} job(s) {', '.join(ids)}")
    return ids


def tick(run_dir, slurm=None, now=None, dry_run=False, log=print, wait=True):
    """One keeper pass. Returns a summary dict (also stored in RUN/keeper/state.json)."""
    spec = read_launch(run_dir)
    slurm = slurm or Slurm()
    p = paths(run_dir)
    os.makedirs(p["dir"], exist_ok=True)
    os.makedirs(os.path.join(run_dir, "chain"), exist_ok=True)
    now = time.time() if now is None else now
    try:
        with open(p["state"]) as f:
            state = json.load(f)
    except (FileNotFoundError, ValueError):
        state = {}
    summary = {"t": now, "host": os.uname().nodename.split(".")[0], "actions": []}

    def act(msg):
        summary["actions"].append(msg)
        log(("[dry run] " if dry_run else "") + msg)

    if cont.stop_requested(run_dir):
        summary["actions"].append("STOP present: nothing to do")
    else:
        prefix = f"wwc-{spec['run_name']}-"
        roles = by_role(spec, slurm.jobs(prefix))
        node_parts = list(dict.fromkeys(spec["gap_partitions"] + spec.get("extra_partitions", [])))
        nodes = slurm.nodes(node_parts)
        requests = []
        if spec.get("extra_partitions") and hasattr(slurm, "waiting_requests"):
            for part in spec["extra_partitions"]:
                requests += [r for r in slurm.waiting_requests(part) if r["job"] not in {x["job"] for x in requests}]
        queue, cancel, fill = plan(spec, roles, nodes, state, now, stopped_chains(run_dir, spec), requests)
        for role, after, count in queue:
            act(f"chain {role}: queue {count} job(s) after {after or 'nothing'}")
            if not dry_run:
                submit_chain_jobs(spec, slurm, role, after, count, log)
        for role, why in cancel:
            act(f"end {role}: {why}")
            if not dry_run:
                slurm.cancel_name(job_name(spec, role))
        if fill and cancel and not dry_run and wait:  # let the cancelled jobs release their GPUs
            gone = {job_name(spec, r) for r, _ in cancel}
            for _ in range(18):
                if not any(j["name"] in gone for j in slurm.jobs(prefix)):
                    break
                time.sleep(5)
            nodes = slurm.nodes(node_parts)
            roles = by_role(spec, slurm.jobs(prefix))
        short = shortfall(spec, roles)
        gap_gpus = sum(j["gpus"] for g in gap_roles(spec) if g not in fill for j in roles.get(g, [])
                       if j["state"] == "RUNNING")
        limit = sum(short.values()) - gap_gpus
        parts, mode = None, "chains lack"
        if sum(short.values()) <= 0 and spec.get("extra_partitions"):
            # Every chain holds its target: the gap slot grabs extra GPUs, but never ones a waiting
            # job of another user could use (it gives them back as soon as one could: yield_reason).
            parts, mode, limit = spec["extra_partitions"], "extra", spec["gap_max_gpus"]
        running = sum(1 for js in roles.values() for j in js if j["state"] == "RUNNING")
        for g in fill:
            if limit <= 0 or running >= spec["job_cap"] > 0:
                break
            tried = 0
            cands = holes(spec, nodes, limit, short, state.get("backoff"), now, parts=parts)
            if mode == "extra":
                # Not on a partition where another user's waiting job could use the free GPUs now.
                wanted = {}
                for r in requests:
                    if request_fits(r, _free_nodes(nodes, r["partitions"])):
                        for rp in r["partitions"]:
                            wanted.setdefault(rp, r)
                for wp in [x for x in parts if x in wanted and any(c[0] == x for c in cands)]:
                    act(f"{g}: no extra GPUs taken on {wp}: another user's job {wanted[wp]['job']} could use them")
                cands = [c for c in cands if c[0] not in wanted]
                # Most work first: an H100 does ~2.3x a V100's (GPU_SPEED).
                cands.sort(key=lambda c: -c[2] * GPU_SPEED.get(c[0], 1.0))
            for part, node, n in cands:
                if tried >= 3:
                    break
                tried += 1
                argv = [*sbatch_opts(spec, part, n), "--nodes=1", "--time=1-00:00:00",
                        f"--job-name={job_name(spec, g)}"]
                # sbatch --test-only simulates the whole backlog (including jobs the per-user limit
                # holds back) and predicts days for jobs that start in seconds, so by default just
                # submit: a job that is still waiting after patience_minutes is cancelled and backed off.
                start = slurm.will_start(argv) if spec.get("test_only") else now
                if start is None or start > now + spec["start_tolerance_minutes"] * 60:
                    act(f"{g}: {n} GPU(s) on {part} ({node}) would not start now"
                        f"{'' if start is None else ' (predicted ' + time.strftime('%F %T', time.localtime(start)) + ')'}")
                    continue
                act(f"{g}: start a {n}-GPU job on {part} (hole on {node}); "
                    + (f"chains lack {limit} GPUs" if mode != "extra" else
                       "extra GPUs, given back when another user needs them"))
                if not dry_run:
                    jid = slurm.submit([*sbatch_opts(spec, part, n), f"--job-name={job_name(spec, g)}",
                                        "--export=ALL", spec["sbatch"]], job_env(spec, g))
                    act(f"{g}: submitted job {jid}")
                    nodes.get(part, {}).get(node, {}).update(gpus=0)
                limit -= n
                running += 1
                break
    summary["state"] = state
    if not dry_run and not cont.stop_requested(run_dir):
        try:  # the hourly ntfy GPU report (report.py); never let it break the tick
            from . import report

            if report.maybe_send_report(run_dir, spec, state, now, slurm):
                act("sent the hourly GPU report")
        except Exception as exc:  # noqa: BLE001
            act(f"hourly GPU report failed: {exc}")
    if not dry_run:
        state["last_tick"] = now
        state["host"] = summary["host"]
        state["last_actions"] = summary["actions"][-20:]
        cont.atomic_write_json(p["state"], state)
    return summary


def locked_tick(run_dir, **kw):
    p = paths(run_dir)
    os.makedirs(p["dir"], exist_ok=True)
    with open(p["lock"], "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {"actions": ["another tick is running"]}
        return tick(run_dir, **kw)


# ----------------------------------------------------------------------------
# Cron
# ----------------------------------------------------------------------------
def cron_line(run_dir, repo, python=None):
    python = python or sys.executable
    log = paths(run_dir)["log"]
    return (f"*/5 * * * * cd {shlex.quote(repo)} && {shlex.quote(python)} -m training.werewolf_rl.keeper tick "
            f"--run_dir {shlex.quote(run_dir)} >> {shlex.quote(log)} 2>&1 {CRON_TAG} {run_dir}")


def install_cron(run_dir, repo, remove=False):
    """Add (or remove) this run's keeper line in this login node's crontab; other lines are kept."""
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
    mine = f"{CRON_TAG} {run_dir}"
    lines = [ln for ln in (current.stdout.splitlines() if current.returncode == 0 else []) if not ln.endswith(mine)]
    if not remove:
        lines.append(cron_line(run_dir, repo))
    r = subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True, capture_output=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"crontab failed: {r.stderr.strip()}")
    return lines


def keeper_status(run_dir):
    """``(age_seconds or None, host, last_actions)`` of the last tick."""
    try:
        with open(paths(run_dir)["state"]) as f:
            state = json.load(f)
    except (FileNotFoundError, ValueError):
        return None, "", []
    t = state.get("last_tick")
    return (time.time() - t if t else None), state.get("host", ""), state.get("last_actions", [])


GPU_KIND = {"dgxh100": "H100", "dgx": "V100", "teaching": "T4"}


def idle_jobs(run_dir):
    """``{job_id: "IDLE" | "paused"}`` for running chain jobs that hold GPUs without using them
    (chain.sh's marks in RUN/chain/holding: ``<role>_<job>_<what>``)."""
    hold_dir = os.path.join(run_dir, "chain", "holding")
    out = {}
    for f in sorted(os.listdir(hold_dir)) if os.path.isdir(hold_dir) else []:
        parts = f.split("_")
        if len(parts) < 3:
            continue
        try:
            with open(os.path.join(hold_dir, f)) as fh:
                kind = "paused" if "paused" in fh.read() else "IDLE"
        except OSError:
            continue
        if out.get(parts[1]) != "IDLE":
            out[parts[1]] = kind
    return out


def queue_summary(spec, jobs, idle=None):
    """One short line per chain for alerts: what runs where and what waits, e.g.
    ``h100 4xH100: next 302973 (Priority) +14 queued``. ``idle`` (from ``idle_jobs``)
    marks running jobs that hold their GPUs unused."""
    idle = idle or {}
    roles = by_role(spec, jobs)
    parts = []
    for c in spec["chains"] + [{"role": g, "partition": None, "gpus": 0} for g in gap_roles(spec)]:
        js = roles.get(c["role"], [])
        size = f" {c['gpus']}x{GPU_KIND.get(c['partition'], c['partition'])}" if c["gpus"] else ""
        running = [j for j in js if j["state"] == "RUNNING"]
        waiting = [j for j in js if j["state"] != "RUNNING"]
        if running:
            now = ", ".join(f"{j['id']} running on {j['node']}" + ("" if c["gpus"] else f" ({j['gpus']} GPU)")
                            + (f" {idle[j['id']]}" if j["id"] in idle else "") for j in running)
        elif waiting:
            now = f"next {waiting[0]['id']} ({waiting[0]['reason']})"
            waiting = waiting[1:]
        else:
            now = "no jobs"
        parts.append(f"{c['role']}{size}: {now}" + (f" +{len(waiting)} queued" if waiting else ""))
    return "; ".join(parts)


def announce(run_dir, headline, slurm=None, send=None):
    """Send ``headline`` plus the queue summary as a ``status`` alert (stop.sh / start.sh)."""
    spec = read_launch(run_dir)
    slurm = slurm or Slurm()
    jobs = slurm.jobs(f"wwc-{spec['run_name']}-")
    idle = idle_jobs(run_dir)
    live = {j["id"] for j in jobs if j["state"] == "RUNNING"}
    message = f"{headline} squeue: {queue_summary(spec, jobs, idle)}"
    if any(v == "IDLE" and k in live for k, v in idle.items()):
        message += ". IDLE = holds its GPUs but runs nothing (see status.sh)"
    if send is None:
        from .notify import notify as send
    ok = send("status", message, run_dir, force=True)
    return message, ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tick", help="One keeper pass (cron runs this every 5 minutes).")
    t.add_argument("--run_dir", required=True)
    t.add_argument("--dry_run", action="store_true")
    w = sub.add_parser("write-launch", help="Write RUN/launch.json and RUN/launch.env (start.sh).")
    w.add_argument("--run_dir", required=True)
    w.add_argument("--run_name", required=True)
    w.add_argument("--repo", required=True)
    w.add_argument("--chains", required=True, help='e.g. "h100:dgxh100:4 v100:dgx:8 v100b:dgx:4"')
    w.add_argument("--depth", type=int, default=DEFAULTS["depth"])
    w.add_argument("--job_cap", type=int, default=None, help="Default: the QOS MaxJobsPU (4 on Rosie).")
    w.add_argument("--gap_partitions", default=",".join(DEFAULTS["gap_partitions"]))
    w.add_argument("--gap_slots", type=int, default=None, help="Default: job_cap minus the number of chains.")
    w.add_argument("--gap_max_gpus", type=int, default=DEFAULTS["gap_max_gpus"])
    w.add_argument("--cpus_per_gpu", type=int, default=DEFAULTS["cpus_per_gpu"])
    w.add_argument("--mem_per_gpu_gb", type=int, default=DEFAULTS["mem_per_gpu_gb"])
    w.add_argument("--extra_sbatch", default="", help="Extra sbatch options for every job (e.g. --exclude=...).")
    w.add_argument("--env", action="append", default=[], help="KEY=VALUE for the jobs (repeatable).")
    w.add_argument("--time_min", default=DEFAULTS["time_min"],
                   help="sbatch --time-min for chain and gap jobs ('' = off).")
    w.add_argument("--extra_partitions", default=",".join(DEFAULTS["extra_partitions"]),
                   help="Where the gap slot grabs GPUs beyond the chain targets, given back when others need them.")
    w.add_argument("--report_minutes", type=int, default=DEFAULTS["report_minutes"],
                   help="Send the ntfy GPU report every N minutes, on the clock (0 = off).")
    for name in ("install-cron", "uninstall-cron", "show"):
        s = sub.add_parser(name)
        s.add_argument("--run_dir", required=True)
    y = sub.add_parser("yield-check", help="Exit 0 and print why if this gap job should give its extra GPUs "
                                           "back now (chain.sh asks every minute); exit 1 to keep them.")
    y.add_argument("--run_dir", required=True)
    y.add_argument("--job", required=True)
    n = sub.add_parser("announce", help="Send an ntfy status alert with a squeue summary (stop.sh / start.sh).")
    n.add_argument("--run_dir", required=True)
    n.add_argument("headline")
    a = ap.parse_args(argv)
    run_dir = os.path.abspath(a.run_dir)
    if a.cmd == "write-launch":
        env = dict(kv.split("=", 1) for kv in a.env)
        repo = os.path.abspath(a.repo)
        spec = write_launch(run_dir, {
            "run_name": a.run_name, "run_dir": run_dir, "repo": repo,
            "sbatch": os.path.join(repo, "scripts/training/werewolf_continuous/main.sbatch"),
            "python_bin": os.path.dirname(sys.executable), "chains": parse_chains(a.chains), "depth": a.depth,
            "job_cap": a.job_cap if a.job_cap is not None else detect_job_cap(),
            "gap_partitions": a.gap_partitions, "gap_slots": a.gap_slots, "gap_max_gpus": a.gap_max_gpus,
            "cpus_per_gpu": a.cpus_per_gpu, "mem_per_gpu_gb": a.mem_per_gpu_gb, "extra_sbatch": a.extra_sbatch,
            "report_minutes": a.report_minutes, "time_min": a.time_min, "extra_partitions": a.extra_partitions,
            "env": env})
        print(f"chains: {', '.join(f'{c['role']}={c['gpus']}x{c['partition']}' for c in spec['chains'])}; "
              f"depth {spec['depth']}; job cap {spec['job_cap']}; gap slots {len(gap_roles(spec))} "
              f"on {','.join(spec['gap_partitions'])}")
    elif a.cmd == "tick":
        stamp = time.strftime("%F %T")
        summary = locked_tick(run_dir, dry_run=a.dry_run, log=lambda m: print(f"[{stamp}] {m}", flush=True))
        if not summary["actions"]:
            print(f"[{stamp}] nothing to do", flush=True)
    elif a.cmd in ("install-cron", "uninstall-cron"):
        spec = read_launch(run_dir)
        install_cron(run_dir, spec["repo"], remove=a.cmd == "uninstall-cron")
        print(("removed" if a.cmd == "uninstall-cron" else "installed") +
              f" the keeper cron line for {run_dir} on {os.uname().nodename.split('.')[0]}")
    elif a.cmd == "yield-check":
        spec = read_launch(run_dir)
        slurm = Slurm()
        jobs = slurm.jobs(f"wwc-{spec['run_name']}-")
        job = next((j for j in jobs if j["id"] == a.job), None)
        if job is None:
            sys.exit(1)
        reqs = slurm.waiting_requests(job["partition"])
        nodes = slurm.nodes(sorted({p for r in reqs for p in r["partitions"]} | {job["partition"]}))
        why = yield_reason(spec, by_role(spec, jobs), job, reqs, nodes)
        if why:
            print(why)
            sys.exit(0)
        sys.exit(1)
    elif a.cmd == "announce":
        message, ok = announce(run_dir, a.headline)
        print(f"[announce] {'sent' if ok else 'not sent (alerts off, or the same text within 30 min)'}: {message}")
    elif a.cmd == "show":
        age, host, actions = keeper_status(run_dir)
        print("never ran" if age is None else f"last tick {age / 60:.0f} min ago on {host}")
        for m in actions:
            print("  " + m)


if __name__ == "__main__":
    main()
