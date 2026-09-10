#!/usr/bin/env python3
"""Auto-yield/reclaim daemon for the ``dgx`` (V100) and ``dgxh100`` (H100)
partitions -- an always-on implementation of
``.cursor/rules/yield-gpus-when-others-pending.mdc``, instead of a manual
action taken during hourly/six-hour checkups.

Every ``--interval-seconds`` tick:

1. Polls ``squeue`` on ``--partitions`` for jobs belonging to OTHER users.
2. For each foreign PENDING job not already being accommodated: figures out
   how many GPUs (or, for a near-full-node memory request, a whole node) it
   needs, holds our own pending jobs on that partition (so they can't
   backfill the hole first), and cancels enough of our own RUNNING jobs
   (lowest-sunk-cost-first, approximated by shortest elapsed runtime) to
   free the space. Also cancels any of our jobs whose ``Dependency`` chains
   back to a cancelled job, so an ``afterok`` eval doesn't refill the node.
3. For each foreign job we're already accommodating: once it transitions to
   RUNNING or leaves the queue entirely, releases our holds and attempts to
   resubmit whatever we cancelled for it (best-effort -- see "Reclaim
   mechanism" below).

Every action is appended to ``logs/gpu_yield_daemon/decisions.jsonl`` and
printed to stdout (captured in the sbatch ``.out`` file), so a human
reviewing later sees exactly what happened and why.

SAFETY RAILS -- read before passing ``--live``
-----------------------------------------------
- Defaults to ``--dry-run``: every action that WOULD be taken is logged, but
  no real ``scontrol``/``scancel``/``sbatch`` mutation happens. Only pass
  ``--live`` after reviewing a burn-in period of dry-run output.
- ``--protect-pattern`` (regex against job name) is NEVER selected as a
  cancellation victim, no matter how much GPU it would free. Default
  denylist covers anything that looks like a sealed final-run/paper-results
  job -- widen it (``--protect-pattern`) if other job names should also be
  untouchable.
- Never touches jobs outside ``--partitions`` or jobs not owned by
  ``--our-user``.
- "Lowest sunk cost" victim selection is only an approximation of the
  standing rule's richer judgment (elapsed runtime as a proxy) -- it does
  NOT know "this 60-example screen is almost finished" or similar nuance.
  Review the audit log periodically; don't assume it is always right.
- 1-second polling (``--interval-seconds 1``) is available but hits the
  shared ``slurmctld`` far harder than anything else in this project uses --
  the default is deliberately higher. Only force 1s if you've confirmed the
  cluster can tolerate it.

Reclaim mechanism (best-effort, not fully general)
---------------------------------------------------
At cancel time, snapshots the job's ``hyperparameters/<run_id>.json`` (the
authoritative record ``main.py`` itself writes of every parsed argparse
field) plus the sbatch script used (from ``scontrol show job``'s ``Command``
field). At reclaim time, uppercases every hyperparameters.json scalar key
into an env var (matching the ``--flag_name "$FLAG_NAME"`` convention every
current ``scripts/training/*.sbatch``/``scripts/search/*.sbatch`` script
uses), adds ``RUN_NAME``/``RESUME_FROM``/an incremented ``NUM_STEPS`` (per
``.cursor/rules/resume-needs-explicit-num-steps.mdc``), and resubmits the
same sbatch script. This exactly reproduces every hyperparameter that
happens to also be exposed as an env var in that particular script (true for
every training script in this project today) -- but is NOT guaranteed for
an arbitrary future script that doesn't follow this convention. Always
spot-check the first few real reclaims' new ``hyperparameters/<run_id>.json``
against the pre-cancel snapshot before trusting this unattended.

**This ad-hoc snapshot/reclaim path only applies to jobs NOT managed by the
job queue** (``scripts/ops/job_queue.py``) -- i.e. anything submitted
directly via ``sbatch``/Slurm and auto-detected here. A cancelled victim
that IS a queue-managed job (its Slurm job ID matches a queue entry's
``submitted_job_id``) instead goes back to ``queued`` status in the queue
itself (with ``RESUME_FROM``/``NUM_STEPS`` merged into its env), so it
re-enters the normal priority queue rather than jumping straight back in
via ``reclaim()`` -- see ``requeue_after_yield``.

Job-queue integration (2026-08-25)
-----------------------------------
Every tick, after the yield/reclaim logic above, this daemon also drives the
job queue (``scripts/ops/job_queue.py``): computes each partition's current
GPU usage from the SAME ``squeue`` snapshot (covering every job under
``--our-user`` regardless of whether it was submitted through the queue or
directly via ``sbatch`` -- "auto-detecting" manually-run jobs into the
budget accounting exactly as the user asked for), and if there's headroom
under ``--budget-dgx-gpus``/``--budget-h100-gpus``, submits the
highest-priority ``queued`` entries (in priority order, ties broken by
``added_at``) that fit in the remaining headroom. See ``fill_from_queue``.
This deliberately uses the snapshot captured at the START of the tick (before
this tick's own cancellations), so a GPU just vacated to accommodate a
foreign job is never immediately re-filled with a queued job in the same
tick -- it only becomes available for `fill_from_queue` on the NEXT tick,
once the foreign job's own need has been (re-)assessed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.search.search_state import (  # noqa: E402
    acquire_directory_lock,
    append_jsonl,
    atomic_write_json,
    read_json,
    release_directory_lock,
    utc_now,
)
from scripts.ops import job_queue  # noqa: E402

DEFAULT_PARTITIONS = ["dgx", "dgxh100"]
# V100 dgx nodes are ~489907M CfgTRES mem each; dgxh100 nodes are larger.
# A pending job requesting mem within this margin of a full node cannot be
# satisfied by freeing individual GPUs -- an entire node must be vacated.
NODE_MEM_THRESHOLD_MB = {"dgx": 420000.0, "dgxh100": 600000.0}
DEFAULT_PROTECT_PATTERN = r"final_test|paper_results|frozen|sealed"
DEFAULT_RESUME_STEP_INCREMENT = 2000
# 2 DGX/V100 nodes (8 GPUs/node, confirmed via `sinfo -p dgx -o "%N %G"`) and
# 4 H100 GPUs -- the user's standing GPU-usage cap (2026-08-10). The queue
# never auto-submits a job that would push total RUNNING usage past this,
# but does NOT itself enforce the cap on jobs submitted outside the queue.
DEFAULT_BUDGET_GPUS = {"dgx": 16, "dgxh100": 4}
# When the daemon submits a queued entry, pass partition-appropriate Slurm
# resource requests on the command line (overriding any mismatched #SBATCH
# defaults baked into the .sbatch script, e.g. evaluate_final_checkpoint.sbatch
# defaults to V100 but is routinely queued on dgxh100).
PARTITION_SBATCH_DEFAULTS = {
    "dgx": {"gres": "gpu:v100:1", "mem": "56G"},
    "dgxh100": {"gres": "gpu:h100:1", "mem": "80G"},
}
# hyperparameters.json keys that are never meaningful as a same-named env
# var override in these sbatch scripts (structural/derived fields, not a
# tunable an sbatch script exposes) -- skip rather than pass through.
_SKIP_HP_KEYS = {"run_name", "resume_from", "device"}


def _run(cmd: List[str], check: bool = True) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {result.stderr.strip()}")
    return result.stdout


def squeue_partition(partition: str) -> List[Dict[str, str]]:
    """All jobs (any user) currently in ``partition``, any state."""
    fmt = "%i|%u|%T|%j|%M|%N"
    out = _run(["squeue", "-h", "-p", partition, "-o", fmt], check=False)
    jobs = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 6:
            continue
        job_id, user, state, name, elapsed, nodelist = parts
        jobs.append({
            "job_id": job_id.strip(),
            "user": user.strip(),
            "state": state.strip(),
            "name": name.strip(),
            "elapsed": elapsed.strip(),
            "nodelist": nodelist.strip(),
            "partition": partition,
        })
    return jobs


def scontrol_show_job(job_id: str) -> Dict[str, str]:
    out = _run(["scontrol", "show", "job", "-o", job_id], check=False)
    fields: Dict[str, str] = {}
    for token in out.split():
        if "=" in token:
            key, _, value = token.partition("=")
            fields[key] = value
    return fields


def _elapsed_seconds(elapsed: str) -> float:
    """Parse squeue's %M elapsed format (e.g. "1:23:45", "3-04:15:00")."""
    days = 0
    rest = elapsed
    if "-" in elapsed:
        days_str, rest = elapsed.split("-", 1)
        days = int(days_str)
    try:
        parts = [int(p) for p in rest.split(":")]
    except ValueError:
        return 0.0
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts[-3:]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def requested_gpus(job_id: str) -> int:
    fields = scontrol_show_job(job_id)
    match = re.search(r"gres/gpu=(\d+)", fields.get("ReqTRES", "") or fields.get("TRES", ""))
    if match:
        return int(match.group(1))
    # Fallback: TresPerNode can be "gpu:1" or "gpu:v100:1" (type-qualified).
    match = re.search(r"gpu(?::\w+)?:(\d+)", fields.get("TresPerNode", ""))
    return int(match.group(1)) if match else 1


def requested_mem_mb(job_id: str) -> float:
    fields = scontrol_show_job(job_id)
    match = re.search(r"mem=(\d+)([MG]?)", fields.get("ReqTRES", "") or fields.get("TRES", ""))
    if not match:
        return 0.0
    value = float(match.group(1))
    return value * 1024 if match.group(2) == "G" else value


def choose_gpu_victims(
    running: List[Dict[str, str]], need_gpus: int, protect_re: re.Pattern
) -> List[Dict[str, str]]:
    """Lowest-sunk-cost-first (shortest elapsed runtime as a proxy for least
    sunk cost), skipping anything matching the protect pattern. See the
    module docstring's "SAFETY RAILS" section for the limits of this
    approximation."""
    candidates = [j for j in running if not protect_re.search(j["name"])]
    candidates.sort(key=lambda j: _elapsed_seconds(j["elapsed"]))
    chosen: List[Dict[str, str]] = []
    freed = 0
    for job in candidates:
        if freed >= need_gpus:
            break
        chosen.append(job)
        freed += requested_gpus(job["job_id"])
    return chosen


def choose_node_to_vacate(
    running: List[Dict[str, str]], protect_re: re.Pattern
) -> Optional[str]:
    """Pick the node (among ones hosting any of our running jobs) whose jobs
    have the least total sunk cost, to vacate entirely."""
    by_node: Dict[str, float] = {}
    for job in running:
        if protect_re.search(job["name"]) or not job["nodelist"]:
            continue
        by_node[job["nodelist"]] = by_node.get(job["nodelist"], 0.0) + _elapsed_seconds(job["elapsed"])
    if not by_node:
        return None
    return min(by_node, key=by_node.get)


def find_dependents(all_our_pending: List[Dict[str, str]], job_id: str) -> List[str]:
    dependents = []
    for job in all_our_pending:
        fields = scontrol_show_job(job["job_id"])
        if job_id in fields.get("Dependency", ""):
            dependents.append(job["job_id"])
    return dependents


def snapshot_for_resume(job_id: str, root: Path) -> Optional[Dict]:
    fields = scontrol_show_job(job_id)
    command = fields.get("Command", "")
    sbatch_script = command.split(" ")[0].split(",")[0] if command else None
    # Convention (confirmed across every current sbatch script): the job's
    # own name IS the run_id main.py uses for hyperparameters/<id>.json,
    # logs/<id>/, and checkpoints/<id>/.
    run_id = fields.get("JobName") or fields.get("Name")
    if not run_id or not sbatch_script:
        return None
    hp_path = root / "hyperparameters" / f"{run_id}.json"
    hyperparameters = read_json(hp_path)
    if hyperparameters is None:
        return None
    metrics_path = root / "logs" / run_id / "metrics.jsonl"
    last_step = 0
    if metrics_path.is_file():
        try:
            lines = metrics_path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                if not line.strip():
                    continue
                last_step = int(json.loads(line).get("step", 0))
                break
        except (json.JSONDecodeError, ValueError):
            last_step = 0
    return {
        "job_id": job_id,
        "run_id": run_id,
        "sbatch_script": sbatch_script,
        "hyperparameters": hyperparameters,
        "last_known_step": last_step,
    }


def build_resume_env(snapshot: Dict, resume_step_increment: int) -> Dict[str, str]:
    env = {}
    for key, value in snapshot["hyperparameters"].items():
        if key in _SKIP_HP_KEYS or isinstance(value, (dict, list)):
            continue
        env[key.upper()] = str(value)
    env["RUN_NAME"] = snapshot["run_id"]
    env["RESUME_FROM"] = f"checkpoints/{snapshot['run_id']}"
    env["NUM_STEPS"] = str(snapshot["last_known_step"] + resume_step_increment)
    return env


class Daemon:
    def __init__(self, args):
        self.args = args
        self.our_user = args.our_user
        self.partitions = args.partitions
        self.protect_re = re.compile(args.protect_pattern, re.IGNORECASE)
        self.root = Path(args.project_root).resolve()
        self.state_path = self.root / args.state_file
        self.decisions_path = self.root / args.audit_log
        self.state = read_json(self.state_path, {"held_job_ids": [], "vacated": {}})
        self.queue_path = self.root / args.queue_file
        self.budget = {"dgx": args.budget_dgx_gpus, "dgxh100": args.budget_h100_gpus}

    def log(self, action: str, **payload) -> None:
        entry = {"timestamp": utc_now(), "action": action, "dry_run": not self.args.live, **payload}
        print(f"[gpu_yield_daemon] {json.dumps(entry, sort_keys=True)}", flush=True)
        append_jsonl(self.decisions_path, entry)

    def save_state(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def _do(self, cmd: List[str]) -> None:
        if not self.args.live:
            return
        subprocess.run(cmd, capture_output=True, text=True)

    def hold(self, job_id: str) -> None:
        self.log("hold", job_id=job_id)
        self._do(["scontrol", "hold", job_id])
        if job_id not in self.state["held_job_ids"]:
            self.state["held_job_ids"].append(job_id)

    def release(self, job_id: str) -> None:
        self.log("release", job_id=job_id)
        self._do(["scontrol", "release", job_id])
        if job_id in self.state["held_job_ids"]:
            self.state["held_job_ids"].remove(job_id)

    def cancel(self, job_id: str, reason: str) -> None:
        self.log("cancel", job_id=job_id, reason=reason)
        self._do(["scancel", job_id])

    def resubmit(self, snapshot: Dict) -> Optional[str]:
        env = build_resume_env(snapshot, self.args.resume_step_increment)
        self.log(
            "resubmit",
            run_id=snapshot["run_id"],
            sbatch_script=snapshot["sbatch_script"],
            resume_from=env["RESUME_FROM"],
            num_steps=env["NUM_STEPS"],
        )
        if not self.args.live:
            return None
        # NEVER inline these into `--export=ALL,K=V,...` -- Slurm's --export
        # parser splits on every comma regardless of quoting, so any
        # comma-containing value (e.g. a resumed dataset_weights hyperparameter)
        # would be silently truncated/corrupted (see
        # .cursor/rules/sbatch-env-overrides.mdc). Pass overrides via this
        # subprocess call's own `env=` instead, with a bare `--export=ALL` so
        # Slurm forwards that (isolated, one-shot) environment verbatim.
        result = subprocess.run(
            ["sbatch", "--export=ALL", snapshot["sbatch_script"]],
            capture_output=True, text=True, env={**os.environ, **env},
        )
        if result.returncode != 0:
            self.log("resubmit_failed", run_id=snapshot["run_id"], error=result.stderr.strip())
            return None
        new_job_id = result.stdout.strip().split()[-1]
        self.log("resubmit_ok", run_id=snapshot["run_id"], new_job_id=new_job_id)
        return new_job_id

    def accommodate(self, foreign_job: Dict[str, str], partition_jobs: List[Dict[str, str]]) -> None:
        job_id = foreign_job["job_id"]
        partition = foreign_job["partition"]
        need_gpus = requested_gpus(job_id)
        need_mem = requested_mem_mb(job_id)
        our_running = [j for j in partition_jobs if j["user"] == self.our_user and j["state"] == "RUNNING"]
        our_pending = [j for j in partition_jobs if j["user"] == self.our_user and j["state"] == "PENDING"]

        for job in our_pending:
            self.hold(job["job_id"])

        vacated_node = None
        cancelled_snapshots: List[Dict] = []
        if need_mem >= NODE_MEM_THRESHOLD_MB.get(partition, float("inf")):
            vacated_node = choose_node_to_vacate(our_running, self.protect_re)
            victims = [j for j in our_running if j["nodelist"] == vacated_node] if vacated_node else []
        else:
            victims = choose_gpu_victims(our_running, need_gpus, self.protect_re)

        for job in victims:
            queue_entry = self.find_queue_entry_by_job_id(job["job_id"])
            if queue_entry is not None:
                # Queue-managed: re-enter the priority queue instead of the
                # ad-hoc immediate-resubmit path below, so priority/pausing
                # still applies once headroom reopens (see module docstring).
                self.requeue_after_yield(queue_entry)
            else:
                snap = snapshot_for_resume(job["job_id"], self.root)
                if snap:
                    cancelled_snapshots.append(snap)
            self.cancel(job["job_id"], reason=f"yielding to foreign job {job_id} on {partition}")
            for dep in find_dependents(our_pending, job["job_id"]):
                self.cancel(dep, reason=f"dependent of yielded job {job['job_id']}")

        self.state["vacated"][job_id] = {
            "partition": partition,
            "detected_at": utc_now(),
            "held_pending": [j["job_id"] for j in our_pending],
            "vacated_node": vacated_node,
            "cancelled_snapshots": cancelled_snapshots,
        }
        self.save_state()

    def reclaim(self, foreign_job_id: str) -> None:
        record = self.state["vacated"].pop(foreign_job_id)
        # Don't release a hold still needed by another foreign job we're
        # concurrently accommodating on the same partition.
        still_needed = {
            job_id
            for other in self.state["vacated"].values()
            for job_id in other["held_pending"]
        }
        for job_id in record["held_pending"]:
            if job_id not in still_needed:
                self.release(job_id)
        for snap in record["cancelled_snapshots"]:
            self.resubmit(snap)
        self.save_state()

    # --- job-queue integration (2026-08-25) --------------------------------

    def find_queue_entry_by_job_id(self, job_id: str) -> Optional[Dict]:
        for entry in job_queue.load_queue(self.queue_path):
            if entry.get("submitted_job_id") == job_id:
                return entry
        return None

    def _last_known_step(self, run_id: str) -> int:
        metrics_path = self.root / "logs" / run_id / "metrics.jsonl"
        if not metrics_path.is_file():
            return 0
        try:
            for line in reversed(metrics_path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                return int(json.loads(line).get("step", 0))
        except (json.JSONDecodeError, ValueError):
            pass
        return 0

    def requeue_after_yield(self, target_entry: Dict) -> None:
        entry_id = target_entry["id"]
        is_infra = (
            "worker_pool" in target_entry.get("sbatch_script", "")
            or "workers" in target_entry.get("name", "").lower()
        )

        def _mutate(entries):
            entry = job_queue.find_entry(entries, entry_id)
            if entry is None:
                return None
            if not is_infra:
                last_step = self._last_known_step(entry["name"])
                env = dict(entry["env"])
                env["RUN_NAME"] = entry["name"]
                env["RESUME_FROM"] = f"checkpoints/{entry['name']}"
                env["NUM_STEPS"] = str(last_step + self.args.resume_step_increment)
                entry["env"] = env
                self.log(
                    "queue_requeue_after_yield", entry_id=entry_id,
                    resume_from=env["RESUME_FROM"], num_steps=env["NUM_STEPS"],
                )
            else:
                self.log("queue_requeue_after_yield", entry_id=entry_id, infra=True)
            entry["status"] = job_queue.STATUS_QUEUED
            entry["submitted_job_id"] = None
            entry["updated_at"] = utc_now()
            return entries

        job_queue._with_lock(self.queue_path, _mutate)

    def sync_queue_with_slurm(self, all_jobs: Dict[str, List[Dict[str, str]]]) -> None:
        """Auto-detect a queue-managed job finishing (leaving squeue on its
        own, not via our own yield-cancel) and mark its entry completed."""
        by_job_id = {j["job_id"]: j for jobs in all_jobs.values() for j in jobs}

        def _mutate(entries):
            changed = False
            for entry in entries:
                if entry["status"] != job_queue.STATUS_SUBMITTED:
                    continue
                job_id = entry.get("submitted_job_id")
                if job_id and job_id not in by_job_id:
                    entry["status"] = job_queue.STATUS_COMPLETED
                    entry["updated_at"] = utc_now()
                    self.log("queue_entry_finished", entry_id=entry["id"], job_id=job_id)
                    changed = True
            return entries if changed else None

        job_queue._with_lock(self.queue_path, _mutate)

    def submit_queue_entry(self, entry: Dict, headroom_partition: str) -> bool:
        self.log(
            "queue_submit", entry_id=entry["id"], name=entry["name"],
            partition=headroom_partition, gpu_count=entry["gpu_count"],
        )
        if not self.args.live:
            return False
        # Same comma-splitting hazard as `resubmit` -- see the comment there
        # and .cursor/rules/sbatch-env-overrides.mdc. A queued entry's env is
        # user-authored and may legitimately contain comma-bearing values
        # (e.g. DATASET_WEIGHTS="ami=0.5,meld=0.15,werewolf=0.35"), so this
        # MUST go through `env=`, never an inline `--export=ALL,K=V,...` string.
        part_defaults = PARTITION_SBATCH_DEFAULTS.get(entry["partition"], {})
        gres = entry.get("gres") or part_defaults.get("gres")
        mem = entry.get("mem") or part_defaults.get("mem")
        cmd = ["sbatch", "--export=ALL", f"--partition={entry['partition']}"]
        if gres:
            cmd.append(f"--gres={gres}")
        if mem:
            cmd.append(f"--mem={mem}")
        if entry.get("nodelist"):
            cmd.append(f"--nodelist={entry['nodelist']}")
        if entry.get("ntasks"):
            cmd.append(f"--ntasks={entry['ntasks']}")
        cmd.append(entry["sbatch_script"])
        result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **entry["env"]})
        if result.returncode != 0:
            self.log("queue_submit_failed", entry_id=entry["id"], error=result.stderr.strip())
            return False
        new_job_id = result.stdout.strip().split()[-1]
        entry["status"] = job_queue.STATUS_SUBMITTED
        entry["submitted_job_id"] = new_job_id
        entry["updated_at"] = utc_now()
        self.log("queue_submit_ok", entry_id=entry["id"], job_id=new_job_id)
        return True

    def fill_from_queue(self, all_jobs: Dict[str, List[Dict[str, str]]]) -> None:
        """For each partition with GPU headroom under the fixed budget,
        submit the highest-priority fitting `queued` entries. Uses the
        `all_jobs` snapshot taken at the START of this tick (before any of
        this tick's own yield-cancellations), so a GPU just vacated for a
        foreign job is never immediately re-filled in the same tick."""

        def _mutate(entries):
            changed = False
            for partition in self.partitions:
                budget = self.budget.get(partition)
                if budget is None:
                    continue
                jobs = all_jobs.get(partition, [])
                used = sum(
                    requested_gpus(j["job_id"]) for j in jobs
                    if j["user"] == self.our_user and j["state"] == "RUNNING"
                )
                headroom = budget - used
                if headroom <= 0:
                    continue
                candidates = [e for e in job_queue.sorted_for_processing(entries) if e["partition"] == partition]
                for entry in candidates:
                    if entry["gpu_count"] > headroom:
                        continue
                    if self.submit_queue_entry(entry, partition):
                        headroom -= entry["gpu_count"]
                        changed = True
                    if headroom <= 0:
                        break
            return entries if changed else None

        job_queue._with_lock(self.queue_path, _mutate)

    def tick(self) -> None:
        all_jobs: Dict[str, List[Dict[str, str]]] = {p: squeue_partition(p) for p in self.partitions}

        for partition, jobs in all_jobs.items():
            foreign_pending = [j for j in jobs if j["user"] != self.our_user and j["state"] == "PENDING"]
            for fj in foreign_pending:
                if fj["job_id"] not in self.state["vacated"]:
                    self.accommodate(fj, jobs)

        by_job_id = {j["job_id"]: j for jobs in all_jobs.values() for j in jobs}
        for foreign_job_id in list(self.state["vacated"].keys()):
            current = by_job_id.get(foreign_job_id)
            if current is None or current["state"] != "PENDING":
                self.reclaim(foreign_job_id)

        self.sync_queue_with_slurm(all_jobs)
        self.fill_from_queue(all_jobs)

    def run(self) -> None:
        started = time.time()
        while True:
            try:
                self.tick()
            except RuntimeError as exc:
                self.log("tick_error", error=str(exc))
            if self.args.once:
                return
            if (time.time() - started) / 3600.0 >= self.args.max_runtime_hours:
                self.self_chain()
                return
            time.sleep(self.args.interval_seconds)

    def self_chain(self) -> None:
        export_str = f"ALL,LIVE={'1' if self.args.live else '0'},INTERVAL_SECONDS={self.args.interval_seconds}"
        result = subprocess.run(
            ["sbatch", f"--export={export_str}",
             f"--dependency=afterany:{os.environ.get('SLURM_JOB_ID', '')}",
             self.args.self_sbatch],
            capture_output=True, text=True,
        )
        replacement = result.stdout.strip().split()[-1] if result.returncode == 0 else None
        self.log("self_chain", replacement_job_id=replacement, error=result.stderr.strip() if not replacement else None)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partitions", nargs="+", default=DEFAULT_PARTITIONS)
    parser.add_argument("--our-user", default=os.environ.get("USER", ""))
    parser.add_argument("--project-root", default=str(_ROOT))
    parser.add_argument("--interval-seconds", type=float, default=15.0,
                         help="Poll frequency. 1s is supported but hits slurmctld hard -- see module docstring.")
    parser.add_argument("--protect-pattern", default=DEFAULT_PROTECT_PATTERN)
    parser.add_argument("--resume-step-increment", type=int, default=DEFAULT_RESUME_STEP_INCREMENT)
    parser.add_argument("--state-file", default="logs/gpu_yield_daemon/state.json")
    parser.add_argument("--audit-log", default="logs/gpu_yield_daemon/decisions.jsonl")
    parser.add_argument("--queue-file", default="logs/job_queue/queue.json",
                         help="Path (relative to --project-root) to the job_queue.py-managed queue file.")
    parser.add_argument("--budget-dgx-gpus", type=int, default=DEFAULT_BUDGET_GPUS["dgx"],
                         help="Max concurrent GPUs this daemon will use on the dgx (V100) partition when filling from the queue.")
    parser.add_argument("--budget-h100-gpus", type=int, default=DEFAULT_BUDGET_GPUS["dgxh100"],
                         help="Max concurrent GPUs this daemon will use on the dgxh100 partition when filling from the queue.")
    parser.add_argument("--live", action="store_true", help="Actually mutate Slurm state. Default is dry-run (log only).")
    parser.add_argument("--max-runtime-hours", type=float, default=116.0)
    parser.add_argument("--self-sbatch", default="scripts/ops/gpu_yield_daemon.sbatch")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.our_user:
        raise SystemExit("Could not determine --our-user (set $USER or pass explicitly)")
    lock_path = Path(args.project_root).resolve() / Path(args.state_file).parent / "daemon.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not acquire_directory_lock(lock_path):
        raise SystemExit("another gpu_yield_daemon instance is active (lock held)")
    try:
        Daemon(args).run()
    finally:
        release_directory_lock(lock_path)


if __name__ == "__main__":
    main()
