#!/usr/bin/env python3
"""Manipulable priority queue of not-yet-submitted Slurm job specs for the
``dgx``/``dgxh100`` partitions -- a CLI the user can keep adding to, and the
agent can reorder/pause/cancel as priorities change, on top of
``gpu_yield_daemon.py``'s existing yield/reclaim logic.

This module owns ONLY the queue data (``logs/job_queue/queue.json``); actually
launching queued entries when GPU budget allows, and tracking their Slurm
job IDs, is done by ``gpu_yield_daemon.py``'s ``fill_from_queue`` step (kept
in the same daemon so queue-fill and yield-to-foreign-jobs share one
consistent, single-threaded view of current Slurm usage -- running them as
separate processes would race on which one claims freed-up GPUs first).

Queue entry lifecycle::

    queued --(daemon submits via sbatch, headroom available)--> submitted
    queued --(pause)--> held --(resume)--> queued
    submitted --(daemon notices job left squeue)--> completed
    submitted --(cancel)--> cancelled (scancel'd)
    queued/held --(remove)--> deleted from the queue entirely
    submitted --(yielded to a foreign job by gpu_yield_daemon)--> queued
        (re-queued with RESUME_FROM/NUM_STEPS added to env, same priority,
        so it naturally gets resubmitted once headroom reopens)

Entries are plain dicts in a JSON list, sorted for processing by
``(priority ascending, added_at ascending)`` -- lower priority number =
sooner. No external dependencies beyond the stdlib plus this project's own
``scripts/search/search_state`` file-locking helpers (same pattern already
used by the Optuna ticket queue and ``gpu_yield_daemon.py``).

CLI examples::

    # Add a job (queued, not yet submitted -- the daemon submits it once
    # GPU budget allows).
    python scripts/ops/job_queue.py add --name my_run \\
        --sbatch scripts/training/train_dynstate_ablation.sbatch \\
        --gpu-count 1 --partition dgx --priority 50 \\
        --env RUN_NAME=my_run TURN_REWARD_MODE=balanced_ce \\
        --notes "why this job exists"

    python scripts/ops/job_queue.py list
    python scripts/ops/job_queue.py show q_a1b2c3d4
    python scripts/ops/job_queue.py set-priority q_a1b2c3d4 --priority 5
    python scripts/ops/job_queue.py bump q_a1b2c3d4          # move to front
    python scripts/ops/job_queue.py pause q_a1b2c3d4
    python scripts/ops/job_queue.py resume q_a1b2c3d4
    python scripts/ops/job_queue.py cancel q_a1b2c3d4        # scancel if submitted
    python scripts/ops/job_queue.py remove q_a1b2c3d4        # queued/held only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.search.search_state import (  # noqa: E402
    acquire_directory_lock,
    atomic_write_json,
    read_json,
    release_directory_lock,
    utc_now,
)

DEFAULT_QUEUE_PATH = _ROOT / "logs" / "job_queue" / "queue.json"

STATUS_QUEUED = "queued"
STATUS_HELD = "held"
STATUS_SUBMITTED = "submitted"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"
_MUTABLE_STATUSES = {STATUS_QUEUED, STATUS_HELD}


def new_id() -> str:
    return f"q_{uuid.uuid4().hex[:8]}"


def load_queue(path: Path = DEFAULT_QUEUE_PATH) -> List[Dict]:
    return read_json(path, {"entries": []}).get("entries", [])


def save_queue(entries: List[Dict], path: Path = DEFAULT_QUEUE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"entries": entries})


def _with_lock(path: Path, fn):
    """Run ``fn(entries) -> entries_or_None`` under the queue's directory
    lock; if ``fn`` returns None, no write happens (used for read-only ops
    like ``show``/``list`` that still want a consistent snapshot)."""
    lock_path = path.parent / "queue.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not acquire_directory_lock(lock_path):
        raise SystemExit("another job_queue writer is active (lock held); retry shortly")
    try:
        entries = load_queue(path)
        result = fn(entries)
        if result is not None:
            save_queue(result, path)
        return result if result is not None else entries
    finally:
        release_directory_lock(lock_path)


def find_entry(entries: List[Dict], entry_id: str) -> Optional[Dict]:
    for entry in entries:
        if entry["id"] == entry_id:
            return entry
    return None


def sorted_for_processing(entries: List[Dict]) -> List[Dict]:
    """Queued entries only, in the order the daemon should try to submit
    them: lower ``priority`` first, ties broken by earlier ``added_at``."""
    queued = [e for e in entries if e["status"] == STATUS_QUEUED]
    return sorted(queued, key=lambda e: (e["priority"], e["added_at"]))


def add_job(
    entries: List[Dict],
    name: str,
    sbatch_script: str,
    env: Dict[str, str],
    gpu_count: int,
    partition: Optional[str],
    priority: int,
    notes: str,
    gres: Optional[str] = None,
    mem: Optional[str] = None,
    nodelist: Optional[str] = None,
    ntasks: Optional[int] = None,
) -> Dict:
    entry = {
        "id": new_id(),
        "name": name,
        "sbatch_script": sbatch_script,
        "env": env,
        "gpu_count": gpu_count,
        "partition": partition,
        "priority": priority,
        "status": STATUS_QUEUED,
        "added_at": utc_now(),
        "updated_at": utc_now(),
        "submitted_job_id": None,
        "notes": notes,
    }
    if gres:
        entry["gres"] = gres
    if mem:
        entry["mem"] = mem
    if nodelist:
        entry["nodelist"] = nodelist
    if ntasks is not None:
        entry["ntasks"] = ntasks
    entries.append(entry)
    return entry


def _parse_env_pairs(pairs: List[str]) -> Dict[str, str]:
    env = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--env entries must be KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        env[key] = value
    return env


def cmd_add(args) -> None:
    env = _parse_env_pairs(args.env)

    def _mutate(entries):
        entry = add_job(
            entries, args.name, args.sbatch, env, args.gpu_count,
            args.partition, args.priority, args.notes or "",
            gres=args.gres, mem=args.mem, nodelist=args.nodelist, ntasks=args.ntasks,
        )
        print(f"queued {entry['id']} ({entry['name']}, priority={entry['priority']})")
        return entries

    _with_lock(args.queue_path, _mutate)


def _print_entry(entry: Dict) -> None:
    print(f"{entry['id']}  [{entry['status']:>9}]  prio={entry['priority']:<5}  {entry['name']}")
    print(f"    sbatch={entry['sbatch_script']}  gpus={entry['gpu_count']}  partition={entry['partition']}")
    if entry.get("submitted_job_id"):
        print(f"    submitted_job_id={entry['submitted_job_id']}")
    if entry.get("env"):
        print(f"    env={entry['env']}")
    if entry.get("notes"):
        print(f"    notes: {entry['notes']}")
    print(f"    added_at={entry['added_at']}  updated_at={entry['updated_at']}")


def cmd_list(args) -> None:
    entries = load_queue(args.queue_path)
    if args.status:
        entries = [e for e in entries if e["status"] == args.status]
    order = {STATUS_QUEUED: 0, STATUS_HELD: 1, STATUS_SUBMITTED: 2, STATUS_COMPLETED: 3, STATUS_CANCELLED: 4}
    entries.sort(key=lambda e: (order.get(e["status"], 9), e["priority"], e["added_at"]))
    if not entries:
        print("(queue is empty)")
        return
    for entry in entries:
        _print_entry(entry)


def cmd_show(args) -> None:
    entries = load_queue(args.queue_path)
    entry = find_entry(entries, args.id)
    if entry is None:
        raise SystemExit(f"no such queue entry: {args.id}")
    _print_entry(entry)


def cmd_set_priority(args) -> None:
    def _mutate(entries):
        entry = find_entry(entries, args.id)
        if entry is None:
            raise SystemExit(f"no such queue entry: {args.id}")
        entry["priority"] = args.priority
        entry["updated_at"] = utc_now()
        print(f"{entry['id']} priority -> {args.priority}")
        return entries

    _with_lock(args.queue_path, _mutate)


def cmd_bump(args) -> None:
    def _mutate(entries):
        entry = find_entry(entries, args.id)
        if entry is None:
            raise SystemExit(f"no such queue entry: {args.id}")
        min_priority = min((e["priority"] for e in entries if e["id"] != args.id), default=0)
        entry["priority"] = min_priority - 1
        entry["updated_at"] = utc_now()
        print(f"{entry['id']} bumped to front, priority -> {entry['priority']}")
        return entries

    _with_lock(args.queue_path, _mutate)


def cmd_pause(args) -> None:
    def _mutate(entries):
        entry = find_entry(entries, args.id)
        if entry is None:
            raise SystemExit(f"no such queue entry: {args.id}")
        if entry["status"] != STATUS_QUEUED:
            raise SystemExit(f"can only pause a {STATUS_QUEUED} entry (currently {entry['status']})")
        entry["status"] = STATUS_HELD
        entry["updated_at"] = utc_now()
        print(f"{entry['id']} paused (held)")
        return entries

    _with_lock(args.queue_path, _mutate)


def cmd_resume(args) -> None:
    def _mutate(entries):
        entry = find_entry(entries, args.id)
        if entry is None:
            raise SystemExit(f"no such queue entry: {args.id}")
        if entry["status"] != STATUS_HELD:
            raise SystemExit(f"can only resume a {STATUS_HELD} entry (currently {entry['status']})")
        entry["status"] = STATUS_QUEUED
        entry["updated_at"] = utc_now()
        print(f"{entry['id']} resumed (queued)")
        return entries

    _with_lock(args.queue_path, _mutate)


def cmd_remove(args) -> None:
    def _mutate(entries):
        entry = find_entry(entries, args.id)
        if entry is None:
            raise SystemExit(f"no such queue entry: {args.id}")
        if entry["status"] not in _MUTABLE_STATUSES:
            raise SystemExit(
                f"cannot remove a {entry['status']} entry -- use `cancel` if it's submitted"
            )
        remaining = [e for e in entries if e["id"] != args.id]
        print(f"{entry['id']} removed")
        return remaining

    _with_lock(args.queue_path, _mutate)


def cmd_cancel(args) -> None:
    def _mutate(entries):
        entry = find_entry(entries, args.id)
        if entry is None:
            raise SystemExit(f"no such queue entry: {args.id}")
        if entry["status"] == STATUS_SUBMITTED and entry.get("submitted_job_id"):
            subprocess.run(["scancel", entry["submitted_job_id"]], capture_output=True, text=True)
            print(f"scancel'd {entry['submitted_job_id']}")
        entry["status"] = STATUS_CANCELLED
        entry["updated_at"] = utc_now()
        print(f"{entry['id']} cancelled")
        return entries

    _with_lock(args.queue_path, _mutate)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue-path", type=Path, default=DEFAULT_QUEUE_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a new queued job spec")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--sbatch", required=True, help="Path to the .sbatch script to submit")
    p_add.add_argument("--env", nargs="*", default=[], help="KEY=VALUE pairs exported at submit time")
    p_add.add_argument("--gpu-count", type=int, default=1)
    p_add.add_argument(
        "--partition", choices=["dgx", "dgxh100"], required=True,
        help="Required (not optional) so gpu_yield_daemon's per-partition GPU "
             "budget accounting stays accurate -- the daemon passes this as an "
             "explicit `sbatch --partition=` override, so it's guaranteed to "
             "land where accounted, regardless of the .sbatch script's own "
             "#SBATCH --partition default.",
    )
    p_add.add_argument("--priority", type=int, default=100)
    p_add.add_argument("--notes", default="")
    p_add.add_argument("--gres", default=None, help="Override --gres at submit time (e.g. gpu:v100:7)")
    p_add.add_argument("--mem", default=None, help="Override --mem at submit time (e.g. 350G)")
    p_add.add_argument("--nodelist", default=None, help="Pin submission to a node (e.g. dh-dgx1-2)")
    p_add.add_argument("--ntasks", type=int, default=None, help="Override --ntasks at submit time")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List queue entries")
    p_list.add_argument("--status", choices=[STATUS_QUEUED, STATUS_HELD, STATUS_SUBMITTED, STATUS_COMPLETED, STATUS_CANCELLED])
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show one entry")
    p_show.add_argument("id")
    p_show.set_defaults(func=cmd_show)

    p_prio = sub.add_parser("set-priority", help="Change an entry's priority")
    p_prio.add_argument("id")
    p_prio.add_argument("--priority", type=int, required=True)
    p_prio.set_defaults(func=cmd_set_priority)

    p_bump = sub.add_parser("bump", help="Move an entry to the front of the queue")
    p_bump.add_argument("id")
    p_bump.set_defaults(func=cmd_bump)

    p_pause = sub.add_parser("pause", help="Hold a queued entry (skip it until resumed)")
    p_pause.add_argument("id")
    p_pause.set_defaults(func=cmd_pause)

    p_resume = sub.add_parser("resume", help="Un-hold a paused entry")
    p_resume.add_argument("id")
    p_resume.set_defaults(func=cmd_resume)

    p_remove = sub.add_parser("remove", help="Delete a queued/held entry")
    p_remove.add_argument("id")
    p_remove.set_defaults(func=cmd_remove)

    p_cancel = sub.add_parser("cancel", help="Cancel an entry (scancel's it if already submitted)")
    p_cancel.add_argument("id")
    p_cancel.set_defaults(func=cmd_cancel)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
