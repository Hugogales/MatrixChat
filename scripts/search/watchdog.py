#!/usr/bin/env python3
"""CPU-only watchdog that restores a dead month-search controller."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.search.search_state import (  # noqa: E402
    SearchPaths,
    acquire_directory_lock,
    append_jsonl,
    atomic_write_json,
    read_json,
    release_directory_lock,
    sbatch_search_exports,
    utc_now,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--lease-seconds", type=float, default=900.0)
    parser.add_argument("--max-runtime-hours", type=float, default=116.0)
    parser.add_argument("--controller-sbatch", default="scripts/search/controller.sbatch")
    parser.add_argument("--watchdog-sbatch", default="scripts/search/watchdog.sbatch")
    parser.add_argument("--no-self-chain", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def job_is_live(job_id) -> bool:
    if not job_id:
        return False
    result = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
        capture_output=True,
        text=True,
    )
    return any(
        state.strip() in {"RUNNING", "PENDING", "CONFIGURING"}
        for state in result.stdout.splitlines()
    )


def submit(path: str, search_dir: Path, dependency: str | None = None) -> str:
    command = ["sbatch"]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.extend(
        [
            f"--export={sbatch_search_exports(search_dir)}",
            path,
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip().split()[-1]


def check_controller(
    paths: SearchPaths, lease_seconds: float, controller_sbatch: str
) -> None:
    lease = read_json(paths.lease, {})
    fresh = (
        float(lease.get("heartbeat_unix", 0.0) or 0.0)
        and time.time() - float(lease["heartbeat_unix"]) <= lease_seconds
    )
    if fresh or job_is_live(lease.get("job_id")):
        return
    replacement = submit(controller_sbatch, paths.root)
    append_jsonl(
        paths.decisions,
        {
            "timestamp": utc_now(),
            "action": "watchdog_restart_controller",
            "replacement_job_id": replacement,
            "stale_lease": lease,
        },
    )


def main():
    args = parse_args()
    paths = SearchPaths(Path(args.search_dir).resolve())
    paths.initialize()
    lock = paths.root / "watchdog.lock"
    if not acquire_directory_lock(lock):
        raise SystemExit("another watchdog is active")
    started = time.time()
    try:
        while True:
            check_controller(paths, args.lease_seconds, args.controller_sbatch)
            if args.once:
                return
            if (time.time() - started) / 3600.0 >= args.max_runtime_hours:
                if not args.no_self_chain:
                    replacement = submit(
                        args.watchdog_sbatch,
                        paths.root,
                        dependency=f"afterany:{os.environ.get('SLURM_JOB_ID')}",
                    )
                    append_jsonl(
                        paths.decisions,
                        {
                            "timestamp": utc_now(),
                            "action": "watchdog_self_chain",
                            "replacement_job_id": replacement,
                        },
                    )
                return
            time.sleep(args.interval_seconds)
    finally:
        release_directory_lock(lock)


if __name__ == "__main__":
    main()
