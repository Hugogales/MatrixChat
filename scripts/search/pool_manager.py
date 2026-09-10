#!/usr/bin/env python3
"""Launch five one-GPU workers inside one H100 allocation and self-chain."""

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

from scripts.search.search_state import SearchPaths, append_jsonl, utc_now  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-runtime-hours", type=float, default=116.0)
    parser.add_argument("--drain-hours", type=float, default=3.0)
    parser.add_argument("--worker-sbatch", default="scripts/search/worker_pool.sbatch")
    parser.add_argument("--container", default="/data/containers/msoe-tf2x.sif")
    parser.add_argument("--gpu-type", default="h100", help="Typed Slurm GRES gpu name, e.g. h100 or v100.")
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--no-self-chain", action="store_true")
    return parser.parse_args()


def submit_replacement(args) -> str | None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    result = subprocess.run(
        [
            "sbatch",
            f"--dependency=afterany:{job_id}",
            f"--export=ALL,SEARCH_DIR={Path(args.search_dir).resolve()}",
            args.worker_sbatch,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip().split()[-1]


def main():
    args = parse_args()
    paths = SearchPaths(Path(args.search_dir).resolve())
    paths.initialize()
    shutdown = paths.root / "shutdown_requested"
    shutdown.unlink(missing_ok=True)
    command = [
        "srun",
        "--exclusive",
        f"--ntasks={args.workers}",
        f"--tres-per-task=cpu={args.cpus_per_worker},gres/gpu:{args.gpu_type}:1",
        "singularity",
        "exec",
        "--nv",
        "-B",
        "/data:/data",
        args.container,
        "python",
        str(_ROOT / "scripts" / "search" / "worker.py"),
        "--search-dir",
        str(paths.root),
    ]
    workers = subprocess.Popen(command, cwd=_ROOT)
    started = time.time()
    while workers.poll() is None:
        elapsed_hours = (time.time() - started) / 3600.0
        if elapsed_hours >= args.max_runtime_hours:
            shutdown.touch()
            append_jsonl(
                paths.decisions,
                {
                    "timestamp": utc_now(),
                    "action": "worker_pool_draining",
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                },
            )
            deadline = time.time() + args.drain_hours * 3600.0
            while workers.poll() is None and time.time() < deadline:
                time.sleep(30)
            if workers.poll() is None:
                workers.terminate()
                workers.wait(timeout=120)
            replacement = None if args.no_self_chain else submit_replacement(args)
            append_jsonl(
                paths.decisions,
                {
                    "timestamp": utc_now(),
                    "action": "worker_pool_self_chain",
                    "replacement_job_id": replacement,
                },
            )
            return
        time.sleep(30)
    if workers.returncode:
        raise SystemExit(workers.returncode)


if __name__ == "__main__":
    main()
