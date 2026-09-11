#!/usr/bin/env python3
"""One-time migration: move a live H100 search onto V100 workers.

Sets ``config.gpu_memory_gb`` (gating the memory-safe clamp in
``controller.sample_candidate``) and retroactively re-clamps every
non-terminal candidate's config the exact same way, since an in-flight
candidate resumes training from its saved checkpoint using its OWN stored
config -- without this, any already-sampled candidate with
``batch_size > 1`` would immediately OOM the first time a V100 worker
resumes it.

The running controller process must be stopped first (it holds the old
config in memory and would silently overwrite this edit on its next write).
Restart it after this script finishes.

Usage:
    scancel <controller_job_id> <watchdog_job_id> <old_worker_pool_job_id>
    python scripts/search/migrate_to_v100.py --search-dir searches/month_2026_08
    sbatch --export=ALL,SEARCH_DIR="$PWD/searches/month_2026_08" scripts/search/controller.sbatch
    sbatch --export=ALL,SEARCH_DIR="$PWD/searches/month_2026_08" scripts/search/watchdog.sbatch
    sbatch --export=ALL,SEARCH_DIR="$PWD/searches/month_2026_08" scripts/search/worker_pool_v100.sbatch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.search.search_state import (  # noqa: E402
    SearchPaths,
    atomic_write_json,
    read_json,
    release_directory_lock,
)

TERMINAL_STATUSES = {"pruned", "failed", "finalist"}


def clamp_for_v100(config: dict) -> bool:
    """Apply the same fold-batch-into-accumulation clamp as sample_candidate.

    Returns True if the config was changed.
    """
    changed = False
    if int(config.get("batch_size", 1)) > 1:
        config["gradient_accumulation_steps"] = int(
            config.get("gradient_accumulation_steps", 1)
        ) * int(config["batch_size"])
        config["batch_size"] = 1
        changed = True
    if not config.get("gradient_checkpointing"):
        config["gradient_checkpointing"] = True
        changed = True
    if int(config.get("max_flat_len", 1536)) > 1536:
        config["max_flat_len"] = 1536
        changed = True
    return changed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--target-active-candidates", type=int, default=None,
                         help="Optionally resize the active-candidate pool to match the new worker count.")
    parser.add_argument(
        "--force-unlock",
        action="store_true",
        help="Release controller.lock even if it looks fresh (only use once "
        "the controller job has actually been cancelled).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = SearchPaths(Path(args.search_dir).resolve())
    state = read_json(paths.state)
    if not state:
        raise SystemExit(f"no search_state.json found under {paths.root}")

    if state["config"].get("gpu_memory_gb", 80) <= 40:
        print("gpu_memory_gb already <= 40 -- migration already applied, nothing to do.")
        return

    state["config"]["gpu_memory_gb"] = 32
    if args.target_active_candidates:
        state["config"]["target_active_candidates"] = args.target_active_candidates

    retro_clamped = 0
    for candidate in state.get("candidates", {}).values():
        if candidate.get("status") in TERMINAL_STATUSES:
            continue
        if clamp_for_v100(candidate["config"]):
            retro_clamped += 1

    atomic_write_json(paths.state, state)
    print(f"patched {paths.state} -> config.gpu_memory_gb = 32")
    print(f"retroactively clamped {retro_clamped} non-terminal candidate(s) to V100-safe settings")

    if paths.controller_lock.exists():
        if args.force_unlock:
            release_directory_lock(paths.controller_lock)
            print(f"released stale {paths.controller_lock}")
        else:
            print(
                f"WARNING: {paths.controller_lock} still exists. If the controller "
                "job has been cancelled, rerun with --force-unlock before resubmitting."
            )

    print("Done. Resubmit controller.sbatch + watchdog.sbatch + worker_pool_v100.sbatch.")


if __name__ == "__main__":
    main()
