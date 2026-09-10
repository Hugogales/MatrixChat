#!/usr/bin/env python3
"""One-time migration: switch a live search from random search to TPE.

Backfills every already-scored candidate in ``search_state.json`` into a
fresh Optuna study (using each candidate's most-advanced-rung score) and
flips ``config.optimizer`` to ``"tpe"``. Safe to run against a LIVE search
directory while the controller is stopped -- this script only edits
``search_state.json`` and creates ``optuna_study.db``; it does not touch
tickets, checkpoints, or history.

The running controller process must be stopped first (it holds the old
config in memory and would silently overwrite this edit on its next write).
Restart it after this script finishes.

Usage:
    scancel <controller_job_id>
    python scripts/search/enable_tpe.py --search-dir searches/month_2026_08
    sbatch --export=ALL,SEARCH_DIR="$PWD/searches/month_2026_08" scripts/search/controller.sbatch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.search import bayes_opt  # noqa: E402
from scripts.search.search_state import (  # noqa: E402
    SearchPaths,
    atomic_write_json,
    read_json,
    release_directory_lock,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--tpe-startup-trials", type=int, default=10)
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

    if state["config"].get("optimizer") == "tpe":
        print("optimizer is already 'tpe' -- nothing to do.")
        return

    study = bayes_opt.load_study(
        paths.root, seed=int(state["config"]["search_seed"]), n_startup_trials=args.tpe_startup_trials
    )
    before = len(study.trials)
    seeded = bayes_opt.backfill_from_state(study, state)
    print(f"seeded {seeded} historical candidates into the study (had {before} trials before).")

    total_candidates = len(state.get("candidates", {}))
    scored_candidates = sum(1 for c in state["candidates"].values() if c.get("scores"))
    skipped = scored_candidates - seeded
    if skipped:
        print(
            f"NOTE: {skipped}/{scored_candidates} scored candidates were skipped "
            "(legacy/partial config missing one or more current search-space keys)."
        )

    state["config"]["optimizer"] = "tpe"
    state["config"]["tpe_startup_trials"] = args.tpe_startup_trials
    atomic_write_json(paths.state, state)
    print(f"patched {paths.state} -> config.optimizer = 'tpe'")

    if paths.controller_lock.exists():
        if args.force_unlock:
            release_directory_lock(paths.controller_lock)
            print(f"released stale {paths.controller_lock}")
        else:
            print(
                f"WARNING: {paths.controller_lock} still exists. If the controller "
                "job has been cancelled, rerun with --force-unlock before resubmitting."
            )

    print("Done. Resubmit controller.sbatch (SEARCH_CONFIG is ignored on resume; "
          "the patched search_state.json is now the source of truth).")


if __name__ == "__main__":
    main()
