#!/usr/bin/env python3
"""Recover candidates falsely failed by the broad-eval import-path bug.

Run only while the controller and all worker pools are stopped. The script:
1. backs up canonical state and the contaminated Optuna study;
2. requeues evaluation-only failures whose checkpoints are already complete;
3. starts a clean TPE study seeded only from valid scored candidates; and
4. expands the search population/budget for the larger worker allocation.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.search import bayes_opt  # noqa: E402
from scripts.search.controller import enqueue_evaluation  # noqa: E402
from scripts.search.search_state import (  # noqa: E402
    SearchPaths,
    atomic_write_json,
    read_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--target-active-candidates", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = SearchPaths(Path(args.search_dir).resolve())
    paths.initialize()
    state = read_json(paths.state)
    if not state:
        raise FileNotFoundError(paths.state)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    state_backup = paths.root / f"search_state.before_eval_recovery.{stamp}.json"
    shutil.copy2(paths.state, state_backup)

    study_file = bayes_opt.study_path(paths.root)
    contaminated_study = None
    if study_file.exists():
        contaminated_study = study_file.with_name(
            f"{study_file.stem}.contaminated.{stamp}{study_file.suffix}"
        )
        study_file.rename(contaminated_study)

    # Trial numbers belong to the archived study. Existing candidates remain
    # useful in state/reports, but must never tell those stale trial numbers
    # into the freshly seeded study.
    for candidate in state.get("candidates", {}).values():
        if candidate.get("optuna_trial_number") is not None:
            candidate["optuna_told"] = True

    recovered = []
    marker = "ModuleNotFoundError: No module named 'model'"
    for candidate in state.get("candidates", {}).values():
        if candidate.get("status") != "failed" or marker not in str(
            candidate.get("failure", "")
        ):
            continue
        candidate_id = candidate["candidate_id"]
        checkpoint = _ROOT / "checkpoints" / candidate_id / "last" / "model_state.pt"
        if not checkpoint.is_file():
            continue
        candidate["failure"] = None
        candidate["evaluation_retries"] = 0
        candidate["recovered_from_eval_import_bug"] = stamp
        enqueue_evaluation(paths, state, candidate)
        recovered.append(candidate_id)

    state["config"]["target_active_candidates"] = args.target_active_candidates
    state["config"]["max_candidates"] = args.max_candidates
    state["config"]["eval_import_recovery_at"] = stamp
    atomic_write_json(paths.state, state)

    study = bayes_opt.load_study(
        paths.root,
        seed=int(state["config"]["search_seed"]),
        n_startup_trials=int(state["config"].get("tpe_startup_trials", 10)),
    )
    seeded = bayes_opt.backfill_from_state(study, state)
    print(f"state backup: {state_backup}")
    print(f"contaminated study: {contaminated_study}")
    print(f"recovered evaluations: {len(recovered)}")
    print(f"valid trials seeded into fresh TPE study: {seeded}")
    print(
        "new limits:",
        state["config"]["target_active_candidates"],
        state["config"]["max_candidates"],
    )


if __name__ == "__main__":
    main()
