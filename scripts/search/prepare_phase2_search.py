#!/usr/bin/env python3
"""Prepare a clean phase-two search seeded by corrected checkpoint re-evals.

The source search remains untouched. Strong source candidates are imported as
evaluation-only records, pointed at immutable per-rung checkpoint snapshots,
and queued for a fresh v3 broad sweep. The phase-two controller scores those
results under the new strict+dirty chain objective and adds them to its new
Optuna study before sampling any new candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.search.controller import initial_state, load_run_config  # noqa: E402
from scripts.search import bayes_opt  # noqa: E402
from scripts.search.search_state import (  # noqa: E402
    SearchPaths,
    atomic_write_json,
    enqueue_ticket,
    read_json,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-search-dir", required=True)
    parser.add_argument("--target-search-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-score", type=int, default=8)
    parser.add_argument("--top-chain", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def scored_checkpoints(source_state: dict) -> list[dict]:
    rows = []
    for candidate in source_state.get("candidates", {}).values():
        for rung_text, score in (candidate.get("scores") or {}).items():
            checkpoint_step = score.get("checkpoint_step")
            if checkpoint_step is None or score.get("score") is None:
                continue
            probe = read_json(Path(score.get("probe_path", "")), {})
            rows.append(
                {
                    "candidate": candidate,
                    "rung": int(rung_text),
                    "step": int(checkpoint_step),
                    "score": float(score["score"]),
                    "disqualified": bool(score.get("disqualified")),
                    "chain_2plus_rate": float(
                        probe.get("chain_2plus_rate", 0.0) or 0.0
                    ),
                    "clean_handoff_rate": float(
                        probe.get("clean_handoff_rate", 0.0) or 0.0
                    ),
                    "probe_path": score.get("probe_path"),
                }
            )
    return rows


def checkpoint_path(
    candidate_id: str, rung: int, step: int, current_step: int
) -> Path | None:
    root = _ROOT / "checkpoints" / candidate_id
    snapshot = root / f"rung{rung}_step{step}"
    if (snapshot / "model_state.pt").is_file():
        return snapshot.resolve()
    last = root / "last"
    if (last / "model_state.pt").is_file() and current_step == step:
        return last.resolve()
    return None


def select_imports(
    source_state: dict, top_score: int, top_chain: int
) -> tuple[list[dict], list[dict]]:
    rows = scored_checkpoints(source_state)
    available = []
    skipped = []
    for row in rows:
        candidate_id = row["candidate"]["candidate_id"]
        checkpoint = checkpoint_path(
            candidate_id,
            row["rung"],
            row["step"],
            int(row["candidate"].get("step", -1)),
        )
        if checkpoint is None:
            skipped.append({**row, "reason": "exact checkpoint unavailable"})
            continue
        available.append({**row, "checkpoint_dir": str(checkpoint)})

    # One imported checkpoint per candidate: prefer its strongest ungated
    # score; if all of a candidate's scored rungs are gated, retain its best.
    best_by_candidate = {}
    for row in available:
        candidate_id = row["candidate"]["candidate_id"]
        incumbent = best_by_candidate.get(candidate_id)
        rank = (not row["disqualified"], row["score"], row["rung"])
        incumbent_rank = (
            (not incumbent["disqualified"], incumbent["score"], incumbent["rung"])
            if incumbent
            else None
        )
        if incumbent is None or rank > incumbent_rank:
            best_by_candidate[candidate_id] = row
    unique = list(best_by_candidate.values())

    by_score = sorted(
        unique,
        key=lambda row: (
            not row["disqualified"],
            row["score"],
            row["clean_handoff_rate"],
        ),
        reverse=True,
    )
    by_chain = sorted(
        unique,
        key=lambda row: (
            row["chain_2plus_rate"],
            row["clean_handoff_rate"],
            row["score"],
        ),
        reverse=True,
    )
    selected = []
    seen = set()
    for row in by_score[:top_score] + by_chain[:top_chain]:
        candidate_id = row["candidate"]["candidate_id"]
        if candidate_id not in seen:
            selected.append(row)
            seen.add(candidate_id)
    return selected, skipped


def optimizer_seedable(candidate_config: dict, search_config: dict) -> bool:
    try:
        distributions = bayes_opt.distributions_for(candidate_config, search_config)
        return all(
            distribution._contains(
                distribution.to_internal_repr(candidate_config[key])
            )
            for key, distribution in distributions.items()
        )
    except (KeyError, TypeError, ValueError):
        return False


def imported_candidate(
    row: dict, source_dir: Path, search_config: dict
) -> dict:
    source = row["candidate"]
    return {
        "candidate_id": source["candidate_id"],
        "created_at": utc_now(),
        "config": source["config"],
        "status": "queued_eval",
        "rung": row["rung"],
        "step": row["step"],
        "accumulated_train_seconds": float(
            source.get("accumulated_train_seconds", 0.0)
        ),
        "scores": {},
        "history": [],
        "evaluation_only": True,
        "seed_optimizer": optimizer_seedable(source["config"], search_config),
        "evaluation_checkpoint_dir": row["checkpoint_dir"],
        "imported_from": {
            "search_dir": str(source_dir.resolve()),
            "source_status": source.get("status"),
            "source_rung": row["rung"],
            "source_step": row["step"],
            "source_score": row["score"],
            "source_chain_2plus_rate": row["chain_2plus_rate"],
            "source_probe_path": row["probe_path"],
        },
    }


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_search_dir)
    target_dir = Path(args.target_search_dir)
    source_state = read_json(source_dir / "search_state.json")
    if not source_state:
        raise SystemExit(f"missing source state: {source_dir / 'search_state.json'}")
    if target_dir.exists() and any(target_dir.iterdir()):
        raise SystemExit(f"target directory is not empty: {target_dir}")

    selected, skipped = select_imports(
        source_state, top_score=args.top_score, top_chain=args.top_chain
    )
    if not selected:
        raise SystemExit("no selected candidates have an exact preserved checkpoint")

    config = load_run_config(args.config)
    summary = [
        {
            "candidate_id": row["candidate"]["candidate_id"],
            "rung": row["rung"],
            "step": row["step"],
            "old_score": row["score"],
            "old_clean_handoff_rate": row["clean_handoff_rate"],
            "old_chain_2plus_rate": row["chain_2plus_rate"],
            "checkpoint_dir": row["checkpoint_dir"],
            "optimizer_seedable": optimizer_seedable(
                row["candidate"]["config"], config
            ),
        }
        for row in selected
    ]
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    paths = SearchPaths(target_dir.resolve())
    paths.initialize()
    state = initial_state(config)
    for row in selected:
        candidate = imported_candidate(row, source_dir, config)
        candidate_id = candidate["candidate_id"]
        state["candidates"][candidate_id] = candidate
        candidate_dir = paths.candidate_dir(candidate_id)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(candidate_dir / "config.json", candidate["config"])
        atomic_write_json(candidate_dir / "imported_from.json", candidate["imported_from"])
        rung = config["rungs"][candidate["rung"]]
        enqueue_ticket(
            paths,
            {
                "type": "broad_evaluate",
                "candidate_id": candidate_id,
                "priority": int(rung["priority"]) + 1000,
                "checkpoint_step": candidate["step"],
                "checkpoint_dir": candidate["evaluation_checkpoint_dir"],
                "max_new_tokens": config["eval_max_new_tokens"],
            },
        )

    state["phase2_import"] = {
        "created_at": utc_now(),
        "source_search_dir": str(source_dir.resolve()),
        "config_path": str(Path(args.config).resolve()),
        "selected": summary,
        "skipped_without_exact_checkpoint": len(skipped),
    }
    atomic_write_json(paths.state, state)
    atomic_write_json(target_dir / "migration_manifest.json", state["phase2_import"])
    print(f"prepared {target_dir} with {len(selected)} corrected re-evaluations")


if __name__ == "__main__":
    main()
