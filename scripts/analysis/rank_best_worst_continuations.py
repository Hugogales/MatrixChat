#!/usr/bin/env python3
"""Rank held-out final-run continuations by a turn-taking quality composite,
and print the best/worst examples with decoded text side-by-side.

Purpose: the aggregate stats in a wave's ``analysis/summary.json`` show the
existence and size of a human-vs-model gap, but not WHERE it comes from --
this pulls concrete per-example evidence (quantitative score + qualitative
decoded transcript) so a human can actually read the failure modes, per
``.cursor/rules/read-decoded-text-not-just-metrics.mdc``.

Composite score (per example, model side only -- human side is the ~fixed
reference the model is being compared against):

    score = boundary_handoff.strict.model
          - boundary_handoff.dirty.model
          - 0.5 * boundary_handoff.interruption.model
          - columns.overlap_rate.model
          - max(0, repeated4gram_fraction.model - 0.35)

Higher is better. This is a diagnostic ranking heuristic, not a training
signal -- it is never used for anything other than picking which examples to
read.

Usage:
    python scripts/analysis/rank_best_worst_continuations.py \\
        logs/final_runs_20260827_wave52_dynbal/evaluations/final52_h100_dynbal_s273106/step_1500 \\
        --top 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def composite_score(row: dict) -> float:
    strict = _float(row.get("boundary_handoff.strict.model"))
    dirty = _float(row.get("boundary_handoff.dirty.model"))
    interruption = _float(row.get("boundary_handoff.interruption.model"))
    overlap = _float(row.get("columns.overlap_rate.model"))
    repeated = _float(row.get("repeated4gram_fraction.model"))
    return (
        strict
        - dirty
        - 0.5 * interruption
        - overlap
        - max(0.0, repeated - 0.35)
    )


def load_metrics(eval_dir: Path) -> list[dict]:
    path = eval_dir / "analysis" / "per_example_metrics.csv"
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_transcripts(eval_dir: Path) -> dict[str, dict]:
    path = eval_dir / "analysis" / "per_example_transcripts.jsonl"
    by_trial = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_trial[record["trial_id"]] = record
    return by_trial


def render_side(decoded_by_agent: list[str], label: str) -> str:
    lines = [f"  -- {label} --"]
    any_text = False
    for idx, text in enumerate(decoded_by_agent):
        if text and text.strip():
            any_text = True
            lines.append(f"    col{idx}: {text.strip()[:280]!r}")
    if not any_text:
        lines.append("    (empty -- pure silence)")
    return "\n".join(lines)


def print_example(row: dict, transcript: dict, score: float) -> None:
    print(
        f"\n=== {row['source']} / {row['conversation_id']} "
        f"(trial {row['trial_id']}, score={score:+.3f}) ==="
    )
    print(
        "  metrics (human -> model): "
        f"strict={row['boundary_handoff.strict.human']}->{row['boundary_handoff.strict.model']}  "
        f"dirty={row['boundary_handoff.dirty.human']}->{row['boundary_handoff.dirty.model']}  "
        f"overlap_rate={row['columns.overlap_rate.human']}->{row['columns.overlap_rate.model']}  "
        f"repeated4gram={row['repeated4gram_fraction.human']}->{row['repeated4gram_fraction.model']}"
    )
    print(render_side(transcript["human_decoded_by_agent"], "HUMAN (real continuation)"))
    print(render_side(transcript["model_decoded_by_agent"], "MODEL (generated continuation)"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=Path, help="e.g. logs/.../evaluations/<run>/step_1500")
    parser.add_argument("--top", type=int, default=3, help="How many best/worst examples to show")
    args = parser.parse_args(argv)

    rows = load_metrics(args.eval_dir)
    transcripts = load_transcripts(args.eval_dir)
    scored = [(composite_score(row), row) for row in rows]
    scored.sort(key=lambda pair: pair[0])

    worst = scored[: args.top]
    best = scored[-args.top :][::-1]

    scores_only = [s for s, _ in scored]
    mean_score = sum(scores_only) / len(scores_only) if scores_only else 0.0
    print(f"eval_dir={args.eval_dir}")
    print(f"n_examples={len(scored)}  mean_composite_score={mean_score:+.3f}")

    print("\n" + "#" * 20 + f" TOP {args.top} BEST (model most human-like handoff behavior) " + "#" * 20)
    for score, row in best:
        print_example(row, transcripts[row["trial_id"]], score)

    print("\n" + "#" * 20 + f" TOP {args.top} WORST (model least human-like handoff behavior) " + "#" * 20)
    for score, row in worst:
        print_example(row, transcripts[row["trial_id"]], score)


if __name__ == "__main__":
    main()
