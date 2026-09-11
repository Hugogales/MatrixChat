#!/usr/bin/env python3
"""Measure human same-speaker/handoff topology at RUN boundaries, not columns.

The existing ``analyze_human_turn_topology.py`` computes a per-COLUMN
marginal: every consecutive pair of "clean" (unique-owner) columns is scored
independently. In the timed hybrid conversion, a single spoken utterance
occupies many consecutive token columns on one agent's row, so the vast
majority of "same-speaker" column pairs are trivial mid-utterance
continuations (literally the next token of the same ongoing utterance), not
genuine turn-taking decisions. That per-column marginal is NOT the same
quantity as the model-side evaluation metrics (``clean_handoff_rate`` /
``_boundary_handoff`` in ``evaluation/continuation_metrics.py``), which are
conditioned on an actual run boundary: "after the reference speaker's run of
activity ends, who (if anyone) becomes the next sole speaker."

This script instead collapses each maximal same-owner run into a single
event and asks, at the end of every run, who the eventual next sole speaker
is (scanning forward through any silence/overlap columns), classifying it as
a "self-resumption" (same owner) or a genuine "handoff" (different owner).
This is directly comparable to the model's boundary-conditioned metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk


def parse_weights(value: str) -> dict[str, float]:
    items = {}
    for piece in value.split(","):
        name, weight = piece.split("=", 1)
        items[name.strip()] = float(weight)
    total = sum(items.values())
    return {name: weight / total for name, weight in items.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--dataset-weights", default="meld=0.15,ami=0.50,werewolf=0.35")
    parser.add_argument("--max-flat-len", type=int, default=2048)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _owner_at(activity: list[list[int]], agents: int, column: int) -> int | None:
    """Return the sole active owner at ``column``, or None if 0 or 2+ active."""
    speakers = [agent for agent in range(agents) if activity[agent][column]]
    return speakers[0] if len(speakers) == 1 else None


def run_boundary_counts(dataset, max_flat_len: int) -> tuple[int, int, Counter]:
    """Collapse each maximal same-owner run into one boundary event.

    For every run's end column, scan forward for the next column with a
    unique sole owner (skipping silence/overlap gaps entirely, i.e. resolving
    through them the same way ``_boundary_handoff`` resolves listeners).
    Classifies each resolved boundary as ``self_resume`` (same owner) or
    ``handoff`` (different owner); also tracks whether the gap was zero
    columns (immediate), spanned silence, or spanned overlap, and separately
    counts boundaries that never resolve before the example ends
    (``unresolved``).
    """
    counts: Counter = Counter()
    examples = 0
    runs_total = 0
    for example in dataset:
        agents = int(example["num_agents"])
        length = int(example["length"])
        if max_flat_len and agents * length > max_flat_len:
            continue
        activity = example["input_activity_mask"]
        # Build the full realized activity timeline (context + supervised
        # columns): input_activity_mask already reflects ground truth at
        # each column for real (non-padding) positions.
        owners = [_owner_at(activity, agents, column) for column in range(length)]
        if all(owner is None for owner in owners):
            continue
        examples += 1

        column = 0
        while column < length:
            owner = owners[column]
            if owner is None:
                column += 1
                continue
            run_start = column
            while column < length and owners[column] == owner:
                column += 1
            run_end = column - 1  # last column where `owner` was sole speaker
            runs_total += 1

            # Classify the gap immediately after the run and resolve forward.
            gap_kind = "immediate"
            saw_silence = False
            saw_overlap = False
            scan = column
            resolved_owner = None
            while scan < length:
                speakers = [a for a in range(agents) if activity[a][scan]]
                if len(speakers) == 1:
                    resolved_owner = speakers[0]
                    break
                if len(speakers) == 0:
                    saw_silence = True
                else:
                    saw_overlap = True
                scan += 1
            if saw_overlap:
                gap_kind = "overlap"
            elif saw_silence:
                gap_kind = "silence"

            if resolved_owner is None:
                counts["unresolved"] += 1
                continue

            outcome = "self_resume" if resolved_owner == owner else "handoff"
            counts[outcome] += 1
            counts[f"{outcome}_{gap_kind}"] += 1
    return examples, runs_total, counts


def summarize(examples: int, runs_total: int, counts: Counter) -> dict:
    resolved = counts["self_resume"] + counts["handoff"]
    return {
        "examples": examples,
        "runs_total": runs_total,
        "counts": dict(counts),
        "resolved_boundaries": resolved,
        "unresolved_rate": counts["unresolved"] / runs_total if runs_total else 0.0,
        "rates_of_resolved": {
            "self_resume": counts["self_resume"] / resolved if resolved else 0.0,
            "handoff": counts["handoff"] / resolved if resolved else 0.0,
        },
        "rates_of_resolved_by_gap": {
            "self_resume_immediate": counts["self_resume_immediate"] / resolved if resolved else 0.0,
            "self_resume_silence": counts["self_resume_silence"] / resolved if resolved else 0.0,
            "self_resume_overlap": counts["self_resume_overlap"] / resolved if resolved else 0.0,
            "handoff_immediate": counts["handoff_immediate"] / resolved if resolved else 0.0,
            "handoff_silence": counts["handoff_silence"] / resolved if resolved else 0.0,
            "handoff_overlap": counts["handoff_overlap"] / resolved if resolved else 0.0,
        },
    }


def source_weighted(summaries: dict[str, dict], weights: dict[str, float]) -> dict:
    active = {name: weight for name, weight in weights.items() if name in summaries}
    total_weight = sum(active.values())
    active = {name: weight / total_weight for name, weight in active.items()}
    self_resume = sum(active[name] * summaries[name]["rates_of_resolved"]["self_resume"] for name in active)
    handoff = sum(active[name] * summaries[name]["rates_of_resolved"]["handoff"] for name in active)
    total = self_resume + handoff
    return {
        "source_weights": active,
        "rates_of_resolved": {
            "self_resume": self_resume / total if total else 0.0,
            "handoff": handoff / total if total else 0.0,
        },
    }


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    weights = parse_weights(args.dataset_weights)
    report = {
        "processed_dir": str(processed_dir.resolve()),
        "max_flat_len": args.max_flat_len,
        "dataset_weights": weights,
        "methodology": (
            "Collapses each maximal same-owner run into ONE boundary event, "
            "then resolves forward through any silence/overlap to find the "
            "next sole speaker. Classifies same-owner resolution as "
            "self_resume and different-owner resolution as handoff. This is "
            "directly comparable to evaluation/continuation_metrics.py's "
            "_boundary_handoff, unlike the per-column marginal in "
            "analyze_human_turn_topology.py."
        ),
        "splits": {},
    }
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        source_summaries = {}
        for source in weights:
            dataset = load_from_disk(str(processed_dir / source))
            if "dataset_split" in dataset.column_names:
                dataset = dataset.filter(lambda example: example.get("dataset_split") == split)
            if len(dataset) == 0:
                continue
            examples, runs_total, counts = run_boundary_counts(dataset, args.max_flat_len)
            if runs_total:
                source_summaries[source] = summarize(examples, runs_total, counts)
        report["splits"][split] = {
            "by_source": source_summaries,
            "source_weighted": source_weighted(source_summaries, weights) if source_summaries else None,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
