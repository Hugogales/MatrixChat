#!/usr/bin/env python3
"""Audit per-source conversation quality on processed MatrixChat shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import load_from_disk

from data.conversation_quality import score_matrix_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--max-flat-len", type=int, default=2048)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def summarize_source(dataset, max_flat_len: int) -> dict:
    scores = []
    overlap_rates = []
    handoff_rates = []
    exactly_one_rates = []
    examples = 0
    for example in dataset:
        agents = int(example["num_agents"])
        length = int(example["length"])
        if max_flat_len and agents * length > max_flat_len:
            continue
        examples += 1
        if "conversation_quality_score" in example:
            score = float(example["conversation_quality_score"])
            overlap = float(example.get("conversation_quality_overlap_rate", 0.0))
            handoff = float(example.get("conversation_quality_handoff_rate", 0.0))
            exactly_one = float(example.get("conversation_quality_exactly_one_rate", 0.0))
        else:
            scored = score_matrix_example(example)
            score = float(scored["conversation_quality_score"])
            overlap = float(scored["conversation_quality_overlap_rate"])
            handoff = float(scored["conversation_quality_handoff_rate"])
            exactly_one = float(scored["conversation_quality_exactly_one_rate"])
        scores.append(score)
        overlap_rates.append(overlap)
        handoff_rates.append(handoff)
        exactly_one_rates.append(exactly_one)

    if not scores:
        return {"examples": 0}

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    return {
        "examples": examples,
        "avg_conversation_quality_score": mean(scores),
        "median_conversation_quality_score": sorted(scores)[len(scores) // 2],
        "high_quality_fraction": sum(1 for value in scores if value >= 0.55) / len(scores),
        "avg_overlap_rate": mean(overlap_rates),
        "avg_handoff_rate": mean(handoff_rates),
        "avg_exactly_one_rate": mean(exactly_one_rates),
    }


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "processed_dir": str(processed_dir.resolve()),
        "split": args.split,
        "max_flat_len": args.max_flat_len,
        "by_source": {},
    }
    for source in sorted(manifest.get("sources", {})):
        dataset = load_from_disk(str(processed_dir / source))
        if args.split and "dataset_split" in dataset.column_names:
            dataset = dataset.filter(lambda example: example.get("dataset_split") == args.split)
        report["by_source"][source] = summarize_source(dataset, args.max_flat_len)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
