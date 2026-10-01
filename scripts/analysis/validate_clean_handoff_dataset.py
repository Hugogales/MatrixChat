#!/usr/bin/env python3
"""Validate structural integrity and summary statistics of a processed dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


def validate_source(path: Path, source: str) -> dict:
    dataset = load_from_disk(str(path / source))
    total_overlap = total_pause = total_active = 0
    lengths = []
    for index, example in enumerate(dataset):
        rows = example["num_agents"]
        length = example["length"]
        fields = ("input_ids", "input_activity_mask", "labels", "activity_labels", "agent_ids")
        for field in fields:
            matrix = example[field]
            if len(matrix) != rows or any(len(row) != length for row in matrix):
                raise ValueError(f"{source}[{index}] invalid {field} shape")
        if "is_private_mask" in example:
            matrix = example["is_private_mask"]
            if len(matrix) != rows or any(len(row) != length for row in matrix):
                raise ValueError(f"{source}[{index}] invalid is_private_mask shape")
        active = sum(sum(row) for row in example["input_activity_mask"])
        overlap = sum(
            sum(example["input_activity_mask"][row][col] for row in range(rows)) >= 2
            for col in range(length)
        )
        pause = sum(
            sum(example["input_activity_mask"][row][col] for row in range(rows)) == 0
            for col in range(length)
        )
        if overlap != example["num_overlap_columns"] or pause != example["num_pause_columns"]:
            raise ValueError(f"{source}[{index}] summary counters disagree with matrix")
        total_overlap += overlap
        total_pause += pause
        total_active += active
        lengths.append(length)
    columns = sum(lengths)
    return {
        "examples": len(dataset),
        "columns": columns,
        "active_tokens": total_active,
        "overlap_columns": total_overlap,
        "overlap_rate_pct": 100 * total_overlap / columns if columns else 0.0,
        "pause_columns": total_pause,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    args = parser.parse_args()
    root = Path(args.processed_dir).expanduser()
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    results = {
        source: validate_source(root, source)
        for source in manifest.get("sources", {})
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
