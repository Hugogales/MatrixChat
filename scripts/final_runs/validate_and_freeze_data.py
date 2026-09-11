#!/usr/bin/env python3
"""Validate every final-data row and freeze full/screening eval contracts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.contract import build_eval_contract, write_eval_contract

_MATRIX_FIELDS = (
    "input_ids",
    "labels",
    "agent_ids",
    "input_activity_mask",
    "activity_labels",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--contract-dir", required=True)
    parser.add_argument("--screening-max-per-source", type=int, default=20)
    return parser.parse_args(argv)


def _validate_matrix(
    source: str, index: int, row: dict[str, Any], name: str, agents: int, length: int
) -> None:
    value = row.get(name)
    if value is None:
        raise ValueError(f"{source}[{index}] is missing {name}")
    if len(value) != agents or any(len(agent) != length for agent in value):
        raise ValueError(
            f"{source}[{index}] {name} shape does not equal [{agents}, {length}]"
        )


def validate_processed_data(processed_dir: Path) -> dict[str, Any]:
    from datasets import load_from_disk

    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    sources = manifest.get("sources") or {}
    if not sources:
        raise ValueError("manifest has no sources")

    split_counts: dict[str, Counter] = {}
    identifiers: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    row_counts: dict[str, int] = {}
    for source, info in sorted(sources.items()):
        dataset = load_from_disk(str(processed_dir / info.get("path", source)))
        counts: Counter = Counter()
        for index, raw_row in enumerate(dataset):
            row = dict(raw_row)
            agents = int(row["num_agents"])
            length = int(row["length"])
            if agents <= 0 or length <= 0:
                raise ValueError(f"{source}[{index}] has invalid agents/length")
            for name in _MATRIX_FIELDS:
                _validate_matrix(source, index, row, name, agents, length)
            if row.get("is_private_mask") is not None:
                _validate_matrix(
                    source, index, row, "is_private_mask", agents, length
                )
            visibility = row.get("agent_visibility")
            if visibility is not None and (
                len(visibility) != agents
                or any(len(agent) != agents for agent in visibility)
            ):
                raise ValueError(
                    f"{source}[{index}] agent_visibility shape does not equal "
                    f"[{agents}, {agents}]"
                )
            split = str(
                row.get("dataset_split", row.get("split")) or "train"
            ).lower()
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"{source}[{index}] has invalid split {split!r}")
            counts[split] += 1
            for kind in ("conversation_id", "group_id"):
                identifier = str(row.get(kind) or "")
                if not identifier:
                    continue
                key = (source, identifier)
                previous = identifiers[kind].get(key)
                if previous is not None and previous != split:
                    raise ValueError(
                        f"{kind} leakage for {source}:{identifier}: "
                        f"{previous} and {split}"
                    )
                identifiers[kind][key] = split
        row_counts[source] = len(dataset)
        split_counts[source] = counts
        for split in ("train", "validation", "test"):
            if counts[split] <= 0:
                raise ValueError(f"{source} has no {split} examples")

    return {
        "processed_dir": str(processed_dir),
        "row_counts": row_counts,
        "split_counts": {
            source: dict(sorted(counts.items()))
            for source, counts in split_counts.items()
        },
    }


def _contract_counts(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        split: {
            "total": len(contract[split]),
            "by_source": dict(
                sorted(Counter(entry["source"] for entry in contract[split]).items())
            ),
        }
        for split in ("development", "final_test")
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    processed_dir = Path(args.processed_dir).expanduser().resolve()
    contract_dir = Path(args.contract_dir).expanduser().resolve()
    contract_dir.mkdir(parents=True, exist_ok=True)

    summary = validate_processed_data(processed_dir)
    full = build_eval_contract(processed_dir)
    screening = build_eval_contract(
        processed_dir, max_per_source=args.screening_max_per_source
    )
    write_eval_contract(full, contract_dir / "publication_full.json")
    write_eval_contract(screening, contract_dir / "development_screening.json")
    summary["publication_contract"] = _contract_counts(full)
    summary["screening_contract"] = _contract_counts(screening)
    (contract_dir / "data_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
