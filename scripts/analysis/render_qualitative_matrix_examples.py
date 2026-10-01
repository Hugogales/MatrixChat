#!/usr/bin/env python3
"""Render best and worst held-out continuations as token-cell matrices."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def numeric(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except ValueError:
        return 0.0


def quality(row: dict) -> float:
    """Strict clean handoffs, minus silence, overlap, dirty transitions, repeats."""
    return (
        numeric(row, "boundary_handoff.strict.model")
        - numeric(row, "boundary_handoff.no_handoff.model")
        - numeric(row, "boundary_handoff.dirty.model")
        - numeric(row, "columns.overlap_rate.model")
        - numeric(row, "repeated4gram_fraction.model")
    )


def matrix(side: dict, max_columns: int) -> str:
    cells = side.get("cells") or []
    num_agents = max((len(column) for column in cells), default=0)
    display = cells[:max_columns]
    lines = ["| col | " + " | ".join(f"A{agent}" for agent in range(num_agents)) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in range(num_agents)) + " |")
    for col, values in enumerate(display):
        escaped = [str(value or "—").replace("|", "\\|").replace("\n", " ") for value in values]
        escaped.extend(["—"] * (num_agents - len(escaped)))
        lines.append(f"| {col} | " + " | ".join(escaped) + " |")
    if len(cells) > max_columns:
        lines.append(f"\n*(showing first {max_columns} of {len(cells)} columns)*")
    return "\n".join(lines)


def render_record(record: dict, score: float, label: str, max_columns: int) -> str:
    return "\n".join(
        [
            f"### {label}: `{record.get('source')} / {record.get('conversation_id')}`",
            "",
            f"Screen quality score: `{score:.3f}`",
            "",
            "#### Teacher-forced first half",
            "",
            matrix(record.get("prefix") or {}, max_columns),
            "",
            "#### Human continuation",
            "",
            matrix(record.get("human") or {}, max_columns),
            "",
            "#### Model continuation",
            "",
            matrix(record.get("model") or {}, max_columns),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-columns", type=int, default=48)
    args = parser.parse_args()

    by_trial = {record["trial_id"]: record for record in read_jsonl(args.pairs)}
    by_source: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    with args.metrics.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record = by_trial.get(row.get("trial_id", ""))
            if record and record.get("source") in {"ami", "meld", "werewolf"}:
                by_source[record["source"]].append((quality(row), record))

    lines = [
        f"# Matrix Examples: {args.label}",
        "",
        "Each case uses a real held-out first half, followed by the human and "
        "autoregressive model continuations. Examples are selected as the "
        "highest and lowest score within each source, where higher rewards "
        "strict handoffs and penalizes silence, dirty transitions, overlap, "
        "and repeated 4-grams.",
        "",
    ]
    for source in ("ami", "meld", "werewolf"):
        examples = sorted(by_source[source], key=lambda item: item[0])
        lines.extend([f"## {source.upper()}", ""])
        if not examples:
            lines.extend(["_No screen examples found._", ""])
            continue
        lines.append(render_record(examples[-1][1], examples[-1][0], "Best", args.max_columns))
        lines.append(render_record(examples[0][1], examples[0][0], "Worst", args.max_columns))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
