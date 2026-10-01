#!/usr/bin/env python3
"""Render representative held-out half-conversation continuations to Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCES = ("ami", "meld", "werewolf")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text_by_agent(side: dict) -> str:
    lines = []
    for agent, text in enumerate(side.get("decoded_by_agent") or []):
        text = str(text or "").strip()
        if text:
            lines.append(f"- **A{agent}:** {text}")
    return "\n".join(lines) or "_silent_"


def repetition_score(record: dict) -> float:
    metrics = (record.get("model") or {}).get("metrics") or {}
    return float(metrics.get("repeated_4gram_fraction") or 0.0)


def handoff_summary(record: dict) -> str:
    model = record.get("model") or {}
    metrics = model.get("metrics") or {}
    return (
        f"clean={metrics.get('clean_handoff_rate', 'n/a')}, "
        f"overlap={metrics.get('overlap_rate', 'n/a')}, "
        f"repeat-4gram={repetition_score(record):.3f}"
    )


def choose(records: list[dict], source: str) -> list[tuple[str, dict]]:
    source_records = [r for r in records if r.get("source") == source]
    if not source_records:
        return []
    ranked = sorted(source_records, key=repetition_score)
    if all(repetition_score(record) == 0.0 for record in ranked):
        picks = [("first screen example", ranked[0]), ("last screen example", ranked[-1])]
        if len(ranked) > 2:
            picks.insert(1, ("middle screen example", ranked[len(ranked) // 2]))
        return picks
    picks = [("lowest repetition", ranked[0]), ("highest repetition", ranked[-1])]
    if len(ranked) > 2:
        picks.insert(1, ("median repetition", ranked[len(ranked) // 2]))
    return picks


def render(model_label: str, records: list[dict]) -> str:
    lines = [
        f"# Qualitative Held-Out Screen: {model_label}",
        "",
        "Each example teacher-forces the first half of a real held-out "
        "conversation, then compares the human continuation with the "
        "autoregressive model continuation.",
        "",
    ]
    for source in SOURCES:
        lines.extend([f"## {source.upper()}", ""])
        for label, record in choose(records, source):
            lines.extend(
                [
                    f"### {label}: `{record.get('conversation_id') or record.get('trial_id')}`",
                    "",
                    f"Model metrics: {handoff_summary(record)}",
                    "",
                    "**Teacher-forced first half**",
                    "",
                    text_by_agent(record.get("prefix") or {}),
                    "",
                    "**Human second half**",
                    "",
                    text_by_agent(record.get("human") or {}),
                    "",
                    "**Model second half**",
                    "",
                    text_by_agent(record.get("model") or {}),
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.label, read_jsonl(args.input)), encoding="utf-8")


if __name__ == "__main__":
    main()
