#!/usr/bin/env python3
"""Pick the highest-judge conversations per source for paper examples."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.continuation_metrics import count_clean_handoffs
from evaluation.paired_continuation import read_jsonl

CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff]")
WEREWOLF_LEAK = re.compile(
    r"\b(werewolf|werewolves|seer|villager|villagers|minion|witch)\b",
    re.IGNORECASE,
)
SOURCES = ("ami", "meld", "werewolf")


def _text(side: dict) -> str:
    return " ".join(str(part or "") for part in (side.get("decoded_by_agent") or []))


def _has_cjk(*sides: dict) -> bool:
    return any(CJK.search(_text(side)) for side in sides)


def _conversation_key(record: dict) -> str:
    group = str(record.get("group_id") or "").strip()
    conversation = str(record.get("conversation_id") or "").strip()
    return group or conversation or str(record.get("trial_id") or "")


def _judge_score(record: dict, who: str) -> float:
    judge = (record.get("judge") or {}).get(who) or {}
    try:
        return float(judge.get("aggregate") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _clean_handoff_count(record: dict, who: str = "model") -> int:
    side = record.get(who) or {}
    activity = side.get("activity")
    if not activity:
        return 0
    prefix_activity = (record.get("prefix") or {}).get("activity")
    try:
        return count_clean_handoffs(activity, prefix_activity)
    except (TypeError, ValueError):
        return 0


def _non_degeneracy(record: dict, who: str) -> float:
    axes = ((record.get("judge") or {}).get(who) or {}).get("axes") or {}
    try:
        return float(axes.get("non_degeneracy") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _turns(cells: list) -> list[dict]:
    """Collapse column cells into readable speaker turns."""
    turns: list[dict] = []
    current_agent: int | None = None
    buffer: list[str] = []
    overlap_by_agent: dict[int, list[str]] | None = None

    def flush_speech():
        nonlocal current_agent, buffer
        if current_agent is not None and buffer:
            turns.append({"agent": current_agent, "text": "".join(buffer).strip()})
        current_agent = None
        buffer = []

    def flush_overlap():
        nonlocal overlap_by_agent
        if not overlap_by_agent:
            overlap_by_agent = None
            return
        pieces = [
            f"A{index}: {''.join(tokens).strip()}"
            for index, tokens in sorted(overlap_by_agent.items())
            if "".join(tokens).strip()
        ]
        if pieces:
            turns.append({"agent": "overlap", "text": " | ".join(pieces)})
        overlap_by_agent = None

    for column in cells or []:
        speakers = [index for index, token in enumerate(column) if str(token or "").strip()]
        if not speakers:
            flush_speech()
            flush_overlap()
            continue
        if len(speakers) > 1:
            flush_speech()
            if overlap_by_agent is None:
                overlap_by_agent = {}
            for index in speakers:
                overlap_by_agent.setdefault(index, []).append(str(column[index] or ""))
            continue
        flush_overlap()
        agent = speakers[0]
        token = str(column[agent] or "")
        if agent != current_agent:
            flush_speech()
            current_agent = agent
        buffer.append(token)
    flush_speech()
    flush_overlap()
    return [turn for turn in turns if turn.get("text")]


def _example(record: dict) -> dict:
    prefix = record.get("prefix") or {}
    human = record.get("human") or {}
    model = record.get("model") or {}
    return {
        "trial_id": record.get("trial_id"),
        "source": record.get("source"),
        "conversation_id": record.get("conversation_id"),
        "group_id": record.get("group_id"),
        "judge_score_human": _judge_score(record, "human"),
        "judge_score_model": _judge_score(record, "model"),
        "non_degeneracy_model": _non_degeneracy(record, "model"),
        "clean_handoffs_model": _clean_handoff_count(record, "model"),
        "prefix_by_agent": prefix.get("decoded_by_agent") or [],
        "human_by_agent": human.get("decoded_by_agent") or [],
        "model_by_agent": model.get("decoded_by_agent") or [],
        "prefix_turns": _turns(prefix.get("cells") or []),
        "human_turns": _turns(human.get("cells") or []),
        "model_turns": _turns(model.get("cells") or []),
    }


def select_best(
    records: list[dict],
    *,
    per_source: int,
    min_clean_handoffs: int = 2,
) -> dict[str, list[dict]]:
    chosen: dict[str, list[dict]] = {source: [] for source in SOURCES}
    used: dict[str, set[str]] = {source: set() for source in SOURCES}
    ranked = sorted(
        records,
        key=lambda record: (
            _judge_score(record, "model"),
            _clean_handoff_count(record, "model"),
            _non_degeneracy(record, "model"),
            _judge_score(record, "human"),
        ),
        reverse=True,
    )
    for record in ranked:
        source = str(record.get("source") or "").lower()
        if source not in chosen or len(chosen[source]) >= per_source:
            continue
        if _clean_handoff_count(record, "model") < min_clean_handoffs:
            continue
        prefix = record.get("prefix") or {}
        human = record.get("human") or {}
        model = record.get("model") or {}
        if _has_cjk(prefix, human, model):
            continue
        if source != "werewolf" and WEREWOLF_LEAK.search(_text(model)):
            continue
        if _non_degeneracy(record, "model") < 1.0:
            continue
        key = _conversation_key(record)
        if key in used[source]:
            continue
        used[source].add(key)
        chosen[source].append(_example(record))
    return chosen


def _md_turns(turns: list[dict]) -> str:
    if not turns:
        return "_empty_"
    lines = []
    for turn in turns:
        agent = turn.get("agent")
        label = "overlap" if agent == "overlap" else f"A{agent}"
        text = str(turn.get("text") or "").replace("\n", " ")
        lines.append(f"- **{label}:** {text}")
    return "\n".join(lines)


def _md_agents(texts: list) -> str:
    lines = []
    for index, text in enumerate(texts or []):
        cleaned = str(text or "").replace("\n", " ").strip()
        if cleaned:
            lines.append(f"- **A{index}:** {cleaned}")
    return "\n".join(lines) if lines else "_empty_"


def render_markdown(chosen: dict[str, list[dict]]) -> str:
    parts = [
        "# Best conversations per dataset",
        "",
        "Selected by blinded judge score on the **model** continuation, requiring **at least two clean handoffs** in that continuation (chained ranking-event classifier: resolved listener floor, zero overlap). One conversation id per slot, no CJK, no Werewolf-term leakage onto AMI/MELD. MELD is included here as qualitative examples even though it is excluded from the statistical sample.",
        "",
        "Each example shows per-agent decoded text (easiest to paste into the paper) plus a collapsed turn view.",
        "",
    ]
    for source in SOURCES:
        parts.append(f"## {source.upper()}")
        parts.append("")
        if not chosen[source]:
            parts.append("_No examples passed the filters._")
            parts.append("")
            continue
        for index, example in enumerate(chosen[source], start=1):
            parts.extend(
                [
                    f"### {source.upper()} {index}: `{example.get('conversation_id') or example.get('group_id')}`",
                    "",
                    f"Judge score: model **{example['judge_score_model']:.3f}**, human {example['judge_score_human']:.3f}. Clean handoffs (model): **{example['clean_handoffs_model']}**.",
                    "",
                    "**Prefix (by agent)**",
                    "",
                    _md_agents(example["prefix_by_agent"]),
                    "",
                    "**Human continuation (by agent)**",
                    "",
                    _md_agents(example["human_by_agent"]),
                    "",
                    "**MatrixChat continuation (by agent)**",
                    "",
                    _md_agents(example["model_by_agent"]),
                    "",
                    "<details><summary>Turn view</summary>",
                    "",
                    "**Prefix**",
                    "",
                    _md_turns(example["prefix_turns"]),
                    "",
                    "**Human**",
                    "",
                    _md_turns(example["human_turns"]),
                    "",
                    "**MatrixChat**",
                    "",
                    _md_turns(example["model_turns"]),
                    "",
                    "</details>",
                    "",
                ]
            )
    return "\n".join(parts)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(
            _ROOT
            / "logs/final_runs_20260819_handoff_redef/full_development_t1"
            / "final40_h100_c106_s176106_t1_judged/paired_continuations_judged.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_ROOT / "paper_results_176106_20260819/best_conversations"),
    )
    parser.add_argument("--per-source", type=int, default=5)
    parser.add_argument(
        "--min-clean-handoffs",
        type=int,
        default=2,
        help="Keep only model continuations with at least this many clean floor transfers.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    records = read_jsonl(args.input)
    chosen = select_best(
        records,
        per_source=args.per_source,
        min_clean_handoffs=args.min_clean_handoffs,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(chosen)
    (output / "best_conversations.md").write_text(markdown, encoding="utf-8")
    (output / "best_conversations.json").write_text(
        json.dumps(chosen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    counts = {source: len(items) for source, items in chosen.items()}
    print(json.dumps({"output_dir": str(output.resolve()), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
