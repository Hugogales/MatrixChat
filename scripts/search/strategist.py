"""Bounded Llama experiment proposals for the month search.

The strategist cannot submit jobs or execute code. It returns schema-validated
parameter overrides; the controller may allocate at most 20% of new candidates
to these proposals.
"""

from __future__ import annotations

import json
from typing import Any

from scripts.search.judge import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    call_chat,
    make_client,
    parse_json_object,
)

ALLOWED = {
    "activity_pos_weight": (0.65, 0.95),
    "overlap_base_weight": (0.2, 3.0),
    "overlap_max_weight": (1.0, 8.0),
    "overlap_grace": (0, 3),
    "overlap_tau": (0.5, 10.0),
    "handoff_bonus_weight": (0.0, 3.0),
    "lambda_activity": (0.25, 1.5),
    "lambda_reward": (0.01, 0.5),
    "gradient_accumulation_steps": {1, 2, 4, 8, 16},
    "batch_size": {1, 2, 4},
    "context_lookback_columns": {64, 128, 192},
    "lora_r": {16, 32, 64},
    "max_flat_len": {1536, 2048, 3072},
}

SYSTEM = """You are an experiment strategist for multi-agent turn-taking training.
Propose controlled, falsifiable experiments. Do not optimize content loss:
actual conversation behavior is primary. Avoid changing more than two
hyperparameters in one experiment unless you also specify a matched control.
Return JSON only. You cannot execute code or schedule jobs."""


def compact_state(state: dict, limit: int = 12) -> list[dict]:
    rows = []
    for candidate in state.get("candidates", {}).values():
        scores = candidate.get("scores", {})
        latest = scores.get(str(candidate.get("rung", 0))) or scores.get(
            str(max(0, candidate.get("rung", 0) - 1))
        )
        if not latest:
            continue
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "rung": candidate["rung"],
                "status": candidate["status"],
                "score": latest.get("score"),
                "behavior": latest.get("behavior"),
                "diversity": latest.get("diversity"),
                "gates": latest.get("gates"),
                "config": {
                    key: candidate["config"].get(key)
                    for key in ALLOWED
                    if key in candidate["config"]
                },
            }
        )
    rows.sort(key=lambda row: row.get("score") or -1, reverse=True)
    return rows[:limit]


def validate_changes(changes: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(changes, dict) or not changes:
        raise ValueError("changes must be a non-empty object")
    if len(changes) > 2:
        raise ValueError("strategist proposal changes more than two parameters")
    clean = {}
    for key, value in changes.items():
        allowed = ALLOWED.get(key)
        if allowed is None:
            raise ValueError(f"unsupported strategist parameter: {key}")
        if isinstance(allowed, tuple):
            numeric = float(value)
            if not allowed[0] <= numeric <= allowed[1]:
                raise ValueError(f"{key} outside allowed range")
            clean[key] = int(numeric) if all(isinstance(v, int) for v in allowed) else numeric
        elif value not in allowed:
            raise ValueError(f"{key} value not allowed")
        else:
            clean[key] = value
    return clean


def propose(
    state: dict,
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> dict:
    summary = compact_state(state)
    user = f"""Here are recent controlled experiments:
{json.dumps(summary, indent=2)}

Return one proposal:
{{
  "hypothesis": "why this should improve clean, non-overlapping listener response",
  "changes": {{"one_parameter": "value", "optional_second": "value"}},
  "control_candidate_id": "existing candidate or null",
  "expected_signal": "which behavioral probe/broad metric should move",
  "early_stop_condition": "what falsifies it",
  "risk": "likely failure mode"
}}"""
    client = make_client(base_url)
    raw = call_chat(client, model, SYSTEM, user)
    payload = parse_json_object(raw)
    proposal = {
        "hypothesis": str(payload["hypothesis"])[:1000],
        "changes": validate_changes(payload["changes"]),
        "control_candidate_id": payload.get("control_candidate_id"),
        "expected_signal": str(payload["expected_signal"])[:1000],
        "early_stop_condition": str(payload["early_stop_condition"])[:1000],
        "risk": str(payload["risk"])[:1000],
        "raw": raw,
        "model": model,
    }
    return proposal

