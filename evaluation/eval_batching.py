"""Batching helpers for held-out continuation evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class EvalTrialSpec:
    trial_id: str
    entry: Mapping[str, Any]
    row: Mapping[str, Any]
    cut: Mapping[str, Any]
    seed: int


@dataclass
class EvalBatch:
    specs: list[EvalTrialSpec]
    bucket_key: tuple[Any, ...]


def _visibility_signature(visibility: Any) -> str:
    if visibility is None:
        return "public"
    return hashlib.sha256(
        json.dumps(visibility, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def eval_bucket_key(
    entry: Mapping[str, Any],
    row: Mapping[str, Any],
    cut: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[Any, ...]:
    length = int(entry["length"])
    column = int(cut["column"])
    return (
        int(entry["num_agents"]),
        length - column,
        float(cut["fraction"]),
        _visibility_signature(row.get("agent_visibility")),
        int(seed),
    )


def bucket_eval_specs(
    specs: Sequence[EvalTrialSpec],
    *,
    batch_size: int,
) -> list[EvalBatch]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    buckets: dict[tuple[Any, ...], list[EvalTrialSpec]] = {}
    for spec in specs:
        key = eval_bucket_key(spec.entry, spec.row, spec.cut, seed=spec.seed)
        buckets.setdefault(key, []).append(spec)

    batches: list[EvalBatch] = []
    for key, bucket_specs in buckets.items():
        for start in range(0, len(bucket_specs), batch_size):
            batches.append(EvalBatch(specs=bucket_specs[start : start + batch_size], bucket_key=key))
    return batches


def collate_prefix_batch(
    specs: Sequence[EvalTrialSpec],
    *,
    placeholder_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not specs:
        raise ValueError("collate_prefix_batch requires at least one spec")

    agents = int(specs[0].entry["num_agents"])
    continuation_columns = int(specs[0].entry["length"]) - int(specs[0].cut["column"])
    for spec in specs[1:]:
        if int(spec.entry["num_agents"]) != agents:
            raise ValueError("all specs in a batch must share num_agents")
        spec_cont = int(spec.entry["length"]) - int(spec.cut["column"])
        if spec_cont != continuation_columns:
            raise ValueError("all specs in a batch must share continuation_columns")

    prefix_lengths = [int(spec.cut["column"]) for spec in specs]
    max_prefix = max(prefix_lengths)
    batch_size = len(specs)

    prefix_ids = []
    prefix_activity = []
    prefix_private = []
    visibility = []
    for spec, prefix_len in zip(specs, prefix_lengths):
        row = spec.row
        length = int(spec.entry["length"])
        ids = _matrix(row, "input_ids", agents, length, placeholder_token_id)
        activity = _matrix(row, "input_activity_mask", agents, length, False)
        private = _matrix(row, "is_private_mask", agents, length, False)
        vis = row.get("agent_visibility")
        if vis is None:
            vis = [[True] * agents for _ in range(agents)]
        vis = [[bool(value) for value in values] for values in vis]
        if len(vis) != agents or any(len(values) != agents for values in vis):
            raise ValueError("agent_visibility must be [A][A]")

        pad = max_prefix - prefix_len
        prefix_ids.append([([placeholder_token_id] * pad) + values[:prefix_len] for values in ids])
        prefix_activity.append([([False] * pad) + values[:prefix_len] for values in activity])
        prefix_private.append([([False] * pad) + values[:prefix_len] for values in private])
        visibility.append(vis)

    return {
        "specs": list(specs),
        "agents": agents,
        "continuation_columns": continuation_columns,
        "prefix_lengths": prefix_lengths,
        "input_ids": torch.tensor(prefix_ids, dtype=torch.long, device=device),
        "input_activity": torch.tensor(prefix_activity, dtype=torch.bool, device=device),
        "input_private": torch.tensor(prefix_private, dtype=torch.bool, device=device),
        "agent_visibility": torch.tensor(visibility, dtype=torch.bool, device=device),
    }


def _matrix(row: Mapping[str, Any], name: str, agents: int, length: int, default: Any):
    values = row.get(name)
    if values is None:
        return [[default] * length for _ in range(agents)]
    if len(values) != agents or any(len(values[row_idx]) != length for row_idx in range(agents)):
        raise ValueError(f"{name} must be [{agents}][{length}]")
    return values
