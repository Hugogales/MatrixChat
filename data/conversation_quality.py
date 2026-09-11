"""Ground-truth conversation structure quality scoring for training data.

Scores human/synthetic matrix examples by clean turn-taking topology --
single-speaker columns, moderate handoffs, participation balance, and low
overlap -- rather than raw speaker-change count alone.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


def _topology_rates(
    input_activity_mask: Sequence[Sequence[int]],
    activity_labels: Sequence[Sequence[int]],
    num_agents: int,
    length: int,
    *,
    start_col: int = 0,
) -> Dict[str, float]:
    counts = {
        "silence": 0,
        "same": 0,
        "handoff": 0,
        "acquisition": 0,
        "overlap": 0,
        "total": 0,
    }
    for column in range(start_col, length - 1):
        if not all(activity_labels[agent][column] in (0, 1) for agent in range(num_agents)):
            continue
        previous = [
            agent
            for agent in range(num_agents)
            if bool(input_activity_mask[agent][column])
        ]
        following = [
            agent
            for agent in range(num_agents)
            if activity_labels[agent][column] == 1
        ]
        counts["total"] += 1
        if not following:
            counts["silence"] += 1
        elif len(following) >= 2:
            counts["overlap"] += 1
        elif len(previous) != 1:
            counts["acquisition"] += 1
        elif following[0] == previous[0]:
            counts["same"] += 1
        else:
            counts["handoff"] += 1

    total = max(counts["total"], 1)
    rates = {
        key: counts[key] / total
        for key in ("silence", "same", "handoff", "acquisition", "overlap")
    }
    rates["exactly_one"] = rates["same"] + rates["handoff"] + rates["acquisition"]
    rates["reward_handoff"] = rates["handoff"] + rates["acquisition"]
    return rates


def participation_balance(
    input_activity_mask: Sequence[Sequence[int]],
    num_agents: int,
    length: int,
    *,
    start_col: int = 0,
) -> float:
    """Normalized entropy of per-agent active-column counts in [0, 1]."""
    counts = [0] * num_agents
    for column in range(start_col, length):
        for agent in range(num_agents):
            if input_activity_mask[agent][column]:
                counts[agent] += 1
    total = sum(counts)
    if total <= 0 or num_agents <= 1:
        return 0.0
    probs = [count / total for count in counts if count > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(probability * math.log(probability) for probability in probs)
    max_entropy = math.log(num_agents)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def composite_quality_score(
    rates: Dict[str, float],
    *,
    participation: float,
    speaker_change_density: float,
) -> float:
    """Higher is better. Rewards clean floor control, not overlap-heavy chaos."""
    overlap = rates["overlap"]
    silence = rates["silence"]
    handoff = rates["handoff"]
    exactly_one = rates["exactly_one"]

    handoff_term = min(handoff * 8.0, 1.0)
    change_term = (
        min(speaker_change_density * 12.0, 1.0) if overlap < 0.08 else 0.0
    )

    raw = (
        0.25 * exactly_one
        + 0.20 * handoff_term
        + 0.15 * participation
        + 0.10 * change_term
        - 0.30 * overlap
        - 0.15 * max(0.0, silence - 0.15)
    )
    return max(0.0, min(1.0, raw))


def score_matrix_example(example: dict, *, start_col: int | None = None) -> dict:
    """Return topology rates, participation balance, and composite quality score."""
    num_agents = int(example["num_agents"])
    length = int(example["length"])
    if start_col is None:
        start_col = int(example.get("context_lookback_columns", 0) or 0)

    rates = _topology_rates(
        example["input_activity_mask"],
        example["activity_labels"],
        num_agents,
        length,
        start_col=start_col,
    )
    participation = participation_balance(
        example["input_activity_mask"],
        num_agents,
        length,
        start_col=start_col,
    )
    supervised_cols = max(length - start_col - 1, 1)
    speaker_changes = int(example.get("num_speaker_changes", 0) or 0)
    speaker_change_density = speaker_changes / supervised_cols
    score = composite_quality_score(
        rates,
        participation=participation,
        speaker_change_density=speaker_change_density,
    )
    return {
        "conversation_quality_score": score,
        "conversation_quality_participation": participation,
        "conversation_quality_exactly_one_rate": rates["exactly_one"],
        "conversation_quality_handoff_rate": rates["handoff"],
        "conversation_quality_overlap_rate": rates["overlap"],
        "conversation_quality_silence_rate": rates["silence"],
    }


def attach_quality_fields(example: dict, *, start_col: int | None = None) -> dict:
    """Mutate and return ``example`` with conversation-quality fields attached."""
    example.update(score_matrix_example(example, start_col=start_col))
    example["is_high_quality"] = example["conversation_quality_score"] >= 0.55
    return example
