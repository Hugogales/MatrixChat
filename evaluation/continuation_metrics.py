"""CPU-only metrics for matrix-conversation continuations.

Matrices use the project's ``[agent][time]`` convention.  Activity values
are interpreted as booleans; token IDs are counted only where activity is
true.  All public return values contain plain Python JSON-compatible types.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np


def _matrix(values: Sequence[Sequence[Any]], name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rectangular [A][T] matrix")
    return array


def _rate(count: int, denominator: int) -> float:
    return float(count / denominator) if denominator else 0.0


def _summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "total": int(array.sum()),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


def _runs(row: np.ndarray) -> list[int]:
    padded = np.pad(row.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [int(end - start) for start, end in zip(starts, ends)]


def lexical_metrics(tokens: Iterable[int]) -> dict[str, Any]:
    """Return distinct-1/2 and repeated-4-gram metrics for one token stream."""
    stream = [int(token) for token in tokens]

    def ngrams(n: int) -> list[tuple[int, ...]]:
        return [tuple(stream[index : index + n]) for index in range(len(stream) - n + 1)]

    bigrams = ngrams(2)
    fourgrams = ngrams(4)
    repeated_fourgrams = sum(count - 1 for count in Counter(fourgrams).values() if count > 1)
    return {
        "total_active_tokens": len(stream),
        "distinct1": _rate(len(set(stream)), len(stream)),
        "distinct2": _rate(len(set(bigrams)), len(bigrams)),
        "repeated4gram_fraction": _rate(repeated_fourgrams, len(fourgrams)),
    }


# Overlapped floor-change at the cut counts as an interruption when the
# reference and listener share 1..MAX_INTERRUPTION_OVERLAP columns before
# the listener holds the floor alone.  MIN_DIRTY_HANDOFF_OVERLAP or more
# shared columns is a dirty (sustained-overlap) handoff.
MAX_INTERRUPTION_OVERLAP = 4
MIN_DIRTY_HANDOFF_OVERLAP = 5


def _last_sole_speaker(prefix: np.ndarray | None) -> int | None:
    if prefix is None:
        return None
    for column in range(prefix.shape[1] - 1, -1, -1):
        speakers = np.flatnonzero(prefix[:, column])
        if speakers.size == 1:
            return int(speakers[0])
    return None


def _no_silent_columns_between(activity: np.ndarray, start: int, end: int) -> bool:
    """Return True when every column in ``[start, end)`` has at least one speaker."""
    if end <= start:
        return True
    return not bool(np.any(activity[:, start:end].sum(axis=0) == 0))


def _boundary_handoff(
    activity: np.ndarray,
    prefix_activity: np.ndarray | None,
    overlap_tolerance: int,
    reference_speaker: int | None = None,
) -> dict[str, Any]:
    reference = (
        int(reference_speaker)
        if reference_speaker is not None
        else _last_sole_speaker(prefix_activity)
    )
    result: dict[str, Any] = {
        "reference_speaker": reference,
        "clean": False,
        "strict": False,
        "dirty": False,
        "interruption": False,
        "first_listener_column": None,
        "first_listener_agent": None,
        "reference_last_activity_column": None,
        "reference_listener_overlap_columns": 0,
        "overlap_columns_before_resolution": 0,
        "listener_resolution_column": None,
        "resolved_speaker": None,
        "genuine_speaker_change": False,
        "silent_gap_before_listener": False,
        "no_handoff": False,
    }
    if reference is None:
        result["no_handoff"] = True
        return result

    reference_columns = np.flatnonzero(activity[reference])
    if reference_columns.size:
        result["reference_last_activity_column"] = int(reference_columns[-1])

    listener_agent = None
    listener_column = None
    for column in range(activity.shape[1]):
        listeners = np.flatnonzero(activity[:, column]).tolist()
        listeners = [int(agent) for agent in listeners if int(agent) != reference]
        if listeners:
            listener_column = column
            listener_agent = listeners[0]
            break
    if listener_agent is None or listener_column is None:
        result["no_handoff"] = True
        return result

    result["first_listener_column"] = int(listener_column)
    result["first_listener_agent"] = int(listener_agent)
    overlap_columns = int(
        np.count_nonzero(activity[reference] & activity[listener_agent])
    )
    result["reference_listener_overlap_columns"] = overlap_columns

    resolution_column = None
    for column in range(listener_column, activity.shape[1]):
        speakers = np.flatnonzero(activity[:, column])
        if speakers.size == 1 and int(speakers[0]) == listener_agent:
            resolution_column = column
            break
    if resolution_column is not None:
        result["listener_resolution_column"] = int(resolution_column)
        result["resolved_speaker"] = int(listener_agent)
        result["genuine_speaker_change"] = True

    transition_overlap = int(
        np.count_nonzero(
            activity[reference, listener_column : resolution_column + 1]
            & activity[listener_agent, listener_column : resolution_column + 1]
        )
        if resolution_column is not None
        else overlap_columns
    )
    result["overlap_columns_before_resolution"] = transition_overlap

    last_ref_before = None
    if listener_column > 0:
        prior = np.flatnonzero(activity[reference, :listener_column])
        if prior.size:
            last_ref_before = int(prior[-1])
    silent_gap = (
        last_ref_before is not None
        and not _no_silent_columns_between(activity, last_ref_before + 1, listener_column)
    )
    result["silent_gap_before_listener"] = silent_gap

    resolved = resolution_column is not None
    clean = resolved and transition_overlap == 0
    interruption = (
        resolved
        and 0 < transition_overlap <= overlap_tolerance
        and not silent_gap
    )
    dirty = transition_overlap >= MIN_DIRTY_HANDOFF_OVERLAP
    result["clean"] = clean
    result["strict"] = clean
    result["interruption"] = interruption
    result["dirty"] = dirty
    result["no_handoff"] = not (clean or interruption or dirty)
    return result


def count_clean_handoffs(
    activity: Sequence[Sequence[Any]],
    prefix_activity: Sequence[Sequence[Any]] | None = None,
    *,
    overlap_tolerance: int = MAX_INTERRUPTION_OVERLAP,
) -> int:
    """Count successive clean floor transfers in a continuation.

    Starts from the last sole prefix speaker (the ranking-event reference)
    and walks resolved handoffs in order.  Each event uses the same
    classifier as ``_boundary_handoff``: clean = resolved listener floor
    with zero reference/listener overlap.  After a resolved transfer the
    listener becomes the next reference.  Unresolved events stop the walk.
    """
    speak = _matrix(activity, "activity", bool)
    prefix = (
        _matrix(prefix_activity, "prefix_activity", bool)
        if prefix_activity is not None
        else None
    )
    reference = _last_sole_speaker(prefix)
    start = 0
    clean_count = 0
    while reference is not None and start < speak.shape[1]:
        event = _boundary_handoff(
            speak[:, start:],
            None,
            overlap_tolerance,
            reference_speaker=reference,
        )
        if not (event["clean"] or event["interruption"] or event["dirty"]):
            break
        if event["clean"]:
            clean_count += 1
        if event["listener_resolution_column"] is None or event["resolved_speaker"] is None:
            break
        start += int(event["listener_resolution_column"]) + 1
        reference = int(event["resolved_speaker"])
    return clean_count


def continuation_metrics(
    token_ids: Sequence[Sequence[int]],
    activity: Sequence[Sequence[Any]],
    *,
    prefix_activity: Sequence[Sequence[Any]] | None = None,
    overlap_tolerance: int = MAX_INTERRUPTION_OVERLAP,
) -> dict[str, Any]:
    """Compute reusable structural and lexical continuation metrics.

    Speaker changes compare the complete active-speaker set in adjacent
    non-silent columns (silence is skipped).  ``interruptions.count`` is a
    continuation-wide run-start tally (unchanged): an inactive agent becomes
    active while another agent was active in the preceding column.  A
    resumption is any speaking-run start after that agent's first run.

    Boundary handoff (the cut / probe event) uses the last sole prefix
    speaker as reference and the first activity by any different agent as
    listener response.  A **clean** handoff is a resolved floor change with
    zero reference/listener overlap; silent columns between them are
    allowed.  An **interruption** is a resolved overlapped floor change
    with 1..``overlap_tolerance`` shared columns and no all-silent column
    between the reference's last pre-listener activity and the listener's
    first column.      A **dirty** handoff is ``MIN_DIRTY_HANDOFF_OVERLAP`` or
    more shared columns (sustained overlap), whether or not it resolves.
    **No handoff** is the residual: none of clean, interruption, or dirty.
    Those four labels are mutually exclusive and exhaustive.
    """
    if overlap_tolerance < 0:
        raise ValueError("overlap_tolerance must be non-negative")
    ids = _matrix(token_ids, "token_ids", np.int64)
    speak = _matrix(activity, "activity", bool)
    if ids.shape != speak.shape:
        raise ValueError("token_ids and activity must have the same shape")
    prefix = None if prefix_activity is None else _matrix(prefix_activity, "prefix_activity", bool)
    if prefix is not None and prefix.shape[0] != speak.shape[0]:
        raise ValueError("prefix_activity must have the same number of agents")

    agents, columns = speak.shape
    active_counts = speak.sum(axis=0)
    silence = int(np.count_nonzero(active_counts == 0))
    exactly_one = int(np.count_nonzero(active_counts == 1))
    overlap = int(np.count_nonzero(active_counts >= 2))
    per_agent_counts = speak.sum(axis=1).astype(int)
    all_runs = [_runs(speak[agent]) for agent in range(agents)]

    non_silent_sets = [
        tuple(int(agent) for agent in np.flatnonzero(speak[:, column]))
        for column in range(columns)
        if active_counts[column] > 0
    ]
    speaker_changes = sum(
        current != previous for previous, current in zip(non_silent_sets, non_silent_sets[1:])
    )

    interruptions = [0] * agents
    resumptions = [0] * agents
    for agent in range(agents):
        starts = np.flatnonzero(speak[agent] & ~np.pad(speak[agent][:-1], (1, 0)))
        resumptions[agent] = max(0, int(starts.size) - 1)
        interruptions[agent] = int(
            sum(
                bool(column > 0)
                and bool(np.delete(speak[:, int(column) - 1], agent).any())
                for column in starts
            )
        )

    # Column-major order preserves conversation time; ties within a column
    # are emitted in agent-index order.
    active_tokens = [
        int(ids[agent, column])
        for column in range(columns)
        for agent in range(agents)
        if speak[agent, column]
    ]
    lexical = lexical_metrics(active_tokens)
    return {
        "num_agents": int(agents),
        "num_columns": int(columns),
        "columns": {
            "silence_count": silence,
            "silence_rate": _rate(silence, columns),
            "exactly_one_count": exactly_one,
            "exactly_one_rate": _rate(exactly_one, columns),
            "overlap_count": overlap,
            "overlap_rate": _rate(overlap, columns),
        },
        "participation": {
            "active_tokens_by_agent": [int(value) for value in per_agent_counts],
            "active_column_rate_by_agent": [_rate(int(value), columns) for value in per_agent_counts],
            "share_of_active_tokens_by_agent": [
                _rate(int(value), int(per_agent_counts.sum())) for value in per_agent_counts
            ],
            "spoke_by_agent": [bool(value) for value in per_agent_counts],
        },
        "speaking_runs": {
            "lengths_by_agent": all_runs,
            "summary_by_agent": [_summary(values) for values in all_runs],
            "summary": _summary([length for values in all_runs for length in values]),
        },
        "speaker_changes": {
            "count": int(speaker_changes),
            "rate": _rate(speaker_changes, max(0, len(non_silent_sets) - 1)),
            "non_silent_columns": len(non_silent_sets),
        },
        "boundary_handoff": _boundary_handoff(speak, prefix, overlap_tolerance),
        "interruptions": {
            "count": int(sum(interruptions)),
            "by_agent": interruptions,
        },
        "resumptions": {
            "count": int(sum(resumptions)),
            "by_agent": resumptions,
        },
        **lexical,
    }


compute_continuation_metrics = continuation_metrics


def recompute_record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Replace stored human/model metrics from lossless token/activity tensors."""
    prefix_activity = (record.get("prefix") or {}).get("activity")
    for side in ("human", "model"):
        part = record.get(side) or {}
        token_ids = part.get("token_ids")
        activity = part.get("activity")
        if token_ids is None or activity is None:
            continue
        part["metrics"] = continuation_metrics(
            token_ids, activity, prefix_activity=prefix_activity
        )
    return record

