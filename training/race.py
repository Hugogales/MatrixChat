"""Robust, conservative diagnostics for the autonomous training race."""

from __future__ import annotations

import math
import statistics
from typing import Dict, Iterable, List, Optional


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else float("nan")


def mad(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    center = median(values)
    return median(abs(value - center) for value in values)


def theil_sen(values: Iterable[float]) -> float:
    values = list(values)
    slopes = [
        (values[j] - values[i]) / (j - i)
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    return median(slopes) if slopes else 0.0


def validation_rows(records: List[dict]) -> List[dict]:
    return [record for record in records if record.get("val_loss") is not None]


def _component(row: dict, key: str) -> Optional[float]:
    value = row.get("val_components", {}).get(key)
    return float(value) if value is not None else None


def topology_error(row: dict) -> float:
    components = row.get("val_components", {})
    pairs = (
        ("speak_rate", "predicted_speak_rate"),
        ("silence_rate", "predicted_silence_rate"),
        ("overlap_rate", "predicted_overlap_rate"),
        ("exactly_one_rate", "predicted_exactly_one_rate"),
    )
    return sum(
        abs(float(components.get(truth, 0.0)) - float(components.get(predicted, 0.0)))
        for truth, predicted in pairs
    )


def meaningful_delta(values: List[float]) -> float:
    if not values:
        return float("inf")
    best = min(values)
    differences = [
        values[index] - values[index - 1] for index in range(1, len(values))
    ]
    return max(0.005 * abs(best), 2.0 * (mad(differences[-4:]) if differences else 0.0))


def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    ``successes`` may be fractional when reconstructing counts from a saved
    rate. The interval is still a useful conservative approximation and avoids
    promoting a candidate because of one lucky success in a tiny probe.
    """
    if total <= 0:
        return 0.0, 1.0
    proportion = min(max(float(successes) / float(total), 0.0), 1.0)
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            (proportion * (1.0 - proportion) / total)
            + (z2 / (4.0 * total * total))
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _rate_lcb(row: dict, key: str, total: int) -> float:
    rate = float(row.get(key, 0.0) or 0.0)
    return wilson_interval(rate * total, total)[0]


def _behavior_bucket(row: dict, total: int) -> float:
    """Confidence-aware behavioral score for one broad-sweep bucket.

    Silence receives no credit. Overlap receives only the small generic
    listener-response term; it cannot masquerade as a handoff.
    """
    listener_rate = float(
        row.get(
            "listener_response_rate",
            1.0 - float(row.get("no_listener_response_rate", 1.0) or 1.0),
        )
    )
    nonoverlap_listener_rate = float(
        row.get(
            "nonoverlap_listener_rate",
            max(0.0, listener_rate - float(row.get("overlap_rate", 0.0) or 0.0)),
        )
    )
    interruption_rate = row.get("interruption_handoff_rate")
    if interruption_rate is None:
        interruption_rate = float(row.get("dirty_handoff_rate", 0.0) or 0.0)
    expanded = {
        **row,
        "listener_response_rate": listener_rate,
        "nonoverlap_listener_rate": nonoverlap_listener_rate,
        "interruption_handoff_rate": interruption_rate,
    }
    overlap = float(row.get("overlap_rate", 0.0) or 0.0)
    overlap_penalty = min(max((overlap - 0.10) / 0.25, 0.0), 1.0)
    base = (
        0.40 * _rate_lcb(expanded, "clean_handoff_rate", total)
        + 0.10 * _rate_lcb(expanded, "interruption_handoff_rate", total)
        + 0.15 * _rate_lcb(expanded, "nonoverlap_listener_rate", total)
        + 0.20 * _rate_lcb(expanded, "chain_2plus_rate", total)
        + 0.05 * _rate_lcb(expanded, "chain_2plus_rate_lenient", total)
        + 0.10 * _rate_lcb(expanded, "listener_response_rate", total)
    )
    return base * (1.0 - 0.30 * overlap_penalty)


def probe_quality_score(probe: dict) -> float:
    """Lightweight turn-taking quality composite for the IN-TRAINING probe.

    Unlike :func:`broad_sweep_score` (which needs ``total_trials``/
    ``chain_2plus_rate``-style fields only available from the larger offline
    broad-sweep suite), this scores exactly the fields ``main.py``'s
    ``run_kitchen_sink_probe`` already computes at every ``probe_every`` step
    during training, with no extra generation cost. Used by
    :class:`training.checkpoint.CheckpointManager` to track a "best by actual
    turn-taking behavior" checkpoint alongside the existing "best by
    validation loss" one -- see
    ``.cursor/rules/read-decoded-text-not-just-metrics.mdc`` and the project
    history of validation loss and turn-taking quality diverging (loss
    improving while conversation quality gets worse).

    Higher is better. Not used for training gradients -- purely a checkpoint
    selection / monitoring heuristic, so there is no exploitability concern
    that would apply to a training-time reward.
    """
    clean_handoff = float(probe.get("clean_handoff_rate", 0.0) or 0.0)
    nonoverlap_listener = float(probe.get("nonoverlap_listener_rate", 0.0) or 0.0)
    overlap = float(probe.get("overlap_rate", 0.0) or 0.0)
    interruption = float(probe.get("unwanted_interruption_rate", 0.0) or 0.0)
    repeated = float(probe.get("repeated_4gram_fraction", 0.0) or 0.0)
    return (
        0.45 * clean_handoff
        + 0.20 * nonoverlap_listener
        - 0.20 * overlap
        - 0.10 * interruption
        - max(0.0, repeated - 0.35)
    )


def broad_sweep_score(
    probe: dict,
    *,
    judge: Optional[dict] = None,
    topology_error_value: float = 0.0,
    judge_quality_exponent: float = 0.5,
) -> Dict:
    """Return gated, confidence-aware score components for a broad sweep.

    Content/total loss is deliberately absent: race evidence showed that it
    can improve while generated conversation quality gets worse.
    """
    total = int(probe.get("total_trials", 0) or 0)
    global_behavior = _behavior_bucket(probe, total)
    by_kind = probe.get("by_context_kind") or {}
    bucket_scores = []
    for bucket in by_kind.values():
        bucket_total = int(bucket.get("total_trials", 0) or 0)
        if bucket_total:
            bucket_scores.append(_behavior_bucket(bucket, bucket_total))
    worst_bucket = min(bucket_scores) if bucket_scores else global_behavior
    real_prefix = by_kind.get("real_prefix")
    real_prefix_total = int(real_prefix.get("total_trials", 0) or 0) if real_prefix else 0
    real_prefix_behavior = (
        _behavior_bucket(real_prefix, real_prefix_total) if real_prefix_total else None
    )
    if real_prefix_behavior is not None:
        behavior = (
            0.45 * global_behavior
            + 0.40 * real_prefix_behavior
            + 0.15 * worst_bucket
        )
    else:
        behavior = 0.70 * global_behavior + 0.30 * worst_bucket

    has_diversity = (
        probe.get("distinct_1") is not None
        and probe.get("distinct_2") is not None
        and probe.get("repeated_4gram_fraction") is not None
    )
    distinct_2 = float(probe.get("distinct_2", 0.0) or 0.0)
    repeated = float(probe.get("repeated_4gram_fraction", 0.0) or 0.0)
    diversity = (
        math.sqrt(
            min(max(distinct_2 / 0.75, 0.0), 1.0)
            * min(max((1.0 - repeated) / 0.90, 0.0), 1.0)
        )
        if has_diversity
        else 0.5  # neutral legacy fallback; never gate missing historical fields
    )
    calibration = min(max(1.0 - topology_error_value / 0.50, 0.25), 1.0)

    axes = (
        "turn_taking_naturalness",
        "coherence_topic_relevance",
        "non_degeneracy",
        "responsiveness",
        "human_likeness_continuity",
    )
    judge_lcbs = (judge or {}).get("axis_lcb") or {}
    judge_values = [
        min(max(float(judge_lcbs.get(axis, 3.0)) / 3.0, 0.0), 1.0)
        for axis in axes
    ]
    judge_quality = math.prod(max(value, 1e-6) for value in judge_values) ** (
        1.0 / len(judge_values)
    )
    judge_quality_exponent = float(judge_quality_exponent)
    if not math.isfinite(judge_quality_exponent) or judge_quality_exponent < 0.0:
        raise ValueError("judge_quality_exponent must be finite and non-negative")

    gates = []
    no_response = float(probe.get("no_listener_response_rate", 0.0) or 0.0)
    if no_response > 0.50:
        gates.append("silence")
    chain_2plus = float(probe.get("chain_2plus_rate", 0.0) or 0.0)
    if chain_2plus > 0.50:
        gates.append("chain_hacking")
    if has_diversity and repeated > 0.15:
        gates.append("repetition")
    # Distinct-1 falls mechanically as more trials are concatenated. The old
    # 0.15 gate was calibrated on the 10-prompt small suite and incorrectly
    # pruned broad 576-trial candidates. Use a fixed-coverage-aware floor.
    distinct_1_floor = 0.15 if total <= 100 else 0.03
    if (
        has_diversity
        and float(probe.get("distinct_1", 0.0) or 0.0) < distinct_1_floor
    ):
        gates.append("low_diversity")
    if judge and float(judge_lcbs.get("non_degeneracy", 3.0)) < 1.0:
        gates.append("judge_degeneracy")
    if judge and (
        judge.get("needs_review")
        or not judge.get("schema_valid", True)
        or not judge.get("position_consistent", True)
    ):
        gates.append("judge_needs_review")
    overlap_rate = float(probe.get("overlap_rate", 0.0) or 0.0)
    if overlap_rate > 0.30:
        gates.append("high_overlap")

    raw_score = (
        100.0
        * behavior
        * judge_quality**judge_quality_exponent
        * diversity**0.20
        * calibration**0.10
    )
    score = raw_score
    if gates:
        score = 0.0
    return {
        "score": score,
        "raw_score": raw_score,
        "disqualified": bool(gates),
        "gates": gates,
        "behavior": behavior,
        "global_behavior": global_behavior,
        "worst_context_behavior": worst_bucket,
        "judge_quality": judge_quality,
        "judge_quality_exponent": judge_quality_exponent,
        "diversity": diversity,
        "calibration": calibration,
    }


def broad_sweep_red_flags(probe: dict) -> list[str]:
    """Hard conversation-quality red lines for promotion (not just composite score)."""
    flags: list[str] = []
    if float(probe.get("no_listener_response_rate", 0.0) or 0.0) > 0.50:
        flags.append("silence")
    if float(probe.get("chain_2plus_rate", 0.0) or 0.0) > 0.50:
        flags.append("chain_hacking")
    repeated = probe.get("repeated_4gram_fraction")
    if repeated is not None and float(repeated) > 0.15:
        flags.append("repetition")
    return flags


def promotion_blocked_by_probe(score_row: dict, *, rung: int) -> bool:
    """Block ASHA promotion/finalist when broad-sweep raw rates fail red lines."""
    import json
    from pathlib import Path

    if score_row.get("disqualified"):
        return True
    if rung < 1:
        return False
    probe_path = score_row.get("probe_path")
    if not probe_path:
        return False
    path = Path(probe_path)
    if not path.is_file():
        return False
    probe = json.loads(path.read_text(encoding="utf-8"))
    return bool(broad_sweep_red_flags(probe))


def diagnose_run(
    records: List[dict],
    *,
    probe: Optional[dict] = None,
    leader_best_content: Optional[float] = None,
) -> Dict:
    """Return conservative status/reasons without making external changes."""
    vals = validation_rows(records)
    result = {
        "decision": "continue",
        "flags": [],
        "reasons": [],
        "validation_count": len(vals),
        "latest_step": records[-1].get("step", 0) if records else 0,
    }
    if not records:
        result.update(decision="wait", reasons=["no metrics yet"])
        return result

    numeric_train = [
        value
        for record in records
        for value in (record.get("train_loss"),)
        if value is not None
    ]
    if any(not math.isfinite(float(value)) for value in numeric_train):
        result.update(decision="cancel")
        result["flags"].append("non_finite")
        result["reasons"].append("NaN/Inf training loss")
        return result
    if not vals:
        result.update(decision="wait", reasons=["awaiting first validation"])
        return result

    latest = vals[-1]
    components = latest.get("val_components", {})
    content_values = [
        value for row in vals if (value := _component(row, "content_loss")) is not None
    ]
    result.update({
        "best_val_loss": min(float(row["val_loss"]) for row in vals),
        "best_content_loss": min(content_values) if content_values else None,
        "latest_val_loss": float(latest["val_loss"]),
        "latest_content_loss": _component(latest, "content_loss"),
        "topology_error": topology_error(latest),
    })

    # Activity-policy collapse needs confirmation across two validations.
    if len(vals) >= 2:
        collapse_rows = []
        for row in vals[-2:]:
            c = row.get("val_components", {})
            true_speak = max(float(c.get("speak_rate", 0.0)), 1e-8)
            predicted_speak = float(c.get("predicted_speak_rate", 0.0))
            true_silence = float(c.get("silence_rate", 0.0))
            predicted_silence = float(c.get("predicted_silence_rate", 0.0))
            true_overlap = max(float(c.get("overlap_rate", 0.0)), 1e-8)
            predicted_overlap = float(c.get("predicted_overlap_rate", 0.0))
            exact_gap = float(c.get("exactly_one_rate", 0.0)) - float(
                c.get("predicted_exactly_one_rate", 0.0)
            )
            collapsed = (
                predicted_speak < 0.02
                or predicted_speak > 0.98
                or not 0.5 <= predicted_speak / true_speak <= 1.5
                or (
                    predicted_silence - true_silence > 0.15
                    and predicted_silence > 1.5 * max(true_silence, 1e-8)
                )
                or predicted_overlap - true_overlap > 0.10
                or predicted_overlap > 2.0 * true_overlap
                or exact_gap > 0.15
            )
            collapse_rows.append(collapsed)
        if all(collapse_rows):
            result.update(decision="cancel")
            result["flags"].append("activity_collapse")
            result["reasons"].append("activity topology collapse persisted for two validations")

    if len(content_values) >= 3:
        best = min(content_values)
        if all(value > 1.03 * best for value in content_values[-2:]):
            recent_train = [
                float(row["train_content_loss"])
                for row in records[-10:]
                if row.get("train_content_loss") is not None
            ]
            older_train = [
                float(row["train_content_loss"])
                for row in records[-20:-10]
                if row.get("train_content_loss") is not None
            ]
            if recent_train and older_train and median(recent_train) < 0.95 * median(older_train):
                result["flags"].append("overfit")
                result["reasons"].append("validation regressed while smoothed train content improved")
                result["decision"] = "cancel"

    if len(content_values) >= 4:
        recent = content_values[-4:]
        delta = meaningful_delta(content_values)
        improvement = max(recent) - min(recent)
        relative_slope = abs(theil_sen(recent)) / max(abs(median(recent)), 1e-8)
        if improvement < delta and relative_slope < 0.005:
            result["flags"].append("plateau")
            result["reasons"].append("four-validation content plateau")
            if len(content_values) >= 6:
                result["decision"] = "branch"

        reward_values = [
            value for row in vals[-4:]
            if (value := _component(row, "turn_reward")) is not None
        ]
        if len(reward_values) == 4 and reward_values[-1] - reward_values[0] >= 0.05:
            content_worsening = recent[-1] > 1.03 * min(recent)
            topology_worsening = topology_error(vals[-1]) > topology_error(vals[-4]) + 0.05
            if content_worsening or topology_worsening:
                result["flags"].append("reward_conflict")
                result["reasons"].append("reward rose while content/topology degraded")
                result["decision"] = "branch"

    if (
        leader_best_content is not None
        and result.get("best_content_loss") is not None
        and len(vals) >= 3
        and result["best_content_loss"] > 1.15 * leader_best_content
    ):
        recent_slope = theil_sen(content_values[-3:])
        if recent_slope >= -0.005 * max(median(content_values[-3:]), 1e-8):
            result["flags"].append("underfit")
            result["reasons"].append("flat and >15% behind sweep leader")
            result["decision"] = "branch"

    if probe:
        repeated = float(probe.get("repeated_4gram_fraction", 0.0))
        if repeated > 0.25:
            result["flags"].append("repetition_warning")
            result["reasons"].append(f"repeated 4-gram fraction {repeated:.1%}")
        if repeated > 0.40 and len(vals) >= 2:
            result["decision"] = "cancel"
        silent = float(probe.get("no_listener_response_rate", 0.0) or 0.0)
        overlap = float(probe.get("overlap_rate", 0.0) or 0.0)
        clean = float(probe.get("clean_handoff_rate", 0.0) or 0.0)
        if silent > 0.95:
            result["flags"].append("probe_silence_collapse")
            result["reasons"].append(f"probe listener non-response rate {silent:.1%}")
        if overlap > 0.80:
            result["flags"].append("probe_overlap_collapse")
            result["reasons"].append(f"probe overlap rate {overlap:.1%}")
        if clean <= 0.0 and silent + overlap > 0.95:
            result["flags"].append("no_conversational_success")
        result["probe"] = probe

    if result["topology_error"] > 0.12:
        result["flags"].append("topology_miscalibration")
        result["reasons"].append("aggregate topology-rate error exceeds 12 points")

    return result
