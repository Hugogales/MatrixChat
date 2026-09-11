"""Publication-oriented analysis of paired held-out continuation records.

Inference is preregistered over an explicit set of outcomes and uses the
conversation (or source-defined group) as the independent sampling unit.
Generation seeds, overlapping chunks, and cut fractions are repeated
measurements, not additional independent observations.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .continuation_metrics import recompute_record_metrics
from .paired_continuation import atomic_write_json, read_jsonl
from .paired_stats import exact_mcnemar, paired_bootstrap_ci, paired_effect_summary, paired_t_test, wilcoxon_signed_rank


def numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    """Recursively flatten finite numeric leaves (diagnostic utility only).

    This function is deliberately not used to select analysis endpoints.
    """
    leaves: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(numeric_leaves(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            leaves.update(numeric_leaves(child, path))
    elif isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
        leaves[prefix] = float(value)
    return leaves


def _path(path: str) -> Callable[[Mapping[str, Any]], Any]:
    keys = path.split(".")

    def extract(metrics: Mapping[str, Any]) -> Any:
        value: Any = metrics
        for key in keys:
            if not isinstance(value, Mapping) or key not in value:
                return None
            value = value[key]
        return value

    return extract


def _participation_gap(metrics: Mapping[str, Any]) -> float | None:
    """Range of active-token shares among agents who speak at least once.

    Never-speaking agents are dropped.  If fewer than two agents speak,
    the gap is 0 (a single speaker is not treated as maximally unbalanced).
    """
    counts = _path("participation.active_tokens_by_agent")(metrics)
    shares = _path("participation.share_of_active_tokens_by_agent")(metrics)
    speaking: list[float] = []
    if (
        isinstance(counts, (list, tuple))
        and isinstance(shares, (list, tuple))
        and len(counts) == len(shares)
        and counts
    ):
        for share, count in zip(shares, counts):
            if not isinstance(share, (int, float)) or not math.isfinite(float(share)):
                continue
            if isinstance(count, (int, float)) and float(count) > 0:
                speaking.append(float(share))
    elif isinstance(shares, (list, tuple)):
        speaking = [
            float(share)
            for share in shares
            if isinstance(share, (int, float))
            and math.isfinite(float(share))
            and float(share) > 0
        ]
    if len(speaking) < 2:
        return 0.0
    return max(speaking) - min(speaking)


def _spec(
    label: str,
    direction: str,
    *,
    extractor: Callable[[Mapping[str, Any]], Any] | None = None,
    family: str = "primary_behavioral",
    binary: bool = False,
    plot: bool = True,
) -> dict[str, Any]:
    return {
        "label": label,
        "direction": direction,
        "difference_definition": "model_minus_human",
        "family": family,
        "binary": binary,
        "plot": plot,
        "extractor": extractor,
    }


# Preregistered inferential whitelist. Counts duplicating a selected rate and
# all dimensions, denominators, agent indices, and array positions are absent.
DEFAULT_METRIC_SPECS: dict[str, dict[str, Any]] = {
    "columns.silence_rate": _spec("Silence Rate", "context_dependent"),
    "columns.exactly_one_rate": _spec("Exactly One Speaker Rate", "higher_is_better"),
    "columns.overlap_rate": _spec("Overlap Rate", "context_dependent"),
    "speaking_runs.summary.mean": _spec("Mean Single-speaker Turn Length", "context_dependent"),
    "speaking_runs.summary.median": _spec("Median Single-speaker Turn Length", "context_dependent"),
    "speaking_runs.summary.count": _spec("Speaking-run Count", "context_dependent"),
    "total_active_tokens": _spec("Conversation Speech Length (active tokens)", "context_dependent"),
    "speaker_changes.rate": _spec("Speaker-change Rate", "context_dependent"),
    "boundary_handoff.strict": _spec("Strict Boundary Handoff", "higher_is_better", binary=True),
    "boundary_handoff.interruption": _spec(
        "Interrupted Boundary Handoff", "context_dependent", binary=True
    ),
    "boundary_handoff.dirty": _spec("Dirty Boundary Handoff", "context_dependent", binary=True),
    "boundary_handoff.no_handoff": _spec("No Boundary Handoff", "context_dependent", binary=True),
    "interruptions.count": _spec("Interruption Count", "context_dependent"),
    "resumptions.count": _spec("Resumption Count", "context_dependent"),
    "participation.active_token_share_gap": _spec(
        "Active-token Participation Gap", "lower_is_better", extractor=_participation_gap
    ),
    "distinct1": _spec("Distinct-1", "higher_is_better"),
    "distinct2": _spec("Distinct-2", "higher_is_better"),
    "repeated4gram_fraction": _spec("Repeated 4-gram Fraction", "lower_is_better"),
    "judge.aggregate": _spec(
        "Judge Aggregate", "higher_is_better", family="judge_quality", extractor=lambda metrics: None
    ),
    "judge.axes.turn_taking_naturalness": _spec(
        "Judge Turn-taking Naturalness", "higher_is_better", family="judge_quality", extractor=lambda metrics: None
    ),
    "judge.axes.coherence_topic_relevance": _spec(
        "Judge Coherence / Topic Relevance", "higher_is_better", family="judge_quality", extractor=lambda metrics: None
    ),
    "judge.axes.non_degeneracy": _spec(
        "Judge Non-degeneracy", "higher_is_better", family="judge_quality", extractor=lambda metrics: None
    ),
    "judge.axes.responsiveness": _spec(
        "Judge Responsiveness", "higher_is_better", family="judge_quality", extractor=lambda metrics: None
    ),
    "judge.axes.human_likeness_continuity": _spec(
        "Judge Human-likeness / Continuity", "higher_is_better", family="judge_quality", extractor=lambda metrics: None
    ),
}

DEFAULT_PLOT_METRICS = (
    "columns.silence_rate",
    "columns.exactly_one_rate",
    "columns.overlap_rate",
    "speaking_runs.summary.mean",
    "speaking_runs.summary.count",
    "total_active_tokens",
    "speaker_changes.rate",
    "participation.active_token_share_gap",
    "distinct2",
    "repeated4gram_fraction",
    "boundary_handoff.strict",
    "boundary_handoff.interruption",
    "boundary_handoff.dirty",
    "boundary_handoff.no_handoff",
    "judge.aggregate",
)


def _is_rate_metric(name: str) -> bool:
    return name.endswith("_rate") or name in {
        "distinct1",
        "distinct2",
        "repeated4gram_fraction",
        "judge.aggregate",
        "boundary_handoff.strict",
        "boundary_handoff.interruption",
        "boundary_handoff.dirty",
        "boundary_handoff.no_handoff",
    } or name.startswith("judge.axes.")


def metric_metadata(name: str) -> dict[str, str]:
    """Return preregistered interpretation metadata."""
    spec = DEFAULT_METRIC_SPECS.get(name)
    if spec is None:
        return {
            "label": name,
            "direction": "context_dependent",
            "difference_definition": "model_minus_human",
            "family": "not_preregistered",
        }
    return {key: str(spec[key]) for key in ("label", "direction", "difference_definition", "family")}


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values in input order."""
    if any(not 0.0 <= float(value) <= 1.0 for value in p_values):
        raise ValueError("p-values must lie in [0, 1]")
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (float(p_values[index]), index))
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        running = min(running, float(p_values[original_index]) * count / (rank_index + 1))
        adjusted[original_index] = min(1.0, running)
    return adjusted


def _seed_for(metric: str, seed: int) -> int:
    digest = hashlib.sha256(metric.encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(digest[:4], "big")) % (2**32)


def _finite(value: Any) -> float | None:
    if isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _record_metric_sides(record: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Extract only registered endpoints; never recursively discover metrics."""
    destinations: tuple[dict[str, float], dict[str, float]] = ({}, {})
    for side_index, side_name in enumerate(("human", "model")):
        metrics = (record.get(side_name) or {}).get("metrics") or {}
        if not isinstance(metrics, Mapping):
            metrics = {}
        for name, spec in DEFAULT_METRIC_SPECS.items():
            if name.startswith("judge."):
                continue
            extractor = spec.get("extractor") or _path(name)
            value = _finite(extractor(metrics))
            if value is not None:
                destinations[side_index][name] = value
        judge_side = (record.get("judge") or {}).get(side_name) or {}
        if isinstance(judge_side, Mapping):
            aggregate = _finite(judge_side.get("aggregate"))
            if aggregate is not None:
                destinations[side_index]["judge.aggregate"] = aggregate
            axes = judge_side.get("axes") or {}
            for name in DEFAULT_METRIC_SPECS:
                if not name.startswith("judge.axes."):
                    continue
                value = _finite(axes.get(name.removeprefix("judge.axes."))) if isinstance(axes, Mapping) else None
                if value is not None:
                    destinations[side_index][name] = value
    return destinations


def _cluster_id(record: Mapping[str, Any]) -> str:
    source = str(record.get("source") or "")
    group_id = str(record.get("group_id") or "").strip()
    conversation_id = str(record.get("conversation_id") or "").strip()
    return f"{source}:{group_id if group_id else conversation_id}"


def _raw_pairs(records: Sequence[Mapping[str, Any]], sides: Sequence[tuple[dict[str, float], dict[str, float]]], metric: str) -> list[dict[str, Any]]:
    pairs = []
    for record, (human, model) in zip(records, sides):
        if metric not in human or metric not in model:
            continue
        fraction = (record.get("cut") or {}).get("fraction")
        pairs.append(
            {
                "cluster_id": _cluster_id(record),
                "source": str(record.get("source") or ""),
                "fraction": _finite(fraction),
                "human": human[metric],
                "model": model[metric],
            }
        )
    return pairs


def _aggregate(pairs: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[tuple(pair[key] for key in keys)].append(pair)
    rows = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        rows.append(
            {
                **dict(zip(keys, key)),
                "human": float(np.mean([float(pair["human"]) for pair in group])),
                "model": float(np.mean([float(pair["model"]) for pair in group])),
                "raw_record_n": len(group),
            }
        )
    return rows


def _effect(rows: Sequence[Mapping[str, Any]], *, seed: int, n_resamples: int) -> dict[str, Any]:
    human = [float(row["human"]) for row in rows]
    model = [float(row["model"]) for row in rows]
    result = paired_effect_summary(human, model)
    result["raw_record_n"] = sum(int(row["raw_record_n"]) for row in rows)
    result["independent_cluster_n"] = len(rows)
    result["bootstrap_unit"] = "cluster"
    result["bootstrap_ci"] = (
        paired_bootstrap_ci(human, model, n_resamples=n_resamples, seed=seed) if rows else None
    )
    return result


def _stratified(pairs: Sequence[Mapping[str, Any]], keys: Sequence[str], *, metric: str, seed: int, n_resamples: int) -> list[dict[str, Any]]:
    outer: defaultdict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        outer[tuple(pair[key] for key in keys)].append(pair)
    result = []
    for group_key, group in sorted(outer.items(), key=lambda item: tuple(map(str, item[0]))):
        aggregation_keys = ("cluster_id", "fraction") if "fraction" in keys else ("cluster_id",)
        rows = _aggregate(group, aggregation_keys)
        result.append(
            {
                **dict(zip(keys, group_key)),
                **_effect(rows, seed=_seed_for(metric + repr(group_key), seed), n_resamples=n_resamples),
            }
        )
    return result


def _failure_tag_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    human: Counter[str] = Counter()
    model: Counter[str] = Counter()
    judged = 0
    for record in records:
        judge = record.get("judge")
        if not isinstance(judge, Mapping):
            continue
        judged += 1
        human.update(map(str, ((judge.get("human") or {}).get("failure_tags") or [])))
        model.update(map(str, ((judge.get("model") or {}).get("failure_tags") or [])))
    return {
        "records_with_judge": judged,
        "records_missing_judge": len(records) - judged,
        "tags": [
            {"tag": tag, "human_count": human[tag], "model_count": model[tag], "model_minus_human": model[tag] - human[tag]}
            for tag in sorted(human.keys() | model.keys())
        ],
    }


def analyze_records(records: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int = 10_000, seed: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create cluster-aware descriptive and preregistered inferential reports."""
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    sides = [_record_metric_sides(record) for record in records]
    metrics: dict[str, Any] = {}
    tests: dict[str, Any] = {}

    for metric, spec in DEFAULT_METRIC_SPECS.items():
        pairs = _raw_pairs(records, sides, metric)
        human_present = sum(metric in human for human, _ in sides)
        model_present = sum(metric in model for _, model in sides)
        cluster_rows = _aggregate(pairs, ("cluster_id",))
        metadata = metric_metadata(metric)
        metrics[metric] = {
            "metadata": metadata,
            "missingness": {
                "total_records": len(records),
                "human_present": human_present,
                "model_present": model_present,
                "complete_pairs": len(pairs),
                "missing_pair": len(records) - len(pairs),
                "independent_clusters_with_pair": len(cluster_rows),
            },
            "overall": _effect(cluster_rows, seed=_seed_for(metric, seed), n_resamples=bootstrap_resamples),
            "by_source": _stratified(pairs, ("source",), metric=metric, seed=seed, n_resamples=bootstrap_resamples),
            "by_fraction": _stratified(pairs, ("fraction",), metric=metric, seed=seed, n_resamples=bootstrap_resamples),
            "by_source_fraction": _stratified(pairs, ("source", "fraction"), metric=metric, seed=seed, n_resamples=bootstrap_resamples),
        }
        if not pairs:
            continue

        exactly_one_binary_pair = bool(spec["binary"]) and all(int(row["raw_record_n"]) == 1 for row in cluster_rows)
        human = [float(row["human"]) for row in cluster_rows]
        model = [float(row["model"]) for row in cluster_rows]
        if exactly_one_binary_pair:
            test = exact_mcnemar(human, model)
            test.update(test="exact_mcnemar", metric_kind="binary")
        else:
            test = wilcoxon_signed_rank(human, model)
            test.update(
                test="clustered_wilcoxon_signed_rank",
                metric_kind="cluster_level_proportion" if spec["binary"] else "cluster_level_mean",
            )
        supplementary = paired_t_test(human, model)
        supplementary.update(test="paired_t", role="supplementary_parametric")
        test["raw_p_value"] = test.pop("p_value")
        test["metadata"] = metadata
        test["raw_record_n"] = len(pairs)
        test["independent_cluster_n"] = len(cluster_rows)
        test["analysis_unit"] = "cluster mean"
        test["supplementary_paired_t"] = supplementary
        tests[metric] = test

    families: dict[str, list[str]] = defaultdict(list)
    for metric, spec in DEFAULT_METRIC_SPECS.items():
        families[str(spec["family"])].append(metric)
    family_reports = []
    for family, preregistered_members in sorted(families.items()):
        preregistered_members = sorted(preregistered_members)
        tested_members = [name for name in preregistered_members if name in tests]
        adjusted = benjamini_hochberg([float(tests[name]["raw_p_value"]) for name in tested_members])
        for name, value in zip(tested_members, adjusted):
            tests[name]["bh_adjusted_p_value"] = value
            tests[name]["bh_family"] = family
            tests[name]["bh_family_size"] = len(tested_members)
        family_reports.append(
            {
                "name": family,
                "preregistered_members": preregistered_members,
                "tested_members": tested_members,
                "test_count": len(tested_members),
            }
        )

    groups = Counter((str(record.get("source") or ""), (record.get("cut") or {}).get("fraction")) for record in records)
    cluster_ids = sorted({_cluster_id(record) for record in records})
    summary = {
        "record_count": len(records),
        "independent_cluster_count": len(cluster_ids),
        "cluster_definition": "source + ':' + (nonempty group_id else conversation_id)",
        "metric_count": len(metrics),
        "difference_definition": "model_minus_human",
        "interpretation_note": "Interpret signs using direction metadata; context-dependent outcomes are not inherently better when larger.",
        "groups": [
            {"source": source, "fraction": fraction, "raw_record_count": count}
            for (source, fraction), count in sorted(groups.items(), key=lambda item: str(item[0]))
        ],
        "metrics": metrics,
        "failure_tags": _failure_tag_summary(records),
    }
    stats = {
        "raw_record_count": len(records),
        "independent_cluster_count": len(cluster_ids),
        "analysis_unit": "cluster mean (cluster+fraction mean for fraction-stratified summaries)",
        "test_protocol": {
            "pairing": "Each development example is cut in half; human and model continuations share the same prefix and the same number of time columns.",
            "independent_unit": "conversation cluster, not raw example",
            "binary_metrics": "exact McNemar on cluster-level binary outcomes when each cluster contributes one record; otherwise clustered Wilcoxon",
            "continuous_metrics_primary": "Wilcoxon signed-rank on cluster means (paired, no normality assumption). Preferred because rates are bounded and often zero-inflated.",
            "continuous_metrics_supplementary": "Paired t-test on the same cluster means, reported but not used for Benjamini-Hochberg selection.",
            "interval": "Percentile paired bootstrap CI for model-minus-human mean difference",
            "multiple_comparison": "Benjamini-Hochberg within each preregistered family (primary_behavioral, judge_quality)",
            "percent_difference": "mean_difference_percentage_points is 100 times the raw difference. mean_relative_percent_difference is 100 * (model-human) / |human mean|.",
            "conversation_length": "Continuation column count is identical by contract. Speech length is total_active_tokens. Average single-person turn length is speaking_runs.summary.mean (columns).",
        },
        "multiple_comparison_correction": {
            "method": "Benjamini-Hochberg within each explicitly preregistered family",
            "families": family_reports,
            "test_count": len(tests),
        },
        "tests": tests,
    }
    return summary, stats


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "metric"


def _atomic_savefig(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    try:
        figure.savefig(temporary, dpi=200, bbox_inches="tight")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def generate_plots(records: Sequence[Mapping[str, Any]], output_dir: str | os.PathLike[str], *, metrics: Sequence[str] | None = None) -> list[str]:
    """Generate cluster-level endpoint, sensitivity, judge, and failure plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    sides = [_record_metric_sides(record) for record in records]
    available = {name for name in DEFAULT_METRIC_SPECS if any(name in human and name in model for human, model in sides)}
    requested = list(metrics) if metrics is not None else list(DEFAULT_PLOT_METRICS)
    selected = [name for name in requested if name in available]
    destination = Path(output_dir)
    written: list[str] = []

    for metric in selected:
        pairs = _raw_pairs(records, sides, metric)
        overall = _aggregate(pairs, ("cluster_id",))
        if not overall:
            continue
        metadata = metric_metadata(metric)
        differences = np.asarray([row["model"] - row["human"] for row in overall], dtype=float)
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.hist(differences, bins=min(20, max(3, int(math.sqrt(len(differences))) + 1)), color="#4472C4", edgecolor="white")
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set(xlabel=f"Model − human ({metadata['label']})", ylabel="Independent clusters", title=f"Paired cluster differences: {metadata['label']}")
        figure.tight_layout()
        path = destination / f"{_safe_name(metric)}_paired_differences.png"
        _atomic_savefig(figure, path)
        plt.close(figure)
        written.append(str(path))

        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        bins = min(20, max(3, int(math.sqrt(len(overall))) + 1))
        axis.hist(
            [row["human"] for row in overall],
            bins=bins,
            alpha=0.55,
            label="Human",
            color="#70AD47",
            edgecolor="white",
        )
        axis.hist(
            [row["model"] for row in overall],
            bins=bins,
            alpha=0.55,
            label="Model",
            color="#4472C4",
            edgecolor="white",
        )
        axis.set(
            xlabel=metadata["label"],
            ylabel="Independent clusters",
            title=f"Human vs model distribution: {metadata['label']}",
        )
        axis.legend(frameon=False)
        figure.tight_layout()
        path = destination / f"{_safe_name(metric)}_distributions.png"
        _atomic_savefig(figure, path)
        plt.close(figure)
        written.append(str(path))

        if _is_rate_metric(metric):
            percent_values = 100.0 * differences
            percent_label = f"Model − human, percentage points ({metadata['label']})"
        else:
            percent_values = np.asarray(
                [
                    100.0 * (row["model"] - row["human"]) / abs(row["human"])
                    if row["human"] != 0.0
                    else np.nan
                    for row in overall
                ],
                dtype=float,
            )
            percent_values = percent_values[np.isfinite(percent_values)]
            percent_label = f"Relative percent difference 100×(model−human)/|human| ({metadata['label']})"
        if percent_values.size:
            figure, axis = plt.subplots(figsize=(6.4, 4.0))
            axis.hist(
                percent_values,
                bins=min(20, max(3, int(math.sqrt(len(percent_values))) + 1)),
                color="#ED7D31",
                edgecolor="white",
            )
            axis.axvline(0.0, color="black", linewidth=1)
            axis.set(
                xlabel=percent_label,
                ylabel="Independent clusters",
                title=f"Paired percent differences: {metadata['label']}",
            )
            figure.tight_layout()
            path = destination / f"{_safe_name(metric)}_percent_differences.png"
            _atomic_savefig(figure, path)
            plt.close(figure)
            written.append(str(path))

        stratified = _aggregate(pairs, ("source", "fraction", "cluster_id"))
        labels = sorted({f"{row['source']}\n{row['fraction']}" for row in stratified})
        if labels:
            figure, axis = plt.subplots(figsize=(max(6.4, len(labels) * 1.1), 4.2))
            human = [[row["human"] for row in stratified if f"{row['source']}\n{row['fraction']}" == label] for label in labels]
            model = [[row["model"] for row in stratified if f"{row['source']}\n{row['fraction']}" == label] for label in labels]
            positions = np.arange(len(labels))
            axis.bar(positions - 0.175, [float(np.mean(values)) for values in human], 0.35, label="Human", color="#70AD47")
            axis.bar(positions + 0.175, [float(np.mean(values)) for values in model], 0.35, label="Model", color="#4472C4")
            axis.set_xticks(positions, labels)
            axis.set(ylabel=metadata["label"], title="Human vs model by source and cut fraction")
            axis.legend(frameon=False)
            figure.tight_layout()
            path = destination / f"{_safe_name(metric)}_source_fraction.png"
            _atomic_savefig(figure, path)
            plt.close(figure)
            written.append(str(path))

    # One compact cut-fraction sensitivity plot across selected primary metrics.
    sensitivity = []
    for metric in selected:
        for row in _aggregate(_raw_pairs(records, sides, metric), ("fraction", "cluster_id")):
            sensitivity.append((metric, row["fraction"], row["model"] - row["human"]))
    if sensitivity:
        figure, axis = plt.subplots(figsize=(7.2, 4.4))
        for metric in selected:
            fractions = sorted({row[1] for row in sensitivity if row[0] == metric and row[1] is not None})
            if fractions:
                axis.plot(fractions, [np.mean([row[2] for row in sensitivity if row[0] == metric and row[1] == fraction]) for fraction in fractions], marker="o", label=metric_metadata(metric)["label"])
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set(xlabel="Cut fraction", ylabel="Mean cluster difference (model − human)", title="Cut-fraction sensitivity")
        axis.legend(frameon=False, fontsize="small")
        figure.tight_layout()
        path = destination / "cut_fraction_sensitivity.png"
        _atomic_savefig(figure, path)
        plt.close(figure)
        written.append(str(path))

    judge_axes = [name for name in DEFAULT_METRIC_SPECS if name.startswith("judge.axes.") and name in available]
    if judge_axes:
        figure, axis = plt.subplots(figsize=(7.2, 4.4))
        positions = np.arange(len(judge_axes))
        human_means, model_means = [], []
        for name in judge_axes:
            rows = _aggregate(_raw_pairs(records, sides, name), ("cluster_id",))
            human_means.append(np.mean([row["human"] for row in rows]))
            model_means.append(np.mean([row["model"] for row in rows]))
        axis.bar(positions - 0.175, human_means, 0.35, label="Human", color="#70AD47")
        axis.bar(positions + 0.175, model_means, 0.35, label="Model", color="#4472C4")
        axis.set_xticks(positions, [metric_metadata(name)["label"].removeprefix("Judge ") for name in judge_axes], rotation=25, ha="right")
        axis.set(ylabel="Judge score", title="Judge axes by continuation type")
        axis.legend(frameon=False)
        figure.tight_layout()
        path = destination / "judge_axes.png"
        _atomic_savefig(figure, path)
        plt.close(figure)
        written.append(str(path))

    failure_rows = _failure_tag_summary(records)["tags"]
    if failure_rows:
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        positions = np.arange(len(failure_rows))
        axis.bar(positions - 0.175, [row["human_count"] for row in failure_rows], 0.35, label="Human", color="#70AD47")
        axis.bar(positions + 0.175, [row["model_count"] for row in failure_rows], 0.35, label="Model", color="#4472C4")
        axis.set_xticks(positions, [row["tag"] for row in failure_rows], rotation=25, ha="right")
        axis.set(ylabel="Tagged records", title="Judge failure-tag comparison")
        axis.legend(frameon=False)
        figure.tight_layout()
        path = destination / "failure_tags.png"
        _atomic_savefig(figure, path)
        plt.close(figure)
        written.append(str(path))
    return written


def filter_records(
    records: Sequence[Mapping[str, Any]],
    *,
    exclude_sources: Sequence[str] | None = None,
) -> list[Mapping[str, Any]]:
    """Drop sources that should not enter cluster-weighted tests or paper means."""
    excluded = {str(name).strip().lower() for name in (exclude_sources or []) if str(name).strip()}
    if not excluded:
        return list(records)
    return [
        record
        for record in records
        if str(record.get("source") or "").strip().lower() not in excluded
    ]


def analyze_jsonl(
    input_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    bootstrap_resamples: int = 10_000,
    seed: int = 0,
    plot_metrics: Sequence[str] | None = None,
    make_plots: bool = True,
    recompute_metrics: bool = False,
    exclude_sources: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read records and atomically write summary.json and stats.json."""
    records = read_jsonl(input_path)
    if not records:
        raise ValueError(f"no records found in {input_path}")
    if recompute_metrics:
        records = [recompute_record_metrics(record) for record in records]
    records = filter_records(records, exclude_sources=exclude_sources)
    if not records:
        raise ValueError("no records remain after source filters")
    summary, stats = analyze_records(records, bootstrap_resamples=bootstrap_resamples, seed=seed)
    excluded = sorted({str(name).strip().lower() for name in (exclude_sources or []) if str(name).strip()})
    summary["excluded_sources"] = excluded
    stats["excluded_sources"] = excluded
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary["plots"] = generate_plots(records, destination / "plots", metrics=plot_metrics) if make_plots else []
    atomic_write_json(destination / "summary.json", summary)
    atomic_write_json(destination / "stats.json", stats)
    return summary, stats


__all__ = [
    "DEFAULT_METRIC_SPECS",
    "DEFAULT_PLOT_METRICS",
    "analyze_jsonl",
    "analyze_records",
    "filter_records",
    "benjamini_hochberg",
    "generate_plots",
    "metric_metadata",
    "numeric_leaves",
]
