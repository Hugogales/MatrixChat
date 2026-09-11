"""Dependency-light paired statistics with JSON-safe results."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _pairs(
    before: Sequence[float], after: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(before, dtype=float)
    right = np.asarray(after, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("paired samples must be one-dimensional and equally sized")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("paired samples must contain only finite values")
    return left, right


def paired_bootstrap_ci(
    before: Sequence[float],
    after: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile bootstrap CI for the paired mean difference ``after-before``."""
    left, right = _pairs(before, after)
    if left.size == 0:
        raise ValueError("paired samples must not be empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    differences = right - left
    rng = np.random.default_rng(seed)
    # Chunking avoids allocating n_resamples*n pairs for large evaluations.
    means: list[np.ndarray] = []
    remaining = n_resamples
    while remaining:
        size = min(remaining, max(1, 1_000_000 // differences.size))
        indices = rng.integers(0, differences.size, size=(size, differences.size))
        means.append(differences[indices].mean(axis=1))
        remaining -= size
    bootstrap = np.concatenate(means)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return {
        "n": int(differences.size),
        "mean_difference": float(differences.mean()),
        "confidence": float(confidence),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def exact_mcnemar(
    before: Sequence[Any], after: Sequence[Any]
) -> dict[str, Any]:
    """Two-sided exact McNemar test using the discordant-pair binomial law."""
    left, right = _pairs(
        [float(bool(value)) for value in before],
        [float(bool(value)) for value in after],
    )
    before_only = int(np.count_nonzero((left == 1) & (right == 0)))
    after_only = int(np.count_nonzero((left == 0) & (right == 1)))
    discordant = before_only + after_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(before_only, after_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "n": int(left.size),
        "before_only": before_only,
        "after_only": after_only,
        "discordant": discordant,
        "p_value": float(p_value),
    }


def _average_ranks(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    tie_sizes: list[int] = []
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        if end - start > 1:
            tie_sizes.append(end - start)
        start = end
    return ranks, tie_sizes


def wilcoxon_signed_rank(
    before: Sequence[float],
    after: Sequence[float],
    *,
    exact_max_n: int = 20,
) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test for ``after-before``.

    Zero differences are dropped.  For at most ``exact_max_n`` nonzero
    pairs, the exact random-sign distribution is enumerated, including
    average ranks for tied magnitudes.  Larger samples use a continuity-
    corrected normal approximation whose variance directly uses the tied
    average ranks.
    """
    left, right = _pairs(before, after)
    differences = right - left
    differences = differences[differences != 0]
    n = int(differences.size)
    if n == 0:
        return {
            "n": 0,
            "zero_differences": int(left.size),
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "degenerate",
        }

    ranks, _ = _average_ranks(np.abs(differences))
    w_plus = float(ranks[differences > 0].sum())
    total_rank = float(ranks.sum())
    w_minus = total_rank - w_plus
    statistic = min(w_plus, w_minus)
    mean = total_rank / 2.0

    if n <= exact_max_n:
        observed_distance = abs(w_plus - mean)
        extreme = 0
        for signs in product((0, 1), repeat=n):
            candidate = float(ranks[np.asarray(signs, dtype=bool)].sum())
            if abs(candidate - mean) >= observed_distance - 1e-12:
                extreme += 1
        p_value = extreme / (2**n)
        method = "exact"
    else:
        variance = float(np.square(ranks).sum() / 4.0)
        distance = max(0.0, abs(w_plus - mean) - 0.5)
        z = distance / math.sqrt(variance)
        p_value = math.erfc(z / math.sqrt(2.0))
        method = "normal_approximation"

    return {
        "n": n,
        "zero_differences": int(left.size - n),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": float(statistic),
        "p_value": float(min(1.0, p_value)),
        "method": method,
    }


def paired_t_test(before: Sequence[float], after: Sequence[float]) -> dict[str, Any]:
    """Two-sided paired t-test for ``after-before``.

    This is a supplementary parametric check.  Primary inference for
    clustered continuation metrics remains Wilcoxon / McNemar because
    rates are bounded, often zero-inflated, and need not be Gaussian.
    For ``n >= 2`` the p-value uses a normal approximation to Student's t
    (accurate once the cluster count is more than a few dozen).
    """
    left, right = _pairs(before, after)
    differences = right - left
    n = int(differences.size)
    if n < 2:
        return {
            "n": n,
            "t_statistic": None,
            "df": max(0, n - 1),
            "p_value": 1.0,
            "method": "degenerate",
        }
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    if sd == 0.0:
        return {
            "n": n,
            "t_statistic": 0.0 if mean == 0.0 else None,
            "df": n - 1,
            "p_value": 1.0 if mean == 0.0 else 0.0,
            "method": "degenerate",
        }
    t_statistic = mean / (sd / math.sqrt(n))
    p_value = math.erfc(abs(t_statistic) / math.sqrt(2.0))
    return {
        "n": n,
        "t_statistic": float(t_statistic),
        "df": n - 1,
        "p_value": float(min(1.0, p_value)),
        "method": "normal_approximation_to_t",
    }


def paired_effect_summary(
    before: Sequence[float], after: Sequence[float]
) -> dict[str, Any]:
    """Descriptive paired effects, consistently oriented as ``after-before``."""
    left, right = _pairs(before, after)
    differences = right - left
    n = int(differences.size)
    if n == 0:
        return {
            "n": 0,
            "before_mean": None,
            "after_mean": None,
            "mean_difference": None,
            "median_difference": None,
            "difference_sd": None,
            "cohens_dz": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }
    sd = float(differences.std(ddof=1)) if n > 1 else 0.0
    mean_difference = float(differences.mean())
    before_mean = float(left.mean())
    relative = (
        100.0 * mean_difference / abs(before_mean) if before_mean != 0.0 else None
    )
    paired_relative = []
    for human_value, difference in zip(left, differences):
        if human_value != 0.0:
            paired_relative.append(100.0 * float(difference) / abs(float(human_value)))
    return {
        "n": n,
        "before_mean": before_mean,
        "after_mean": float(right.mean()),
        "mean_difference": mean_difference,
        "mean_difference_percentage_points": 100.0 * mean_difference,
        "mean_relative_percent_difference": relative,
        "mean_paired_relative_percent_difference": (
            float(np.mean(paired_relative)) if paired_relative else None
        ),
        "median_difference": float(np.median(differences)),
        "difference_sd": sd,
        "cohens_dz": float(mean_difference / sd) if sd > 0.0 else None,
        "wins": int(np.count_nonzero(differences > 0)),
        "ties": int(np.count_nonzero(differences == 0)),
        "losses": int(np.count_nonzero(differences < 0)),
    }


def summarize_by_source_fraction(
    records: Iterable[Mapping[str, Any]],
    *,
    before_key: str = "before",
    after_key: str = "after",
    source_key: str = "source",
    fraction_key: str = "fraction",
) -> dict[str, Any]:
    """Group paired records by source/fraction and return effect summaries."""
    groups: defaultdict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record[source_key], record[fraction_key])].append(record)

    rows = []
    for (source, fraction), group in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        summary = paired_effect_summary(
            [float(row[before_key]) for row in group],
            [float(row[after_key]) for row in group],
        )
        rows.append(
            {
                "source": _json_scalar(source),
                "fraction": _json_scalar(fraction),
                **summary,
            }
        )
    return {
        "group_count": len(rows),
        "sources": sorted({str(row["source"]) for row in rows}),
        "groups": rows,
    }


summarize_groups = summarize_by_source_fraction

