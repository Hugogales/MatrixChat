import json

import pytest

from evaluation.paired_stats import (
    exact_mcnemar,
    paired_bootstrap_ci,
    paired_effect_summary,
    paired_t_test,
    summarize_by_source_fraction,
    wilcoxon_signed_rank,
)


def test_paired_bootstrap_is_deterministic_and_preserves_pairing():
    first = paired_bootstrap_ci([1, 2, 5], [3, 4, 7], n_resamples=200, seed=17)
    second = paired_bootstrap_ci([1, 2, 5], [3, 4, 7], n_resamples=200, seed=17)
    assert first == second
    assert first["mean_difference"] == 2.0
    assert first["ci_low"] == 2.0
    assert first["ci_high"] == 2.0
    json.dumps(first)


def test_exact_mcnemar_known_discordant_case():
    # One before-only success and five after-only successes.
    result = exact_mcnemar(
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 1],
    )
    assert result["before_only"] == 1
    assert result["after_only"] == 5
    assert result["discordant"] == 6
    assert result["p_value"] == pytest.approx(0.21875)


def test_wilcoxon_exact_known_all_positive_case_and_zeros():
    result = wilcoxon_signed_rank([0, 0, 0, 4], [1, 2, 3, 4])
    assert result["n"] == 3
    assert result["zero_differences"] == 1
    assert result["w_plus"] == 6.0
    assert result["statistic"] == 0.0
    assert result["p_value"] == 0.25
    assert result["method"] == "exact"


def test_wilcoxon_uses_tie_aware_normal_approximation_for_large_n():
    result = wilcoxon_signed_rank([0] * 21, [1] * 21)
    assert result["method"] == "normal_approximation"
    assert 0.0 <= result["p_value"] < 0.001


def test_effect_and_source_fraction_group_summaries():
    effect = paired_effect_summary([1, 2, 3], [2, 2, 1])
    assert effect["wins"] == 1
    assert effect["ties"] == 1
    assert effect["losses"] == 1
    assert effect["mean_difference"] == pytest.approx(-1 / 3)
    assert effect["mean_difference_percentage_points"] == pytest.approx(-100 / 3)
    assert effect["mean_relative_percent_difference"] == pytest.approx(-50 / 3)


def test_paired_t_detects_positive_mean_shift():
    result = paired_t_test([0.0, 0.2, 0.4, 0.1], [0.9, 0.8, 1.1, 0.7])
    assert result["n"] == 4
    assert result["t_statistic"] is not None
    assert result["t_statistic"] > 0
    assert result["p_value"] < 0.05
    json.dumps(result)

    records = [
        {"source": "ami", "fraction": 0.5, "before": 1, "after": 2},
        {"source": "ami", "fraction": 0.5, "before": 2, "after": 4},
        {"source": "meld", "fraction": 1.0, "before": 3, "after": 2},
    ]
    grouped = summarize_by_source_fraction(records)
    assert grouped["group_count"] == 2
    assert grouped["sources"] == ["ami", "meld"]
    assert grouped["groups"][0]["mean_difference"] == 1.5
    assert grouped["groups"][1]["mean_difference"] == -1.0
    json.dumps(grouped)

