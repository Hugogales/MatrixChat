import json

import pytest

from evaluation.analysis import (
    DEFAULT_METRIC_SPECS,
    analyze_jsonl,
    analyze_records,
    benjamini_hochberg,
    metric_metadata,
    numeric_leaves,
)
from scripts.analysis.analyze_held_out_continuations import main, parse_args


def _side(value, strict=False):
    return {
        "metrics": {
            "num_agents": 4,
            "num_columns": 20,
            "columns": {
                "silence_count": 2,
                "silence_rate": value,
                "exactly_one_count": 15,
                "exactly_one_rate": 1.0 - value,
                "overlap_count": 3,
                "overlap_rate": value / 2,
            },
            "speaking_runs": {
                "summary": {"count": 5, "total": 12, "mean": 2.4 + value, "median": 2 + value},
                "summary_by_agent": [{"mean": 99}],
            },
            "speaker_changes": {"count": 7, "rate": value},
            "boundary_handoff": {
                "strict": strict,
                "interruption": False,
                "dirty": not strict,
                "no_handoff": False,
                "reference_speaker": 3,
                "resolved_speaker": 1,
            },
            "interruptions": {"count": value * 10, "by_agent": [1, 2]},
            "resumptions": {"count": value * 8, "by_agent": [1, 2]},
            "participation": {"share_of_active_tokens_by_agent": [0.5 + value / 2, 0.5 - value / 2]},
            "total_active_tokens": 123,
            "distinct1": 0.8 - value,
            "distinct2": 0.9 - value,
            "repeated4gram_fraction": value,
            "internal": {"denominator": 999, "vector": [1, 2]},
        }
    }


def _record(index, *, source="ami", conversation=None, group="", fraction=0.4, seed=0, human=0.1, model=0.2):
    return {
        "trial_id": f"trial-{index}-{seed}-{fraction}",
        "source": source,
        "conversation_id": conversation or f"conversation-{index}",
        "group_id": group,
        "chunk_index": index,
        "generation_seed": seed,
        "cut": {"fraction": fraction},
        "human": _side(human, strict=False),
        "model": _side(model, strict=True),
        "judge": {
            "human": {
                "aggregate": 0.6,
                "axes": {
                    "turn_taking_naturalness": 2,
                    "coherence_topic_relevance": 2,
                    "non_degeneracy": 2,
                    "responsiveness": 2,
                    "human_likeness_continuity": 2,
                },
                "failure_tags": ["silence"] if index % 2 else [],
            },
            "model": {
                "aggregate": 0.8,
                "axes": {
                    "turn_taking_naturalness": 3,
                    "coherence_topic_relevance": 3,
                    "non_degeneracy": 3,
                    "responsiveness": 3,
                    "human_likeness_continuity": 3,
                },
                "failure_tags": ["overlap"],
            },
        },
    }


def _records():
    return [
        _record(1, source="ami", fraction=0.4),
        _record(2, source="ami", fraction=0.5),
        _record(3, source="meld", fraction=0.4),
        _record(4, source="meld", fraction=0.5, model=0.1),
    ]


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_whitelist_direction_and_bh_exclude_internal_numeric_leaves():
    assert numeric_leaves({"a": [{"b": True}, 2]}) == {"a[0].b": 1.0, "a[1]": 2.0}
    assert metric_metadata("columns.overlap_rate")["direction"] == "context_dependent"
    assert metric_metadata("judge.axes.responsiveness")["direction"] == "higher_is_better"
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])

    summary, stats = analyze_records(_records(), bootstrap_resamples=20)
    assert set(summary["metrics"]) == set(DEFAULT_METRIC_SPECS)
    forbidden = ("num_agents", "num_columns", "denominator", "[0]", "reference_speaker", "by_agent")
    assert not any(token in metric for metric in stats["tests"] for token in forbidden)
    families = {
        row["name"]: row["preregistered_members"]
        for row in stats["multiple_comparison_correction"]["families"]
    }
    assert set(families) == {"primary_behavioral", "judge_quality"}
    assert all(not name.startswith("judge.") for name in families["primary_behavioral"])
    assert all(name.startswith("judge.") for name in families["judge_quality"])


def test_repeated_seeds_and_fractions_do_not_inflate_independent_n_or_significance():
    base = [_record(index, conversation=f"c{index}", human=0.1, model=0.2) for index in range(1, 7)]
    duplicated = [
        {**record, "trial_id": f"{record['trial_id']}-copy-{seed}-{fraction}", "generation_seed": seed, "cut": {"fraction": fraction}}
        for record in base
        for seed in (0, 1, 2)
        for fraction in (0.4, 0.5)
    ]
    _, base_stats = analyze_records(base, bootstrap_resamples=100, seed=7)
    summary, duplicate_stats = analyze_records(duplicated, bootstrap_resamples=100, seed=7)
    base_test = base_stats["tests"]["columns.silence_rate"]
    duplicate_test = duplicate_stats["tests"]["columns.silence_rate"]
    assert summary["record_count"] == 36
    assert summary["independent_cluster_count"] == 6
    assert duplicate_test["raw_record_n"] == 36
    assert duplicate_test["independent_cluster_n"] == 6
    assert duplicate_test["raw_p_value"] == base_test["raw_p_value"]
    assert duplicate_test["bh_adjusted_p_value"] == base_test["bh_adjusted_p_value"]
    assert "supplementary_paired_t" in duplicate_test
    assert summary["metrics"]["columns.silence_rate"]["overall"]["bootstrap_ci"]["n"] == 6
    assert all(row["independent_cluster_n"] == 6 for row in summary["metrics"]["columns.silence_rate"]["by_fraction"])


def test_overlapping_chunks_and_group_ids_cluster_and_binary_test_falls_back():
    records = [
        _record(1, conversation="chunk-a", group="meeting-1", seed=0),
        _record(2, conversation="chunk-b", group="meeting-1", seed=1),
        _record(3, conversation="chunk-c", group="meeting-2", seed=0),
    ]
    summary, stats = analyze_records(records, bootstrap_resamples=20)
    assert summary["independent_cluster_count"] == 2
    strict = stats["tests"]["boundary_handoff.strict"]
    assert strict["independent_cluster_n"] == 2
    assert strict["test"] == "clustered_wilcoxon_signed_rank"
    assert strict["metric_kind"] == "cluster_level_proportion"

    one_per_cluster = [_record(1, conversation="a"), _record(2, conversation="b")]
    _, one_stats = analyze_records(one_per_cluster, bootstrap_resamples=20)
    assert one_stats["tests"]["boundary_handoff.strict"]["test"] == "exact_mcnemar"


def test_analysis_is_deterministic_json_safe_and_reports_missingness():
    records = _records()
    del records[-1]["model"]["metrics"]["columns"]["silence_rate"]
    first, first_stats = analyze_records(records, bootstrap_resamples=50, seed=9)
    second, second_stats = analyze_records(records, bootstrap_resamples=50, seed=9)
    assert first == second
    assert first_stats == second_stats
    silence = first["metrics"]["columns.silence_rate"]
    assert silence["missingness"]["complete_pairs"] == 3
    assert silence["missingness"]["missing_pair"] == 1
    assert silence["overall"]["raw_record_n"] == 3
    assert first["failure_tags"]["records_with_judge"] == 4
    json.dumps(first, allow_nan=False)
    json.dumps(first_stats, allow_nan=False)
    assert "test_protocol" in first_stats
    assert "mean_difference_percentage_points" in first["metrics"]["columns.silence_rate"]["overall"]


def test_jsonl_and_cli_write_atomic_reports_without_plots(tmp_path, capsys):
    input_path = tmp_path / "held-out.jsonl"
    output_dir = tmp_path / "analysis"
    _write_jsonl(input_path, _records())
    summary, stats = analyze_jsonl(input_path, output_dir, bootstrap_resamples=20, seed=3, make_plots=False)
    assert summary["plots"] == []
    assert json.loads((output_dir / "summary.json").read_text())["record_count"] == 4
    assert json.loads((output_dir / "stats.json").read_text()) == stats

    result = main([str(input_path), "--output-dir", str(tmp_path / "cli"), "--bootstrap-resamples", "20", "--no-plots"])
    assert result["record_count"] == 4
    assert result["plot_count"] == 0
    assert "exports" in result
    assert (tmp_path / "cli" / "per_example_metrics.csv").exists()
    assert (tmp_path / "cli" / "per_example_transcripts.jsonl").exists()
    assert '"record_count": 4' in capsys.readouterr().out
    with pytest.raises(SystemExit) as help_exit:
        parse_args(["--help"])
    assert help_exit.value.code == 0


def test_participation_gap_drops_silent_agents_and_is_zero_for_one_speaker():
    from evaluation.analysis import _participation_gap

    two_plus_silent = {
        "participation": {
            "active_tokens_by_agent": [10, 10, 0],
            "share_of_active_tokens_by_agent": [0.5, 0.5, 0.0],
        }
    }
    assert _participation_gap(two_plus_silent) == 0.0
    unbalanced = {
        "participation": {
            "active_tokens_by_agent": [90, 10, 0],
            "share_of_active_tokens_by_agent": [0.9, 0.1, 0.0],
        }
    }
    assert _participation_gap(unbalanced) == pytest.approx(0.8)
    solo = {
        "participation": {
            "active_tokens_by_agent": [20, 0, 0],
            "share_of_active_tokens_by_agent": [1.0, 0.0, 0.0],
        }
    }
    assert _participation_gap(solo) == 0.0
    silent = {
        "participation": {
            "active_tokens_by_agent": [0, 0],
            "share_of_active_tokens_by_agent": [0.0, 0.0],
        }
    }
    assert _participation_gap(silent) == 0.0


def test_exclude_sources_drops_meld_from_cluster_tests(tmp_path):
    input_path = tmp_path / "held-out.jsonl"
    _write_jsonl(input_path, _records())
    summary, stats = analyze_jsonl(
        input_path,
        tmp_path / "no-meld",
        bootstrap_resamples=20,
        seed=3,
        make_plots=False,
        exclude_sources=["meld"],
    )
    assert summary["record_count"] == 2
    assert summary["excluded_sources"] == ["meld"]
    assert stats["excluded_sources"] == ["meld"]
    assert {row["source"] for row in summary["groups"]} == {"ami"}


def test_plot_generation_defaults_are_meaningful_and_graceful(tmp_path):
    pytest.importorskip("matplotlib")
    input_path = tmp_path / "held-out.jsonl"
    _write_jsonl(input_path, _records())
    summary, _ = analyze_jsonl(input_path, tmp_path / "with-plots", bootstrap_resamples=10)
    names = {path.rsplit("/", 1)[-1] for path in summary["plots"]}
    assert "columns.silence_rate_paired_differences.png" in names
    assert "columns.silence_rate_distributions.png" in names
    assert "columns.silence_rate_percent_differences.png" in names
    assert "columns.overlap_rate_source_fraction.png" in names
    assert "cut_fraction_sensitivity.png" in names
    assert "judge_axes.png" in names
    assert "failure_tags.png" in names

    graceful, _ = analyze_jsonl(
        input_path,
        tmp_path / "no-available-plots",
        bootstrap_resamples=10,
        plot_metrics=["not.registered"],
    )
    assert graceful["plots"] == ["%s" % (tmp_path / "no-available-plots" / "plots" / "judge_axes.png"), "%s" % (tmp_path / "no-available-plots" / "plots" / "failure_tags.png")]
