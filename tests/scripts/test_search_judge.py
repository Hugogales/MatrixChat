import pytest

from scripts.search.judge import (
    AXES,
    judge_broad_sweep,
    parse_json_object,
    render_trial,
    stratified_trials,
    validate_judgment,
)
from scripts.search.strategist import validate_changes
from scripts.search.calibrate_judge import tag_macro_f1, weighted_kappa


def test_parse_fenced_json_and_validate():
    parsed = parse_json_object(
        '```json\n{"turn_taking_naturalness":2,"coherence_topic_relevance":3,'
        '"non_degeneracy":2,"responsiveness":2,"human_likeness_continuity":3,'
        '"failure_tags":[],"notes":"ok"}\n```'
    )
    clean = validate_judgment(parsed)
    assert clean["responsiveness"] == 2
    assert clean["human_likeness_continuity"] == 3


def test_invalid_judge_axis_rejected():
    with pytest.raises(ValueError):
        validate_judgment(
            {
                "turn_taking_naturalness": 8,
                "coherence_topic_relevance": 3,
                "non_degeneracy": 3,
                "responsiveness": 3,
                "human_likeness_continuity": 3,
                "failure_tags": [],
            }
        )


def test_missing_human_likeness_continuity_rejected():
    with pytest.raises(KeyError):
        validate_judgment(
            {
                "turn_taking_naturalness": 3,
                "coherence_topic_relevance": 3,
                "non_degeneracy": 3,
                "responsiveness": 3,
                "failure_tags": [],
            }
        )


def test_broad_judge_bootstraps_all_five_axes(monkeypatch):
    monkeypatch.setattr("scripts.search.judge.make_client", lambda _: object())
    monkeypatch.setattr(
        "scripts.search.judge.judge_trial",
        lambda client, model, trial: {
            **{axis: 2 for axis in AXES},
            "failure_tags": [],
            "notes": "plausible continuation",
        },
    )
    result = judge_broad_sweep(
        {"results": [{"context_kind": "cold", "num_agents": 2}]},
        max_trials=1,
    )
    assert set(result["axis_lcb"]) == set(AXES)
    assert result["axis_lcb"]["human_likeness_continuity"] == pytest.approx(2.0)
    assert result["rubric_version"] == 3


def test_render_trial_includes_time_and_decoded_agents():
    text = render_trial(
        {
            "prompt": "thoughts?",
            "context_kind": "cold",
            "num_agents": 2,
            "overlap": False,
            "listener_spoke": True,
            "clean_handoff": True,
            "chain": [
                {
                    "matrix_cells": [[".", "I"], [".", " agree"]],
                    "outputs": ["", "I agree"],
                }
            ],
        }
    )
    assert "time | agent0 | agent1" in text
    assert "agent1: 'I agree'" in text


def test_strategist_changes_are_bounded_and_controlled():
    assert validate_changes({"activity_pos_weight": 0.8}) == {
        "activity_pos_weight": 0.8
    }
    with pytest.raises(ValueError):
        validate_changes(
            {
                "activity_pos_weight": 0.8,
                "overlap_tau": 2,
                "batch_size": 2,
            }
        )
    with pytest.raises(ValueError):
        validate_changes({"shell_command": "rm -rf /"})


def test_stratified_trials_caps_output():
    rows = []
    for index in range(20):
        rows.append(
            {
                "context_kind": "cold" if index % 2 else "real_prefix",
                "num_agents": 2 + index % 3,
                "listener_spoke": index % 4 == 0,
                "overlap": index % 5 == 0,
            }
        )
    selected = stratified_trials({"results": rows}, max_trials=7, seed=1)
    assert len(selected) == 7


def test_weighted_kappa_and_tag_f1():
    assert weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert weighted_kappa([0, 0, 3, 3], [3, 3, 0, 0]) < 0.0
    scores = tag_macro_f1(
        [["silence"], ["overlap"], []],
        [["silence"], ["overlap"], []],
    )
    assert scores["macro_f1"] == pytest.approx(1.0)

