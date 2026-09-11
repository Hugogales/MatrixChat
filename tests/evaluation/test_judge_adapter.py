import json

import pytest

import evaluation.judge_adapter as adapter


def _part(text):
    return {
        "token_ids": [[1, 2], [3, 4]],
        "activity": [[True, True], [False, True]],
        "is_private_mask": [[False, False], [False, False]],
        "agent_visibility": [[True, True], [True, True]],
        "decoded_by_agent": [text, "reply"],
        "cells": [["hello", ""], ["there", "reply"]],
    }


def test_paired_judge_uses_dedicated_prompt_and_scores_both(monkeypatch):
    scores = {
        "human": {
            "turn_taking_naturalness": 3,
            "coherence_topic_relevance": 3,
            "non_degeneracy": 3,
            "responsiveness": 3,
            "human_likeness_continuity": 3,
            "failure_tags": [],
            "notes": "clean",
        },
        "model": {
            "turn_taking_naturalness": 1,
            "coherence_topic_relevance": 2,
            "non_degeneracy": 3,
            "responsiveness": 2,
            "human_likeness_continuity": 1,
            "failure_tags": ["overlap"],
            "notes": "awkward overlap",
        },
    }
    captured = []

    def fake_call(client, model, system, user):
        captured.append({"client": client, "model": model, "system": system, "user": user})
        order = current_order[0]
        payload = {
            "continuation_a": scores[order["continuation_a"]],
            "continuation_b": scores[order["continuation_b"]],
        }
        return "prefix " + json.dumps(payload)

    monkeypatch.setattr(adapter, "call_chat", fake_call)
    trial_ids = {}
    candidate = 0
    while len(trial_ids) < 2:
        trial_id = f"trial-{candidate}"
        order = adapter._presentation_order(trial_id)
        trial_ids.setdefault(order, trial_id)
        candidate += 1

    observed_orders = set()
    for expected_order, trial_id in trial_ids.items():
        current_order = [
            {"continuation_a": expected_order[0], "continuation_b": expected_order[1]}
        ]
        result = adapter.judge_paired_continuations(
            object(),
            "judge-model",
            {
                "trial_id": trial_id,
                "prefix": _part("prefix"),
                "human": _part("human"),
                "model": _part("model"),
            },
        )
        observed_orders.add(tuple(result["presentation_order"].values()))
        assert result["presentation_order"] == current_order[0]
        assert result["human"]["aggregate"] == 1.0
        assert result["model"]["aggregate"] == pytest.approx((12 ** (1 / 5)) / 3)
        assert result["model"]["axes"]["responsiveness"] == 2
        assert result["raw_output"].startswith("prefix ")

    assert observed_orders == {("human", "model"), ("model", "human")}
    assert "same observed prefix" in captured[0]["system"]
    assert "real human" not in captured[0]["system"].lower()
    assert "OBSERVED PREFIX" in captured[0]["user"]
    assert "CONTINUATION A" in captured[0]["user"]
    assert "CONTINUATION B" in captured[0]["user"]
    assert "HUMAN CONTINUATION" not in captured[0]["user"]
    assert "MODEL CONTINUATION" not in captured[0]["user"]


def test_axis_aggregate_is_zero_if_any_axis_fails():
    judgment = {axis: 3 for axis in adapter.AXES}
    judgment[adapter.AXES[0]] = 0
    assert adapter.geometric_axis_aggregate(judgment) == 0.0
