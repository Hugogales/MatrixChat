"""Tests for data/conversation_quality.py."""

from data.conversation_quality import attach_quality_fields, composite_quality_score


def _example(length: int = 6, agents: int = 2):
    input_activity_mask = [[0] * length for _ in range(agents)]
    activity_labels = [[-100] * length for _ in range(agents)]
    for column in range(length - 1):
        speaker = column % agents
        input_activity_mask[speaker][column] = 1
        activity_labels[speaker][column] = 1
    return {
        "input_activity_mask": input_activity_mask,
        "activity_labels": activity_labels,
        "num_agents": agents,
        "length": length,
        "num_speaker_changes": 1,
        "context_lookback_columns": 0,
    }


def test_clean_example_scores_high():
    scored = attach_quality_fields(_example())
    assert 0.0 <= scored["conversation_quality_score"] <= 1.0
    assert scored["conversation_quality_overlap_rate"] == 0.0


def test_overlap_penalizes_score():
    clean = attach_quality_fields(_example())["conversation_quality_score"]
    noisy = _example()
    noisy["input_activity_mask"][0][2] = 1
    noisy["input_activity_mask"][1][2] = 1
    noisy["activity_labels"][0][2] = 1
    noisy["activity_labels"][1][2] = 1
    overlap_score = attach_quality_fields(noisy)["conversation_quality_score"]
    assert overlap_score < clean


def test_composite_quality_score_bounds():
    rates = {
        "silence": 0.1,
        "same": 0.7,
        "handoff": 0.15,
        "acquisition": 0.0,
        "overlap": 0.05,
        "exactly_one": 0.85,
        "reward_handoff": 0.15,
    }
    score = composite_quality_score(
        rates,
        participation=0.8,
        speaker_change_density=0.2,
    )
    assert 0.0 <= score <= 1.0
