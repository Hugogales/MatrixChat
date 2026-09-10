import json

import pytest

from evaluation.continuation_metrics import continuation_metrics, lexical_metrics


def test_hand_calculated_column_run_and_transition_metrics():
    activity = [
        [1, 1, 0, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 0, 0],
    ]
    token_ids = [
        [10, 11, 0, 0, 12, 13, 0, 14],
        [0, 20, 21, 0, 0, 22, 0, 0],
    ]

    result = continuation_metrics(token_ids, activity)

    assert result["columns"] == {
        "silence_count": 2,
        "silence_rate": 0.25,
        "exactly_one_count": 4,
        "exactly_one_rate": 0.5,
        "overlap_count": 2,
        "overlap_rate": 0.25,
    }
    assert result["participation"]["active_tokens_by_agent"] == [5, 3]
    assert result["participation"]["share_of_active_tokens_by_agent"] == [0.625, 0.375]
    assert result["speaking_runs"]["lengths_by_agent"] == [[2, 2, 1], [2, 1]]
    assert result["speaking_runs"]["summary"]["mean"] == pytest.approx(1.6)
    assert result["speaker_changes"] == {
        "count": 5,
        "rate": 1.0,
        "non_silent_columns": 6,
    }
    assert result["interruptions"] == {"count": 2, "by_agent": [0, 2]}
    assert result["resumptions"] == {"count": 3, "by_agent": [2, 1]}
    assert result["total_active_tokens"] == 8
    json.dumps(result)


def test_strict_interruption_and_dirty_boundary_handoff():
    prefix = [[1, 1], [0, 0]]

    strict = continuation_metrics(
        [[0, 0, 0], [7, 8, 9]],
        [[0, 0, 0], [1, 1, 1]],
        prefix_activity=prefix,
    )
    handoff = strict["boundary_handoff"]
    assert handoff["reference_speaker"] == 0
    assert handoff["clean"] is True
    assert handoff["strict"] is True
    assert handoff["interruption"] is False
    assert handoff["dirty"] is False
    assert handoff["no_handoff"] is False
    assert handoff["first_listener_agent"] == 1
    assert handoff["reference_last_activity_column"] is None
    assert handoff["listener_resolution_column"] == 0
    assert handoff["genuine_speaker_change"] is True

    gapped = continuation_metrics(
        [[0, 0, 0, 0], [0, 0, 7, 8]],
        [[0, 0, 0, 0], [0, 0, 1, 1]],
        prefix_activity=prefix,
    )
    assert gapped["boundary_handoff"]["clean"] is True
    assert gapped["boundary_handoff"]["strict"] is True
    assert gapped["boundary_handoff"]["interruption"] is False
    assert gapped["boundary_handoff"]["silent_gap_before_listener"] is False

    gapped_after_ref = continuation_metrics(
        [[4, 0, 0, 0], [0, 0, 7, 8]],
        [[1, 0, 0, 0], [0, 0, 1, 1]],
        prefix_activity=prefix,
    )
    assert gapped_after_ref["boundary_handoff"]["clean"] is True
    assert gapped_after_ref["boundary_handoff"]["silent_gap_before_listener"] is True

    interruption = continuation_metrics(
        [[4, 5, 0], [7, 8, 9]],
        [[1, 1, 0], [1, 1, 1]],
        prefix_activity=prefix,
    )
    assert interruption["boundary_handoff"]["strict"] is False
    assert interruption["boundary_handoff"]["interruption"] is True
    assert interruption["boundary_handoff"]["dirty"] is False
    assert interruption["boundary_handoff"]["no_handoff"] is False
    assert interruption["boundary_handoff"]["first_listener_column"] == 0
    assert interruption["boundary_handoff"]["first_listener_agent"] == 1
    assert interruption["boundary_handoff"]["overlap_columns_before_resolution"] == 2

    four_overlap = continuation_metrics(
        [[4, 5, 6, 7, 0], [8, 9, 10, 11, 12]],
        [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
        prefix_activity=prefix,
    )
    assert four_overlap["boundary_handoff"]["interruption"] is True
    assert four_overlap["boundary_handoff"]["dirty"] is False

    dirty = continuation_metrics(
        [[4, 5, 6, 7, 8, 0], [9, 10, 11, 12, 13, 14]],
        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]],
        prefix_activity=prefix,
    )
    assert dirty["boundary_handoff"]["interruption"] is False
    assert dirty["boundary_handoff"]["dirty"] is True
    assert dirty["boundary_handoff"]["overlap_columns_before_resolution"] == 5

    never_resolves = continuation_metrics(
        [[4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15]],
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        prefix_activity=prefix,
    )
    assert never_resolves["boundary_handoff"]["clean"] is False
    assert never_resolves["boundary_handoff"]["interruption"] is False
    assert never_resolves["boundary_handoff"]["dirty"] is True
    assert never_resolves["boundary_handoff"]["no_handoff"] is False


def test_no_listener_is_no_handoff_and_four_labels_partition():
    prefix = [[1, 1], [0, 0]]
    none = continuation_metrics(
        [[4, 5, 6], [0, 0, 0]],
        [[1, 1, 1], [0, 0, 0]],
        prefix_activity=prefix,
    )
    handoff = none["boundary_handoff"]
    assert handoff["clean"] is False
    assert handoff["interruption"] is False
    assert handoff["dirty"] is False
    assert handoff["no_handoff"] is True
    flags = [handoff["strict"], handoff["interruption"], handoff["dirty"], handoff["no_handoff"]]
    assert sum(flags) == 1


def test_reference_resuming_without_overlap_is_still_clean():
    prefix = [[1], [0]]
    result = continuation_metrics(
        [[4, 0, 0, 5], [0, 0, 7, 0]],
        [[1, 0, 0, 1], [0, 0, 1, 0]],
        prefix_activity=prefix,
    )
    assert result["boundary_handoff"]["clean"] is True
    assert result["boundary_handoff"]["interruption"] is False
    assert result["boundary_handoff"]["dirty"] is False


def test_lexical_metrics_have_hand_calculated_denominators():
    result = lexical_metrics([1, 2, 3, 4, 1, 2, 3, 4])
    assert result["total_active_tokens"] == 8
    assert result["distinct1"] == 0.5
    assert result["distinct2"] == pytest.approx(4 / 7)
    assert result["repeated4gram_fraction"] == pytest.approx(1 / 5)


def test_simultaneous_initial_speech_is_not_an_interruption():
    result = continuation_metrics([[1], [2]], [[1], [1]])
    assert result["interruptions"] == {"count": 0, "by_agent": [0, 0]}


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        continuation_metrics([[1, 2]], [[1]])


def test_count_clean_handoffs_chains_resolved_transfers():
    from evaluation.continuation_metrics import count_clean_handoffs

    prefix = [[1], [0], [0]]
    two_clean = [
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    assert count_clean_handoffs(two_clean, prefix) == 2
    monologue = [
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert count_clean_handoffs(monologue, prefix) == 0
    one_then_dirty = [
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    ]
    assert count_clean_handoffs(one_then_dirty, prefix) == 1

