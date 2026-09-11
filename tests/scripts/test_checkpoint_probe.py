"""Unit tests for the multi-turn / private-context probe builder in
scripts/eval/evaluate_checkpoint_suite.py (pure tensor logic, no model needed).
"""

import pytest
import torch

from scripts.eval.evaluate_checkpoint_suite import (
    PRIMED_HANDOFF_TEMPLATES,
    _build_multiturn_inputs,
    _detect_privacy_leak,
    _primed_handoff_turns,
    _score_handoff,
    ngrams,
    repetition_stats,
    run_probe_suite,
)


class _FakeTokenizer:
    """Maps each character to its ordinal so token counts are predictable."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def test_multiturn_builder_places_each_turn_in_its_own_speaker_only_columns():
    tok = _FakeTokenizer()
    turns = [
        {"speaker": 0, "text": "ab", "visible_to": None},
        {"speaker": 1, "text": "xyz", "visible_to": None},
    ]
    input_ids, activity, private, visibility = _build_multiturn_inputs(tok, turns, num_agents=3, device="cpu")

    assert input_ids.shape == (1, 3, 5)
    assert activity.shape == (1, 3, 5)
    assert private is None
    assert visibility is None

    # First 2 columns: only agent 0 active, with its own token ids.
    assert activity[0, 0, :2].tolist() == [True, True]
    assert activity[0, 1, :2].tolist() == [False, False]
    assert activity[0, 2, :2].tolist() == [False, False]
    assert input_ids[0, 0, :2].tolist() == [ord("a"), ord("b")]

    # Last 3 columns: only agent 1 active.
    assert activity[0, 1, 2:].tolist() == [True, True, True]
    assert activity[0, 0, 2:].tolist() == [False, False, False]
    assert input_ids[0, 1, 2:].tolist() == [ord("x"), ord("y"), ord("z")]


def test_multiturn_builder_private_turn_sets_visibility_and_private_mask():
    tok = _FakeTokenizer()
    turns = [
        {"speaker": 0, "text": "s", "visible_to": [0, 1]},
        {"speaker": 2, "text": "p", "visible_to": None},
    ]
    input_ids, activity, private, visibility = _build_multiturn_inputs(tok, turns, num_agents=3, device="cpu")

    assert private is not None and visibility is not None
    assert private.shape == (1, 3, 2)
    # Column 0 (agent 0's private turn) is private; column 1 (agent 2's public turn) is not.
    assert private[0, 0, 0].item() is True
    assert private[0, :, 1].any().item() is False

    # visibility[owner, viewer]: self always visible; agent 0's secret is also
    # visible to agent 1 (as specified), but not to agent 2.
    vis = visibility[0]
    assert bool(vis[0, 0]) is True
    assert bool(vis[0, 1]) is True
    assert bool(vis[0, 2]) is False
    assert bool(vis[1, 1]) is True
    assert bool(vis[2, 2]) is True


def test_multiturn_builder_empty_turns_yields_single_inactive_column():
    tok = _FakeTokenizer()
    input_ids, activity, private, visibility = _build_multiturn_inputs(
        tok, [{"speaker": 0, "text": "", "visible_to": None}], num_agents=2, device="cpu",
    )
    assert input_ids.shape == (1, 2, 1)
    assert not bool(activity.any())
    assert private is None
    assert visibility is None


def test_multiturn_builder_silence_columns_insert_fully_inactive_gap():
    tok = _FakeTokenizer()
    turns = [
        {"speaker": 0, "text": "ab", "visible_to": None},
        {"silence_columns": 3},
        {"speaker": 1, "text": "x", "visible_to": None},
    ]
    input_ids, activity, private, visibility = _build_multiturn_inputs(tok, turns, num_agents=2, device="cpu")

    assert input_ids.shape == (1, 2, 6)  # 2 (agent0) + 3 (silence) + 1 (agent1)
    # Gap columns (indices 2,3,4): nobody active.
    assert not bool(activity[0, :, 2:5].any())
    assert activity[0, 0, :2].tolist() == [True, True]
    assert activity[0, 1, 5].item() is True


def test_multiturn_builder_zero_silence_columns_is_a_no_op():
    tok = _FakeTokenizer()
    turns = [
        {"speaker": 0, "text": "a", "visible_to": None},
        {"silence_columns": 0},
        {"speaker": 1, "text": "b", "visible_to": None},
    ]
    input_ids, activity, private, visibility = _build_multiturn_inputs(tok, turns, num_agents=2, device="cpu")
    assert input_ids.shape == (1, 2, 2)


def test_repetition_stats_detects_repeated_4grams():
    # Two identical 5-token sequences share several overlapping 4-grams.
    ids = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    stats = repetition_stats(ids)
    assert stats["repeated_4gram_fraction"] > 0.0
    assert 0.0 < stats["distinct_1"] <= 1.0


def test_ngrams_basic():
    assert ngrams([1, 2, 3], 2) == [(1, 2), (2, 3)]
    assert ngrams([1], 2) == []


def test_privacy_leak_ignores_a_single_shared_common_token():
    # Regression test: a lone shared token (e.g. a common function word like
    # " are") must NOT be flagged as a leak -- only observed in production
    # when a bystander agent's generation happened to include the word
    # " are" from the secret "you are secretly teammates with agent 1".
    secret_ids = [9330, 525, 41739, 31646, 448, 8315, 220, 16]  # "you are secretly teammates with agent 1"
    bystander_ids = [1112, 704, 879, 1558, 5019, 15207, 1588, 1112, 773, 582, 525, 1112]  # shares only " are"
    leaked = _detect_privacy_leak(secret_ids, secret_owner=0, ids_by_agent=[[], [], bystander_ids])
    assert leaked == []


def test_privacy_leak_detects_verbatim_ngram_reproduction():
    secret_ids = [9330, 525, 41739, 31646, 448, 8315, 220, 16]
    # Agent 2 reproduces 3 consecutive secret tokens verbatim -- a real leak.
    leaking_ids = [1, 2, 41739, 31646, 448, 3]
    leaked = _detect_privacy_leak(secret_ids, secret_owner=0, ids_by_agent=[[], [], leaking_ids])
    assert leaked == [2]


def test_privacy_leak_excludes_the_secret_owner_itself():
    secret_ids = [9330, 525, 41739, 31646, 448, 8315, 220, 16]
    owner_ids = secret_ids  # the owner naturally reproduces its own secret.
    leaked = _detect_privacy_leak(secret_ids, secret_owner=0, ids_by_agent=[owner_ids, []])
    assert leaked == []


def test_primed_handoff_turns_places_agent0_then_agent1_in_order():
    first, second = PRIMED_HANDOFF_TEMPLATES[0]
    turns = _primed_handoff_turns(first, second)
    assert turns[0] == {"speaker": 0, "text": first, "visible_to": None}
    assert turns[1] == {"speaker": 1, "text": second, "visible_to": None}


def test_score_handoff_regression_reference_speaks_once_then_yields():
    # Agent 0 speaks only at step 0, then goes silent; agent 1 starts 3
    # steps later. A gap is still a clean handoff.
    speak = torch.zeros(2, 12, dtype=torch.bool)
    speak[0, 0] = True
    speak[1, 3:9] = True
    score = _score_handoff(speak, "handoff", reference_speaker=0, other_agents=[1])
    assert score.overlap is False
    assert score.overlap_tokens == 0
    assert score.listener_any is True
    assert score.first_listener == 3
    assert score.first_ref == 0
    assert score.last_ref == 0
    assert score.clean_handoff is True
    assert score.interruption_handoff is False
    assert score.dirty_handoff is False


def test_score_handoff_brief_overlap_is_an_interruption():
    speak = torch.zeros(2, 6, dtype=torch.bool)
    speak[0, 0:4] = True
    speak[1, 2:5] = True  # overlaps with agent 0 at steps 2-3
    score = _score_handoff(speak, "handoff", reference_speaker=0, other_agents=[1])
    assert score.overlap is True
    assert score.overlap_tokens == 2
    assert score.clean_handoff is False
    assert score.interruption_handoff is True
    assert score.dirty_handoff is False


def test_score_handoff_five_plus_overlap_is_dirty():
    speak = torch.zeros(2, 8, dtype=torch.bool)
    speak[0, 0:6] = True
    speak[1, 1:7] = True  # overlaps at steps 1-5 (5 tokens)
    score = _score_handoff(speak, "handoff", reference_speaker=0, other_agents=[1])
    assert score.overlap is True
    assert score.overlap_tokens == 5
    assert score.clean_handoff is False
    assert score.interruption_handoff is False
    assert score.dirty_handoff is True


def test_score_handoff_four_overlap_is_still_an_interruption():
    speak = torch.zeros(2, 8, dtype=torch.bool)
    speak[0, 0:5] = True
    speak[1, 1:6] = True  # overlaps at steps 1-4 (4 tokens), then listener sole
    score = _score_handoff(speak, "handoff", reference_speaker=0, other_agents=[1])
    assert score.overlap_tokens == 4
    assert score.interruption_handoff is True
    assert score.dirty_handoff is False


def test_score_handoff_reference_resuming_without_overlap_is_clean():
    speak = torch.zeros(2, 8, dtype=torch.bool)
    speak[0, 0] = True
    speak[1, 2] = True
    speak[0, 4] = True
    score = _score_handoff(speak, "handoff", reference_speaker=0, other_agents=[1])
    assert score.overlap is False
    assert score.overlap_tokens == 0
    assert score.first_listener == 2
    assert score.last_ref == 4
    assert score.clean_handoff is True
    assert score.interruption_handoff is False
    assert score.dirty_handoff is False


def test_score_handoff_no_listener_response_is_not_clean():
    speak = torch.zeros(2, 5, dtype=torch.bool)
    speak[0, 0] = True
    score = _score_handoff(
        speak, "handoff", reference_speaker=0, other_agents=[1],
    )
    assert score.listener_any is False
    assert score.clean_handoff is False
    assert score.interruption_handoff is False
    assert score.dirty_handoff is False


def test_run_probe_suite_returns_expected_keys_end_to_end(tiny_model, matrix_config):
    """Full smoke test of the reusable, in-training-callable probe entry
    point against a real (tiny) model + generation path -- not just the
    pure-tensor helpers above. Values are meaningless on a random tiny
    model; this only checks the function runs and returns the fields
    main.py's periodic in-training probe (--probe_every) relies on."""
    from model.matrix_qwen import MatrixQwenForCausalLM

    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.eval()
    tokenizer = _FakeTokenizer()

    with torch.no_grad():
        summary = run_probe_suite(
            model, tokenizer, num_agents=3, scaling_agent_counts=(),
            include_secret_scenarios=False, max_new_tokens=3, activity_threshold=0.5,
        )

    for key in (
        "clean_handoff_rate", "no_listener_response_rate", "listener_response_rate",
        "nonoverlap_listener_rate", "overlap_listener_rate", "mixed_response_rate",
        "unwanted_interruption_rate",
        "overlap_rate", "distinct_1", "distinct_2", "repeated_4gram_fraction",
        "calibration_summary", "scaling_summary", "secret_scenarios_summary",
    ):
        assert key in summary
    assert 0.0 <= summary["clean_handoff_rate"] <= 1.0
    assert 0.0 <= summary["no_listener_response_rate"] <= 1.0
    assert summary["listener_response_rate"] + summary["no_listener_response_rate"] == pytest.approx(1.0)
    assert (
        summary["nonoverlap_listener_rate"] + summary["overlap_listener_rate"]
        == pytest.approx(summary["listener_response_rate"])
    )
    # No extra scaling agent counts requested -- the (redundant) scaling
    # sweep at num_agents itself should be skipped entirely.
    assert summary["scaling_summary"] == {}
    assert summary["secret_scenarios_summary"]["num_scenarios"] == 0


def test_run_probe_suite_runs_scaling_sweep_when_requested(tiny_model, matrix_config):
    from model.matrix_qwen import MatrixQwenForCausalLM

    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.eval()
    tokenizer = _FakeTokenizer()

    with torch.no_grad():
        summary = run_probe_suite(
            model, tokenizer, num_agents=3, scaling_agent_counts=(2,),
            include_secret_scenarios=False, max_new_tokens=3, activity_threshold=0.5,
        )
    assert summary["scaling_agent_counts"] == [2, 3]
    assert summary["scaling_summary"]  # non-empty now that a scaling count was requested


def test_primed_handoff_builder_generalizes_to_arbitrary_agent_counts():
    tok = _FakeTokenizer()
    first, second = PRIMED_HANDOFF_TEMPLATES[0]
    turns = _primed_handoff_turns(first, second)
    for num_agents in (2, 3, 5):
        input_ids, activity, private, visibility = _build_multiturn_inputs(
            tok, turns, num_agents=num_agents, device="cpu",
        )
        assert input_ids.shape[1] == num_agents
        # Agent 0's block comes first, agent 1's block comes second.
        first_len = len(first)
        assert activity[0, 0, :first_len].all()
        assert activity[0, 1, first_len:].all()
        if num_agents > 2:
            assert not activity[0, 2:, :].any()
