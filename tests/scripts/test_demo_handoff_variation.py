"""Unit tests for the pure-tensor/pure-logic helpers in
scripts/eval/demo_handoff_variation.py (no model needed).
"""

import json
import types

import torch

import scripts.eval.demo_handoff_variation as dhv
from scripts.eval.demo_handoff_variation import (
    CONTEXT_KINDS,
    _build_context,
    _last_speaking_agents,
    _load_resume_results,
    _run_chain,
)


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(str(i) for i in ids)


def test_build_context_cold_is_a_single_public_turn():
    turns, ref, kind = _build_context("cold", "hello there", num_agents=3)
    assert turns == [{"speaker": 0, "text": "hello there", "visible_to": None}]
    assert ref == 0
    assert kind == "handoff"


def test_build_context_primed_gap0_has_no_silence_turn():
    turns, ref, kind = _build_context("primed_handoff_gap0", "prompt text", num_agents=3)
    assert [t for t in turns if "silence_columns" in t] == [{"silence_columns": 0}]
    assert turns[0]["speaker"] == 0
    assert turns[-1] == {"speaker": 1, "text": "prompt text", "visible_to": None}
    assert ref == 1
    assert kind == "primed_handoff"


def test_build_context_primed_gap3_inserts_a_3_column_gap():
    turns, ref, kind = _build_context("primed_handoff_gap3", "prompt text", num_agents=3)
    gaps = [t["silence_columns"] for t in turns if "silence_columns" in t]
    assert gaps == [3]
    assert ref == 1
    assert kind == "primed_handoff"


def test_build_context_post_intro_has_one_turn_per_agent_plus_prompt():
    turns, ref, kind = _build_context("post_intro", "prompt text", num_agents=4)
    # 4 intro turns + 1 prompt turn.
    assert len(turns) == 5
    assert [t["speaker"] for t in turns[:4]] == [0, 1, 2, 3]
    assert turns[-1] == {"speaker": 0, "text": "prompt text", "visible_to": None}
    assert ref == 0
    assert kind == "handoff"


def test_build_context_lengthy_never_uses_agent0_in_the_filler():
    turns, ref, kind = _build_context("lengthy", "prompt text", num_agents=3)
    filler_speakers = [t["speaker"] for t in turns[:-1]]
    assert 0 not in filler_speakers
    assert turns[-1] == {"speaker": 0, "text": "prompt text", "visible_to": None}
    assert ref == 0
    assert kind == "handoff"


def test_build_context_lengthy_with_two_agents_falls_back_to_agent1_only():
    turns, ref, kind = _build_context("lengthy", "prompt text", num_agents=2)
    filler_speakers = [t["speaker"] for t in turns[:-1]]
    assert set(filler_speakers) == {1}


def test_real_prefix_falls_back_to_lengthy_when_no_excerpts_available(monkeypatch):
    monkeypatch.setattr(dhv, "_load_real_excerpts", lambda: [])
    turns, ref, kind = _build_context("real_prefix", "prompt text", num_agents=3)
    # Same shape as "lengthy": filler turns (never agent 0) + the swept prompt.
    filler_speakers = [t["speaker"] for t in turns[:-1]]
    assert 0 not in filler_speakers
    assert turns[-1] == {"speaker": 0, "text": "prompt text", "visible_to": None}
    assert ref == 0
    assert kind == "handoff"


def test_real_prefix_uses_real_excerpt_and_continues_with_the_swept_prompt(monkeypatch):
    fake_excerpt = [
        {"speaker": 0, "text": "real turn one", "visible_to": None},
        {"speaker": 1, "text": "real turn two", "visible_to": None},
        {"speaker": 0, "text": "real turn three", "visible_to": None},
    ]
    monkeypatch.setattr(dhv, "_load_real_excerpts", lambda: [fake_excerpt])
    turns, ref, kind = _build_context("real_prefix", "prompt text", num_agents=3)
    assert turns[:3] == fake_excerpt
    # The last real speaker (agent 0) "continues" with the swept prompt.
    assert turns[-1] == {"speaker": 0, "text": "prompt text", "visible_to": None}
    assert ref == 0
    assert kind == "handoff"


def test_real_prefix_remaps_speakers_beyond_num_agents(monkeypatch):
    fake_excerpt = [
        {"speaker": 0, "text": "a", "visible_to": None},
        {"speaker": 3, "text": "b", "visible_to": None},  # would be out of range for num_agents=2
    ]
    monkeypatch.setattr(dhv, "_load_real_excerpts", lambda: [fake_excerpt])
    turns, ref, kind = _build_context("real_prefix", "prompt text", num_agents=2)
    assert all(0 <= t["speaker"] < 2 for t in turns)


def test_all_context_kinds_are_buildable_for_a_range_of_agent_counts():
    for kind in CONTEXT_KINDS:
        for num_agents in (2, 3, 5):
            turns, ref, score_kind = _build_context(kind, "some prompt", num_agents)
            assert 0 <= ref < num_agents
            assert score_kind in ("handoff", "primed_handoff")


def test_last_speaking_agents_returns_agent_active_in_final_active_column():
    speak = torch.zeros(3, 6, dtype=torch.bool)
    speak[0, 0:2] = True
    speak[1, 3:5] = True  # agent 1 is the last to speak, ending at step 4
    assert _last_speaking_agents(speak) == [1]


def test_last_speaking_agents_empty_when_fully_silent():
    speak = torch.zeros(2, 4, dtype=torch.bool)
    assert _last_speaking_agents(speak) == []


def test_last_speaking_agents_returns_multiple_on_overlap_in_final_column():
    speak = torch.zeros(3, 4, dtype=torch.bool)
    speak[0, 3] = True
    speak[2, 3] = True  # both active in the very last column
    assert sorted(_last_speaking_agents(speak)) == [0, 2]


def test_run_chain_continues_past_a_brief_dirty_overlap(monkeypatch):
    """Regression test for the 08-05 fix: earlier versions stopped the
    rollout the instant ANY overlap occurred, so a real 2nd handoff could
    never even be observed if the 1st handoff had so much as one
    simultaneous token. A brief/dirty overlap (within tolerance) must let
    the chain keep extending instead of truncating information."""
    calls = []

    def fake_generate_matrix(model, inp, max_new_tokens, **kwargs):
        calls.append(inp.shape[2])  # track how much context each call sees
        if len(calls) == 1:
            # Link 1: reference (agent0) speaks steps0-3, listener (agent1)
            # starts at step2 -- 2 overlapping tokens (steps2-3), within the
            # default dirty-handoff tolerance. Ends with agent1 alone
            # holding the floor at the final step.
            tokens = torch.zeros(1, 2, 8, dtype=torch.long)
            speak = torch.zeros(1, 2, 8, dtype=torch.bool)
            speak[0, 0, 0:4] = True
            speak[0, 1, 2:8] = True
        else:
            # Link 2: new reference (agent1) finishes cleanly, agent0 takes
            # over with zero overlap -- a fully clean 2nd handoff.
            tokens = torch.zeros(1, 2, 6, dtype=torch.long)
            speak = torch.zeros(1, 2, 6, dtype=torch.bool)
            speak[0, 1, 0:2] = True
            speak[0, 0, 3:6] = True
        return tokens, speak, None

    monkeypatch.setattr(dhv, "generate_matrix", fake_generate_matrix)

    inp = torch.zeros(1, 2, 1, dtype=torch.long)
    mask = torch.zeros(1, 2, 1, dtype=torch.bool)
    args = types.SimpleNamespace(max_new_tokens=8, temperature=0.8, activity_threshold=0.5)

    chain = _run_chain(
        model=None, tokenizer=_FakeTokenizer(), args=args,
        inp=inp, mask=mask, private_mask=None, visibility=None,
        num_agents=2, reference_speaker=0, score_kind="handoff", max_links=2,
    )

    assert len(calls) == 2, "chain must not stop after a brief interruption"
    assert len(chain) == 2
    assert chain[0]["interruption_handoff"] is True
    assert chain[0]["dirty_handoff"] is False
    assert chain[0]["clean_handoff"] is False
    assert chain[1]["clean_handoff"] is True
    # The 2nd call's context must have grown (i.e. it really extended the
    # rollout with link 1's own output, not just retried the same window).
    assert calls[1] > calls[0]


def test_load_resume_results_accepts_bare_partial_list(tmp_path):
    """The periodic `<output>.partial.json` checkpoint is a bare list."""
    rows = [{"prompt": "p", "num_agents": 2, "seed": 0, "context_kind": "cold"}]
    path = tmp_path / "run.partial.json"
    path.write_text(json.dumps(rows))
    assert _load_resume_results(str(path)) == rows


def test_load_resume_results_accepts_full_summary_dict(tmp_path):
    """Resuming from a previously-COMPLETED run's `--output` (a summary dict
    with a "results" key), not just a `.partial.json`, must also work."""
    rows = [{"prompt": "p", "num_agents": 2, "seed": 0, "context_kind": "cold"}]
    summary = {"total_trials": 1, "results": rows}
    path = tmp_path / "run_result.json"
    path.write_text(json.dumps(summary))
    assert _load_resume_results(str(path)) == rows


def test_resume_skip_set_matches_on_prompt_agents_seed_kind_tuple():
    """Regression guard for the (prompt, num_agents, seed, context_kind) key
    the main loop uses to decide what to skip on resume -- two rows that
    differ only in an unrelated field (e.g. `clean_handoff`) must still be
    treated as the SAME completed trial, and a row differing in any of the
    4 key fields must NOT be skipped."""
    completed = [
        {"prompt": "p1", "num_agents": 2, "seed": 0, "context_kind": "cold", "clean_handoff": True},
    ]
    done = {(r["prompt"], r["num_agents"], r["seed"], r["context_kind"]) for r in completed}
    assert ("p1", 2, 0, "cold") in done
    assert ("p1", 3, 0, "cold") not in done
    assert ("p2", 2, 0, "cold") not in done
