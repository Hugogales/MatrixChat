"""Tests for the When2Speak adapter (2026-08-24): parsing the OpenAI
chat-format rows released on HF into Conversation/Turn objects, without
touching the network (``_iter_split`` is tested directly on in-memory rows)."""

from data.adapters.when2speak import SPEC, _iter_split, _same_chain, _iter_deduped_rows


def test_speak_row_appends_agent_turn_with_real_text():
    rows = [{
        "messages": [
            {"role": "user", "content": "Speaker_0: I've been wondering about the history of jazz"},
            {"role": "user", "content": "Speaker_1: What do you think [AGENT]?"},
            {"role": "assistant", "content": "Jazz originated in New Orleans in the early 20th century."},
        ]
    }]
    convs = list(_iter_split(rows))
    assert len(convs) == 1
    conv = convs[0]
    assert conv.source == "when2speak"
    assert conv.content_bearing is False
    assert conv.speakers == ["Speaker_0", "Speaker_1", "[AGENT]"]
    assert [t.speaker for t in conv.turns] == [0, 1, 2]
    assert conv.turns[-1].text == "Jazz originated in New Orleans in the early 20th century."


def test_silent_row_has_no_trailing_agent_turn():
    rows = [{
        "messages": [
            {"role": "user", "content": "Speaker_0: I think we should go with option A"},
            {"role": "user", "content": "Speaker_1: I agree, that makes sense"},
            {"role": "assistant", "content": ">"},
        ]
    }]
    convs = list(_iter_split(rows))
    assert len(convs) == 1
    conv = convs[0]
    assert conv.speakers == ["Speaker_0", "Speaker_1"]
    assert "[AGENT]" not in conv.speakers
    assert len(conv.turns) == 2


def test_single_speaker_silent_row_is_skipped():
    """One human speaker + no agent turn (SILENT) => only 1 speaker total,
    below the >=2-speaker structural-signal floor used by every adapter."""
    rows = [{
        "messages": [
            {"role": "user", "content": "Speaker_0: Just thinking out loud here"},
            {"role": "assistant", "content": ">"},
        ]
    }]
    assert list(_iter_split(rows)) == []


def test_prior_agent_turn_reappearing_as_context_is_labeled_agent_not_unknown():
    """Regression (2026-08-25, found by inspecting real downloaded rows): a
    prior window's AGENT turn reappears in a LATER window's context as a
    "user"-role message with content "Assistant: <text>" (not "Speaker_N:
    <text>") -- confirmed against the real HF data. Before this fix,
    `_split_speaker` fell through to speaker="unknown" with the literal
    "Assistant: " prefix left baked into the turn text."""
    rows = [{
        "messages": [
            {"role": "user", "content": "Speaker_0: is this trademark infringement"},
            {"role": "user", "content": "Assistant: Yes, likely so."},
            {"role": "user", "content": "Speaker_1: what about a different color scheme"},
            {"role": "assistant", "content": "That would still risk confusion."},
        ]
    }]
    convs = list(_iter_split(rows))
    assert len(convs) == 1
    conv = convs[0]
    assert conv.speakers == ["Speaker_0", "[AGENT]", "Speaker_1"]
    assert "unknown" not in conv.speakers
    agent_context_turn = conv.turns[1]
    assert agent_context_turn.speaker == 1
    assert agent_context_turn.text == "Yes, likely so."
    assert "Assistant:" not in agent_context_turn.text
    # The final decision turn reuses the SAME [AGENT] speaker index, not a
    # second distinct one.
    assert conv.turns[-1].speaker == 1


def test_malformed_content_falls_back_gracefully():
    rows = [{
        "messages": [
            {"role": "user", "content": "no speaker prefix here"},
            {"role": "user", "content": "Speaker_0: a real turn"},
            {"role": "assistant", "content": "a reply"},
        ]
    }]
    convs = list(_iter_split(rows))
    assert len(convs) == 1
    assert convs[0].speakers[0] == "unknown"


def test_spec_is_structure_only():
    assert SPEC.content_bearing is False
    assert SPEC.hf_repo == "duke-trust-lab/When2Speak"
    assert SPEC.source_id == 6


def _sliding_row(msgs, final):
    """Build a row shaped like one sliding-window position: `msgs` are the
    user-role context turns, `final` is the assistant-role decision."""
    return {
        "messages": [{"role": "user", "content": m} for m in msgs]
        + [{"role": "assistant", "content": final}]
    }


def test_same_chain_detects_sliding_window_overlap():
    prev = ["Speaker_0: a", "Speaker_1: b", "Speaker_0: c"]
    # One step of sliding: same 3 messages plus one appended.
    curr = prev + ["Speaker_1: d"]
    assert _same_chain(prev, curr) is True


def test_same_chain_rejects_unrelated_transcript():
    prev = ["Speaker_0: talking about jazz history today"]
    curr = ["Speaker_0: a totally different topic about tax law"]
    assert _same_chain(prev, curr) is False


def test_same_chain_rejects_empty_inputs():
    assert _same_chain([], ["Speaker_0: a"]) is False
    assert _same_chain(["Speaker_0: a"], []) is False


def test_iter_deduped_rows_keeps_short_chain_intact():
    """A chain no longer than the cap is kept in full (order-preserving)."""
    rows = [
        _sliding_row(["Speaker_0: a"], ">"),
        _sliding_row(["Speaker_0: a", "Speaker_1: b"], ">"),
        _sliding_row(["Speaker_0: a", "Speaker_1: b", "Speaker_0: c"], "reply"),
    ]
    kept = list(_iter_deduped_rows(rows, max_windows_per_chain=4, seed=0))
    assert kept == rows


def test_iter_deduped_rows_caps_long_chain():
    """A chain longer than the cap is subsampled down to exactly the cap, so
    one long synthetic transcript can't flood the dataset with near-dupes."""
    rows = []
    msgs = ["Speaker_0: turn0"]
    for i in range(1, 20):
        msgs = msgs + [f"Speaker_{i % 2}: turn{i}"]
        rows.append(_sliding_row(msgs, ">"))
    assert len(rows) == 19
    kept = list(_iter_deduped_rows(rows, max_windows_per_chain=4, seed=0))
    assert len(kept) == 4
    # Every kept row must actually come from the original chain.
    assert all(r in rows for r in kept)


def test_iter_deduped_rows_splits_at_chain_boundary():
    """Two unrelated transcripts (no overlap) are treated as separate chains,
    each capped independently."""
    chain_a = [_sliding_row(["Speaker_0: jazz history topic"], ">")]
    chain_b = [_sliding_row(["Speaker_0: unrelated tax law topic"], "an answer")]
    kept = list(_iter_deduped_rows(chain_a + chain_b, max_windows_per_chain=4, seed=0))
    assert kept == chain_a + chain_b


def test_iter_deduped_rows_is_deterministic_given_seed():
    rows = []
    msgs = ["Speaker_0: turn0"]
    for i in range(1, 20):
        msgs = msgs + [f"Speaker_{i % 2}: turn{i}"]
        rows.append(_sliding_row(msgs, ">"))
    kept1 = list(_iter_deduped_rows(rows, max_windows_per_chain=4, seed=7))
    kept2 = list(_iter_deduped_rows(rows, max_windows_per_chain=4, seed=7))
    assert kept1 == kept2


def test_iter_split_respects_max_windows_per_chain():
    """End-to-end: a long chain of SILENT windows plus one final SPEAK
    decision yields at most `max_windows_per_chain` Conversations."""
    rows = []
    msgs = ["Speaker_0: turn0", "Speaker_1: turn1"]
    for i in range(2, 15):
        msgs = msgs + [f"Speaker_{i % 2}: turn{i}"]
        rows.append(_sliding_row(msgs, ">"))
    convs = list(_iter_split(rows, max_windows_per_chain=3, seed=0))
    assert len(convs) == 3
    assert all(c.source == "when2speak" for c in convs)
