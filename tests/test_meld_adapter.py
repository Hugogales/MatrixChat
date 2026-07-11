"""Tests for the MELD adapter's maximal-dialogue dedup (offline, no HF download).

MELD-MPCA rows are complete (but redundant) conversation contexts -- the same
underlying dialogue can appear at several lengths, in arbitrary order. The
adapter keeps only MAXIMAL contexts (drop any sequence that is a prefix of a
longer one), so we train on whole dialogues, not truncated fragments.
"""

from data.adapters.meld import reconstruct_dialogues, _to_conversation


def _row(*pairs):
    return {"input": [{"user": u, "content": c} for u, c in pairs]}


def _as_set(dialogues):
    return {tuple(d) for d in dialogues}


def test_prefix_snapshots_collapse_to_longest():
    # Same dialogue at 3 lengths, scattered order -> keep only the longest.
    rows = [
        _row(("user1", "hi"), ("user2", "hey")),
        _row(("user1", "hi")),
        _row(("user1", "hi"), ("user2", "hey"), ("user1", "bye")),
    ]
    dialogues = list(reconstruct_dialogues(rows))
    assert dialogues == [[("user1", "hi"), ("user2", "hey"), ("user1", "bye")]]


def test_independent_dialogues_all_kept():
    # Different dialogues (not prefixes of each other) are all retained.
    rows = [
        _row(("user1", "a"), ("user2", "b")),
        _row(("user3", "c"), ("user4", "d")),
        _row(("user1", "a"), ("user2", "b")),  # exact duplicate -> deduped
    ]
    dialogues = list(reconstruct_dialogues(rows))
    assert len(dialogues) == 2
    assert _as_set(dialogues) == {
        (("user1", "a"), ("user2", "b")),
        (("user3", "c"), ("user4", "d")),
    }


def test_empty_and_blank_content_ignored():
    rows = [
        _row(("user1", "  ")),                     # all-blank -> dropped entirely
        _row(("user1", "hello"), ("user2", "")),   # blank second turn dropped
    ]
    dialogues = list(reconstruct_dialogues(rows))
    assert dialogues == [[("user1", "hello")]]


def test_to_conversation_builds_speakers_and_turns():
    conv = _to_conversation([("user1", "hi"), ("user2", "hey"), ("user1", "bye")])
    assert conv is not None
    assert conv.source == "meld"
    assert conv.content_bearing is True
    assert conv.speakers == ["user1", "user2"]
    assert [t.speaker for t in conv.turns] == [0, 1, 0]
    assert [t.text for t in conv.turns] == ["hi", "hey", "bye"]


def test_single_speaker_conversation_rejected():
    # Needs >= 2 speakers to be a usable multi-party conversation.
    assert _to_conversation([("user1", "hi"), ("user1", "again")]) is None
