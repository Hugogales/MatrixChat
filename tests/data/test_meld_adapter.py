"""Tests for the MELD adapter's maximal-dialogue dedup (offline, no HF download).

MELD-MPCA rows are complete (but redundant) conversation contexts -- the same
underlying dialogue can appear at several lengths, in arbitrary order. The
adapter keeps only MAXIMAL contexts (drop any sequence that is a prefix of a
longer one), so we train on whole dialogues, not truncated fragments.
"""

import re

from data.adapters import meld
from data.adapters._names import RANDOM_NAME_POOL, sample_random_names
from data.adapters.meld import (
    _FRIENDS_CHARACTER_NAMES,
    _find_present_character_names,
    _substitute_character_names,
    _to_conversation,
    reconstruct_dialogues,
)


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


def test_random_name_pool_is_large_and_diverse():
    # Explicit request: "make a longer list than just 50 random names".
    assert len(RANDOM_NAME_POOL) > 200
    assert len(set(RANDOM_NAME_POOL)) == len(RANDOM_NAME_POOL)


def test_find_present_character_names_detects_whole_word_case_insensitive():
    texts = ["hey ROSS what's up", "Rachel is not here", "totally unrelated line"]
    present = _find_present_character_names(texts)
    assert present == ["Ross", "Rachel"]  # preserves _FRIENDS_CHARACTER_NAMES order


def test_find_present_character_names_ignores_substring_matches():
    # "Rossignol" contains "Ross" but is not a whole-word match.
    assert _find_present_character_names(["Rossignol was here"]) == []


def test_substitute_character_names_replaces_every_occurrence_in_the_scene():
    texts = ["hey Ross, where's Rachel?", "Ross went to see Rachel already"]
    substituted = _substitute_character_names(texts, "scene-A")
    assert substituted != texts
    for name in ("Ross", "Rachel"):
        for t in substituted:
            assert name.lower() not in t.lower()

    # Same real name maps to the SAME replacement everywhere within the scene:
    # reconstruct the expected mapping the same way the function does.
    present = _find_present_character_names(texts)
    random_names = sample_random_names("scene-A", len(present), exclude=set(_FRIENDS_CHARACTER_NAMES))
    ross_repl = dict(zip(present, random_names))["Ross"]
    rachel_repl = dict(zip(present, random_names))["Rachel"]
    assert ross_repl in substituted[0] and ross_repl in substituted[1]
    assert rachel_repl in substituted[0] and rachel_repl in substituted[1]


def test_substitute_character_names_is_deterministic_per_scene_key():
    texts = ["Ross and Rachel talked"]
    a = _substitute_character_names(texts, "scene-fixed")
    b = _substitute_character_names(texts, "scene-fixed")
    assert a == b


def test_substitute_character_names_differs_across_scenes():
    texts = ["Ross and Rachel talked"]
    a = _substitute_character_names(texts, "scene-1")
    b = _substitute_character_names(texts, "scene-2")
    assert a != b


def test_substitute_character_names_never_draws_a_known_character_name():
    # Regression: replacement names are drawn excluding the ENTIRE
    # _FRIENDS_CHARACTER_NAMES vocabulary, so a substitution can never
    # coincidentally read as a "leaked" real name during later auditing.
    texts = ["Ross and Rachel and Monica and Chandler and Joey and Phoebe talked"]
    for seed in ("s1", "s2", "s3", "s4", "s5"):
        substituted = _substitute_character_names(texts, seed)[0]
        for name in _FRIENDS_CHARACTER_NAMES:
            assert not re.search(rf"\b{name}\b", substituted, re.IGNORECASE)


def test_substitute_character_names_is_a_noop_without_known_names():
    texts = ["nothing show-related here", "just a normal conversation"]
    assert _substitute_character_names(texts, "scene-noop") == texts


def test_to_conversation_substitutes_character_names_per_scene():
    conv = _to_conversation(
        [("user1", "hey Ross, seen Rachel?"), ("user2", "Ross was with Rachel earlier")],
        conversation_id="conv-xyz",
    )
    assert conv is not None
    for turn in conv.turns:
        assert "Ross" not in turn.text
        assert "Rachel" not in turn.text
    # Same scene -> same substitution for both mentions of "Ross"/"Rachel".
    present = _find_present_character_names(["hey Ross, seen Rachel?", "Ross was with Rachel earlier"])
    random_names = sample_random_names("conv-xyz", len(present), exclude=set(_FRIENDS_CHARACTER_NAMES))
    mapping = dict(zip(present, random_names))
    for turn in conv.turns:
        assert mapping["Ross"] in turn.text
        assert mapping["Rachel"] in turn.text


def test_to_conversation_substitution_is_deterministic_for_same_conversation_id():
    messages = [("user1", "hey Ross, seen Rachel?"), ("user2", "Ross was with Rachel earlier")]
    conv_a = _to_conversation(messages, conversation_id="conv-same")
    conv_b = _to_conversation(messages, conversation_id="conv-same")
    assert [t.text for t in conv_a.turns] == [t.text for t in conv_b.turns]


def test_friends_character_names_has_no_duplicates():
    assert len(_FRIENDS_CHARACTER_NAMES) == len(set(_FRIENDS_CHARACTER_NAMES))


def test_single_hf_split_gets_deterministic_group_safe_validation(monkeypatch):
    rows = [
        _row(("user1", f"hello {i}"), ("user2", f"reply {i}"))
        for i in range(200)
    ]
    monkeypatch.setattr(meld, "_load_splits", lambda raw_dir: [("train", rows)])
    first = list(meld.iter_conversations("unused"))
    second = list(meld.iter_conversations("unused"))

    first_assignments = {
        conv.meta["conversation_id"]: conv.meta["dataset_split"] for conv in first
    }
    second_assignments = {
        conv.meta["conversation_id"]: conv.meta["dataset_split"] for conv in second
    }
    assert first_assignments == second_assignments
    assert "train" in set(first_assignments.values())
    assert "validation" in set(first_assignments.values())
    assert "test" in set(first_assignments.values())
    assert all(conv.meta["group_id"] == conv.meta["conversation_id"] for conv in first)
