"""Tests for coherence-safe long-overlap serialization."""

from data.overlap_cleaner import clean_timed_conversation
from data.schema import Conversation, TimedWord, Turn


def _tokenize(text):
    return [index + 1 for index, _ in enumerate(text.strip().split())]


def _turn(speaker, text, start, end, *, private=False):
    return Turn(
        speaker=speaker,
        text=text,
        start_time=start,
        end_time=end,
        timed_words=[TimedWord(text=text, start_time=start, end_time=end)],
        visible_to=[speaker] if private else None,
    )


def test_short_overlap_is_preserved():
    conversation = Conversation(
        source="ami",
        speakers=["A", "B"],
        turns=[_turn(0, "alpha", 0.0, 2.0), _turn(1, "ok", 1.0, 1.2)],
        meta={"timed": True},
    )

    cleaned = clean_timed_conversation(conversation, _tokenize, overlap_keep_tokens=2)

    assert cleaned.turns[1].start_time == 1.0
    assert cleaned.turns[1].text == "ok"


def test_long_overlap_moves_complete_later_turn_without_reordering():
    conversation = Conversation(
        source="ami",
        speakers=["A", "B", "C"],
        turns=[
            _turn(0, "alpha", 0.0, 2.0),
            _turn(1, "secondary reply now", 1.0, 1.8, private=True),
            _turn(2, "third reply", 2.1, 2.6),
        ],
        meta={"timed": True},
    )

    cleaned = clean_timed_conversation(conversation, _tokenize, overlap_keep_tokens=2)

    assert [turn.text for turn in cleaned.turns] == [
        "alpha", "secondary reply now", "third reply"
    ]
    assert cleaned.turns[1].start_time >= 2.05
    assert cleaned.turns[1].timed_words[0].text == "secondary reply now"
    assert cleaned.turns[1].visible_to == [1]
    # The subsequent two-token interruption is intentionally retained.
    assert cleaned.turns[2].start_time == 2.1


def test_punctuation_only_turn_is_removed_but_yeah_is_preserved():
    conversation = Conversation(
        source="werewolf",
        speakers=["A", "B"],
        turns=[
            _turn(0, "...", 0.0, 0.1),
            _turn(1, "yeah", 0.2, 0.4),
            _turn(0, "42", 0.5, 0.7),
        ],
        meta={"timed": True},
    )

    cleaned = clean_timed_conversation(conversation, _tokenize, overlap_keep_tokens=3)

    assert [turn.text for turn in cleaned.turns] == ["yeah", "42"]
