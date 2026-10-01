"""Coherence-safe cleanup for timed multi-speaker conversations.

Long interruptions are serialized by shifting the *entire later utterance*
after the current floor, retaining every word, token order, and privacy
setting.  This deliberately differs from matrix-level token deferral, which
can detach word fragments and corrupt dialogue.
"""

from __future__ import annotations

import copy
from typing import Callable, Iterable, List

from .schema import Conversation, Turn

Tokenize = Callable[[str], List[int]]


def _has_alphanumeric_text(turn: Turn) -> bool:
    """Whether a closed utterance contains content beyond punctuation."""
    return any(char.isalnum() for char in turn.text)


def _overlapping_token_count(
    later: Turn, earlier: Iterable[Turn], tokenize: Tokenize
) -> int:
    """Count later-turn tokens whose timed words overlap an earlier turn."""
    count = 0
    for word in later.timed_words:
        if any(
            prior.start_time is not None
            and prior.end_time is not None
            and float(word.start_time) < float(prior.end_time)
            and float(word.end_time) > float(prior.start_time)
            for prior in earlier
        ):
            prefix = "" if word.is_punctuation else " "
            count += len(tokenize(prefix + word.text))
    return count


def _shift_turn(turn: Turn, delta: float) -> Turn:
    """Return a copy of a timed turn translated by ``delta`` seconds."""
    shifted = copy.deepcopy(turn)
    if shifted.start_time is not None:
        shifted.start_time = float(shifted.start_time) + delta
    if shifted.end_time is not None:
        shifted.end_time = float(shifted.end_time) + delta
    for word in shifted.timed_words:
        word.start_time = float(word.start_time) + delta
        word.end_time = float(word.end_time) + delta
    return shifted


def clean_timed_conversation(
    conversation: Conversation,
    tokenize: Tokenize,
    overlap_keep_tokens: int,
    inter_turn_gap_seconds: float = 0.05,
) -> Conversation:
    """Return a cleaned copy of a timed conversation.

    Short overlaps are retained. When a later turn has more than
    ``overlap_keep_tokens`` tokens occurring during an earlier active turn,
    it is moved as a complete unit after all active turns. Processing follows
    original onset order, so turn order never changes.
    """
    if overlap_keep_tokens < 0:
        raise ValueError("overlap_keep_tokens must be non-negative")

    cleaned = copy.deepcopy(conversation)
    turns = [turn for turn in cleaned.turns if _has_alphanumeric_text(turn)]
    scheduled: List[Turn] = []

    for turn in turns:
        if turn.start_time is None or turn.end_time is None or not turn.timed_words:
            scheduled.append(turn)
            continue

        active = [
            prior
            for prior in scheduled
            if prior.start_time is not None
            and prior.end_time is not None
            and float(prior.start_time) < float(turn.end_time)
            and float(prior.end_time) > float(turn.start_time)
        ]
        overlap_tokens = _overlapping_token_count(turn, active, tokenize)
        if active and overlap_tokens > overlap_keep_tokens:
            boundary = max(float(prior.end_time) for prior in active)
            delta = boundary + inter_turn_gap_seconds - float(turn.start_time)
            turn = _shift_turn(turn, max(0.0, delta))
        scheduled.append(turn)

    cleaned.turns = scheduled
    cleaned.meta = dict(cleaned.meta)
    cleaned.meta["overlap_cleaned"] = True
    cleaned.meta["overlap_keep_tokens"] = overlap_keep_tokens
    return cleaned
