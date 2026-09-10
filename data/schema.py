"""Unified intermediate schema for MatrixChat datasets.

Every raw dataset is normalized by its adapter into a :class:`Conversation`
(a list of :class:`Turn`s). The matrix converter (``data/convert.py``) consumes
only this schema, so adding a new dataset means writing one adapter -- no changes
to the converter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Canonical role strings. "peer" = an undifferentiated participant in a
# multi-party human conversation (no assistant/user distinction).
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_PEER = "peer"


@dataclass
class TimedWord:
    """A transcript terminal aligned to wall-clock time in seconds."""

    text: str
    start_time: float
    end_time: float
    is_punctuation: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class Turn:
    """A single utterance by one speaker."""

    speaker: int            # index into Conversation.speakers
    text: str
    role: str = ROLE_PEER   # one of the ROLE_* constants
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    timed_words: List[TimedWord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    # Extra speaker indices (into Conversation.speakers) allowed to view this
    # turn's cells, beyond the speaker itself. None = public (default; every
    # existing source). A non-None value marks the turn PRIVATE: the converter
    # will still place its tokens as normal active cells on the speaker's row,
    # but the attention mask will block any agent not in this list (plus the
    # speaker) from attending to those cells as keys.
    visible_to: Optional[List[int]] = None


@dataclass
class Conversation:
    """A normalized multi-party conversation.

    Attributes
    ----------
    source:
        Dataset name (e.g. ``"molweni"``).
    speakers:
        Display names/ids of the distinct speakers; ``Turn.speaker`` indexes this.
    turns:
        Ordered utterances.
    content_bearing:
        If True, this source may carry next-token CONTENT loss on a target seat.
        Structure-only sources (e.g. Molweni, Werewolf) set this False -- they
        teach turn-taking/silence and provide context, but never content labels
        (so the model does not learn their typos/jargon).
    target_speaker:
        Optional pre-chosen target seat (the speaker whose content is supervised).
        If None and ``content_bearing``, the converter picks one (preferring an
        ``assistant`` role, else a rotated random speaker).
    """

    source: str
    speakers: List[str]
    turns: List[Turn]
    content_bearing: bool = True
    target_speaker: Optional[int] = None
    meta: dict = field(default_factory=dict)
