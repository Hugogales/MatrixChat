"""MELD adapter (Friends TV, multi-speaker, casual register).

Source: neoluigi/MELD-MPCA (HF). Each row has ``input`` = a list of
``{user, content}`` messages (anonymized speakers ``user1``...) and ``output`` =
next-reply metadata (``tar_speaker``, ``emotion``, ...).

Each row is a COMPLETE conversation context (a coherent, turn-complete multi-turn
exchange) -- NOT an adjacent prefix of its neighbor. Because the corpus is a "who
replies next" task, a single underlying dialogue can appear as several rows with
different context lengths (scattered, not adjacent). To train on whole dialogues
(not truncated fragments) we keep only MAXIMAL contexts: a row is dropped if its
message sequence is a prefix of a longer kept row (i.e. a shorter snapshot of the
same dialogue). See :func:`reconstruct_dialogues`.

Content-bearing (breadth tier); the ``output`` metadata is currently unused.
"""

from __future__ import annotations

from typing import Iterator, List

from .spec import AdapterSpec
from ..schema import Conversation, Turn, ROLE_PEER


def _messages(row) -> List[dict]:
    """Extract the ``[{user, content}, ...]`` list from a row (defensive)."""
    val = row.get("input")
    if isinstance(val, list):
        return val
    # Fallbacks for other plausible shapes.
    for key in ("dialogue", "utterances", "messages"):
        v = row.get(key)
        if isinstance(v, list):
            return v
    return []


def _norm(messages: List[dict]):
    """Normalize to a tuple of (user, content) pairs with non-empty content."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        user = str(m.get("user") or m.get("speaker") or "unknown")
        content = (m.get("content") or m.get("text") or "").strip()
        if content:
            out.append((user, content))
    return out


def reconstruct_dialogues(rows) -> Iterator[List[tuple]]:
    """Yield MAXIMAL dialogues: drop any context that is a prefix of a longer one.

    Rows are complete but possibly redundant contexts (different-length snapshots
    of the same underlying dialogue, in arbitrary order). We keep only the
    longest version of each: process longest-first, and skip a sequence if it is
    a prefix of (or identical to) an already-kept longer sequence. Each yielded
    item is a full dialogue ``[(user, content), ...]``.
    """
    seqs = []
    for row in rows:
        msgs = _norm(_messages(row))
        if msgs:
            seqs.append(tuple(msgs))
    seqs.sort(key=len, reverse=True)  # longest first so we keep maximal contexts

    covered = set()  # all prefixes of already-kept sequences
    for s in seqs:
        if s in covered:
            continue  # s is a prefix of (or duplicate of) a longer kept dialogue
        yield list(s)
        for k in range(1, len(s) + 1):
            covered.add(s[:k])


def _to_conversation(messages: List[tuple]) -> Conversation | None:
    speakers: List[str] = []
    turns: List[Turn] = []
    for user, content in messages:
        if user not in speakers:
            speakers.append(user)
        turns.append(Turn(speaker=speakers.index(user), text=content, role=ROLE_PEER))
    if len(turns) >= 2 and len(speakers) >= 2:
        return Conversation(source="meld", speakers=speakers, turns=turns, content_bearing=True)
    return None


def _load_all_splits(raw_dir: str):
    """Yield rows across every available split (train + any dev/test)."""
    from datasets import get_dataset_split_names, load_dataset

    try:
        splits = get_dataset_split_names("neoluigi/MELD-MPCA")
    except Exception:
        splits = ["train"]
    for split in splits:
        ds = load_dataset("neoluigi/MELD-MPCA", split=split, cache_dir=raw_dir)
        for row in ds:
            yield row


def iter_conversations(raw_dir: str) -> Iterator[Conversation]:
    for messages in reconstruct_dialogues(_load_all_splits(raw_dir)):
        conv = _to_conversation(messages)
        if conv is not None:
            yield conv


SPEC = AdapterSpec(
    source="meld",
    content_bearing=True,
    default_weight=0.15,
    iter_conversations=iter_conversations,
    hf_repo="neoluigi/MELD-MPCA",
    notes="Breadth tier: casual multi-party (Friends). Content-bearing, rotated target.",
)
