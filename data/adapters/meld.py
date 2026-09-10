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

Character-name overfitting
---------------------------
The ``user`` field is already anonymized (``user1``, ``user2``, ...), but the
dialogue TEXT itself is real Friends script content and mentions the show's
recurring character names by name (e.g. "Ross, come here", "tell Rachel...")
far more often than any real conversation would mention any one person --
the model was observed latching onto these specific names as generic content
regardless of prompt topic. To fix this, every scene (each reconstructed
dialogue) gets its own fresh, deterministic-but-unpredictable mapping from
each detected character name to a randomly drawn replacement (see
:func:`_substitute_character_names`, sharing ``data/adapters/_names.py``'s
large diverse pool with the Werewolf adapter): the same real name maps to a
DIFFERENT random name in a different scene, so no single name-token can act
as a memorizable shortcut.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable, Iterator, List, Optional

from .spec import AdapterSpec
from ._names import build_name_substitution_regex, sample_random_names
from ..schema import Conversation, Turn, ROLE_PEER

# Recurring Friends (TV show) character names that show up throughout
# MELD-MPCA's dialogue text -- not an exhaustive cast list, but broad enough
# to catch the vast majority of in-scene name mentions across all 10 seasons.
_FRIENDS_CHARACTER_NAMES = [
    "Ross", "Rachel", "Monica", "Chandler", "Joey", "Phoebe",
    "Gunther", "Janice", "Emily", "Susan", "Carol", "Ben", "Emma",
    "Mike", "Charlie", "Richard", "Julie", "David", "Kathy", "Elizabeth",
    "Tag", "Mona", "Amy", "Judy", "Jack", "Frank", "Alice", "Ursula",
    "Estelle", "Joshua", "Bob", "Pete", "Barry", "Paolo", "Danny",
    "Nora", "Gary", "Sid", "Precious", "Marcel", "Sandra", "Erica",
    "Kristen", "Elle", "Molly",
]

assert len(_FRIENDS_CHARACTER_NAMES) == len(set(_FRIENDS_CHARACTER_NAMES)), (
    "_FRIENDS_CHARACTER_NAMES must have no duplicates"
)


def _find_present_character_names(texts: List[str]) -> List[str]:
    """Distinct known character names actually mentioned in this scene
    (whole-word, case-insensitive), in ``_FRIENDS_CHARACTER_NAMES`` order so
    the same scene always finds the same names in the same order."""
    combined = " ".join(texts)
    return [
        name for name in _FRIENDS_CHARACTER_NAMES
        if re.search(rf"\b{re.escape(name)}\b", combined, re.IGNORECASE)
    ]


def _substitute_character_names(
    texts: List[str], scene_seed_key: str,
) -> List[str]:
    """Per-scene random name substitution: every character name mentioned
    ANYWHERE in this scene is mapped to one fresh random replacement name
    (consistent within the scene, independent across scenes) and substituted
    everywhere it occurs. Returns the texts unchanged if no known name is
    present."""
    present = _find_present_character_names(texts)
    if not present:
        return texts
    # Exclude the whole known-name vocabulary (not just the names present in
    # THIS scene) from the replacement draw so no substitution can ever
    # coincidentally read as a "leaked" character name during auditing.
    random_names = sample_random_names(
        scene_seed_key, len(present), exclude=set(_FRIENDS_CHARACTER_NAMES),
    )
    substitute: Optional[Callable[[str], str]] = build_name_substitution_regex(
        dict(zip(present, random_names))
    )
    if substitute is None:
        return texts
    return [substitute(t) for t in texts]


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


def _to_conversation(
    messages: List[tuple],
    dataset_split: str = "",
    conversation_id: str = "",
) -> Conversation | None:
    users = [user for user, _ in messages]
    contents = _substitute_character_names(
        [content for _, content in messages], conversation_id or "meld-scene",
    )

    speakers: List[str] = []
    turns: List[Turn] = []
    for user, content in zip(users, contents):
        if user not in speakers:
            speakers.append(user)
        turns.append(Turn(speaker=speakers.index(user), text=content, role=ROLE_PEER))
    if len(turns) >= 2 and len(speakers) >= 2:
        return Conversation(
            source="meld",
            speakers=speakers,
            turns=turns,
            content_bearing=True,
            meta={
                "dataset_split": dataset_split,
                "conversation_id": conversation_id,
                "group_id": conversation_id,
            },
        )
    return None


def _load_splits(raw_dir: str):
    """Yield ``(split, rows)`` for every available HF split."""
    from datasets import get_dataset_split_names, load_dataset

    try:
        splits = get_dataset_split_names("neoluigi/MELD-MPCA")
    except Exception:
        splits = ["train"]
    for split in splits:
        ds = load_dataset("neoluigi/MELD-MPCA", split=split, cache_dir=raw_dir)
        yield split, ds


def iter_conversations(raw_dir: str) -> Iterator[Conversation]:
    split_rows = list(_load_splits(raw_dir))
    has_validation = any(split in ("validation", "valid", "dev") for split, _ in split_rows)
    for split, rows in split_rows:
        normalized_split = "validation" if split in ("dev", "valid") else split
        for messages in reconstruct_dialogues(rows):
            digest = hashlib.sha256(
                json.dumps(messages, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            # MELD-MPCA currently exposes one split and no test split at all.
            # Reserve deterministic, conversation-level, disjoint 5% validation
            # and 5% test subsets when no official ones are available, avoiding
            # the old post-conversion random split. Bucket 0 -> validation,
            # bucket 1 -> test (held out, never trained on), buckets 2-19 -> train.
            example_split = normalized_split
            if normalized_split == "train" and not has_validation:
                bucket = int(digest[:8], 16) % 20
                if bucket == 0:
                    example_split = "validation"
                elif bucket == 1:
                    example_split = "test"
                else:
                    example_split = "train"
            conversation_id = f"meld-{digest[:16]}"
            conv = _to_conversation(
                messages,
                dataset_split=example_split,
                conversation_id=conversation_id,
            )
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
