"""Per-dataset adapters: raw source -> unified :class:`~data.schema.Conversation`.

Each adapter module exposes a :class:`~data.adapters.spec.AdapterSpec`. The
registry below assigns append-only integer ``source_id`` values used to tag
every example and drive the per-dataset loss breakdown.
"""

from __future__ import annotations

from .spec import AdapterSpec
from . import (
    ami,
    molweni,
    meld,
    werewolf,
    conversation_chronicles,
    qwen_distill,
    when2speak,
    bazinga,
)


_SPECS = [
    ami.SPEC,
    molweni.SPEC,
    meld.SPEC,
    werewolf.SPEC,
    conversation_chronicles.SPEC,
    qwen_distill.SPEC,
    when2speak.SPEC,
    bazinga.SPEC,
]

# Preserve IDs already baked into existing Arrow shards; new sources receive
# explicit append-only IDs on their AdapterSpec.
_LEGACY_SOURCE_IDS = {
    "conversation_chronicles": 0,
    "meld": 1,
    "molweni": 2,
    "qwen_distill": 3,
    "werewolf": 4,
}

SOURCES = {spec.source: spec for spec in _SPECS}
SOURCE_IDS = {
    name: (
        spec.source_id
        if spec.source_id is not None
        else _LEGACY_SOURCE_IDS[name]
    )
    for name, spec in SOURCES.items()
}
if len(set(SOURCE_IDS.values())) != len(SOURCE_IDS):
    raise RuntimeError(f"Duplicate dataset source IDs: {SOURCE_IDS}")
ID_TO_SOURCE = {i: name for name, i in SOURCE_IDS.items()}


def get_spec(source: str) -> AdapterSpec:
    if source not in SOURCES:
        raise KeyError(f"Unknown source {source!r}. Known: {sorted(SOURCES)}")
    return SOURCES[source]


def default_weights() -> dict:
    return {name: spec.default_weight for name, spec in SOURCES.items()}
