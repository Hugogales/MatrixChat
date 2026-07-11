"""Per-dataset adapters: raw source -> unified :class:`~data.schema.Conversation`.

Each adapter module exposes a :class:`~data.adapters.spec.AdapterSpec`. The
registry below assigns a stable integer ``source_id`` (sorted by name) used to
tag every example and to drive the per-dataset loss breakdown.
"""

from __future__ import annotations

from .spec import AdapterSpec
from . import molweni, meld, werewolf, conversation_chronicles, qwen_distill


_SPECS = [
    molweni.SPEC,
    meld.SPEC,
    werewolf.SPEC,
    conversation_chronicles.SPEC,
    qwen_distill.SPEC,
]

# Registry name -> spec, with a stable source_id (sorted by name for determinism).
SOURCES = {spec.source: spec for spec in _SPECS}
SOURCE_IDS = {name: i for i, name in enumerate(sorted(SOURCES))}
ID_TO_SOURCE = {i: name for name, i in SOURCE_IDS.items()}


def get_spec(source: str) -> AdapterSpec:
    if source not in SOURCES:
        raise KeyError(f"Unknown source {source!r}. Known: {sorted(SOURCES)}")
    return SOURCES[source]


def default_weights() -> dict:
    return {name: spec.default_weight for name, spec in SOURCES.items()}
