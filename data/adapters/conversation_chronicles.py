"""Conversation Chronicles adapter (large casual multi-session, ~2 speakers).

Source: jihyoung/ConversationChronicles (HF). 1M sessions; mostly dyadic but
casual and high-volume -> breadth tier, content-bearing. Each row has session
dialogue lists (``first_session_dialogue`` ...) with parallel speaker lists
(``first_session_speakers`` ...). We emit one Conversation per session.
"""

from __future__ import annotations

from typing import Iterator

from .spec import AdapterSpec
from ..schema import Conversation, Turn, ROLE_PEER

_SESSION_KEYS = [
    ("first_session_dialogue", "first_session_speakers"),
    ("second_session_dialogue", "second_session_speakers"),
    ("third_session_dialogue", "third_session_speakers"),
    ("fourth_session_dialogue", "fourth_session_speakers"),
    ("fifth_session_dialogue", "fifth_session_speakers"),
]


def iter_conversations(raw_dir: str) -> Iterator[Conversation]:
    from datasets import load_dataset

    ds = load_dataset("jihyoung/ConversationChronicles", split="train", cache_dir=raw_dir)
    for row in ds:
        for dia_key, spk_key in _SESSION_KEYS:
            dialogue = row.get(dia_key) or []
            spk_list = row.get(spk_key) or []
            if not dialogue:
                continue
            speakers: list[str] = []
            turns: list[Turn] = []
            for i, text in enumerate(dialogue):
                text = (text or "").strip()
                if not text:
                    continue
                spk = str(spk_list[i]) if i < len(spk_list) else f"spk{i % 2}"
                if spk not in speakers:
                    speakers.append(spk)
                turns.append(Turn(speaker=speakers.index(spk), text=text, role=ROLE_PEER))
            if len(turns) >= 2 and len(speakers) >= 2:
                yield Conversation(
                    source="conversation_chronicles",
                    speakers=speakers,
                    turns=turns,
                    content_bearing=True,
                )


SPEC = AdapterSpec(
    source="conversation_chronicles",
    content_bearing=True,
    default_weight=0.20,
    iter_conversations=iter_conversations,
    hf_repo="jihyoung/ConversationChronicles",
    notes="Breadth tier: large casual dyadic multi-session. Content-bearing.",
)
