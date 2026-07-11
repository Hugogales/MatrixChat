"""Werewolf-Among-Us adapter (social-deduction game transcripts).

Source: bolinlai/Werewolf-Among-Us (HF). 199 games of One Night Werewolf / Avalon
with speaker-annotated transcripts. Many players -> strong multi-party structure
and a bridge toward the Werewolf RL endgame. Structure-only for now (small,
spoken-transcript disfluencies): no content loss.

Transcripts are JSON with per-utterance ``speaker`` + ``utterance``/``text``.
Parsing is defensive; confirm against the downloaded copy.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterator

from .spec import AdapterSpec
from ..schema import Conversation, Turn, ROLE_PEER


def _iter_transcript_files(raw_dir: str):
    base = os.path.join(raw_dir, "werewolf")
    yield from sorted(glob.glob(os.path.join(base, "**", "*.json"), recursive=True))


def _utterances(obj):
    if isinstance(obj, list):
        return obj
    for key in ("transcript", "utterances", "dialogue"):
        val = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(val, list):
            return val
    return []


def iter_conversations(raw_dir: str) -> Iterator[Conversation]:
    for path in _iter_transcript_files(raw_dir):
        with open(path, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
            except json.JSONDecodeError:
                continue
        utts = _utterances(obj)
        speakers: list[str] = []
        turns: list[Turn] = []
        for u in utts:
            if not isinstance(u, dict):
                continue
            spk = str(u.get("speaker") or u.get("player") or "unknown")
            text = (u.get("utterance") or u.get("text") or u.get("content") or "").strip()
            if not text:
                continue
            if spk not in speakers:
                speakers.append(spk)
            turns.append(Turn(speaker=speakers.index(spk), text=text, role=ROLE_PEER))
        if len(turns) >= 2 and len(speakers) >= 2:
            yield Conversation(
                source="werewolf",
                speakers=speakers,
                turns=turns,
                content_bearing=False,
            )


SPEC = AdapterSpec(
    source="werewolf",
    content_bearing=False,
    default_weight=0.10,
    iter_conversations=iter_conversations,
    hf_repo="bolinlai/Werewolf-Among-Us",
    hf_snapshot=True,  # file-based repo: snapshot into raw_dir/werewolf, parse files
    notes="Structure/domain tier: social-deduction games. Bridge to Werewolf RL.",
)
