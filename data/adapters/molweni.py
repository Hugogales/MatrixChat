"""Molweni adapter (Ubuntu multi-party chat).

Source: HIT-SCIR/Molweni (GitHub). ~10k dialogues, avg ~3.5 speakers (2-9).
Structure-only: teaches multi-party turn-taking/silence; NOT used for content
loss (Ubuntu jargon + typos). Raw layout: train/dev/test JSON files, each a list
of dialogues with ``edus`` = [{"speaker": str, "text": str}, ...].

The exact field names should be confirmed against the downloaded copy on Rosie;
parsing here is defensive.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterator

from .spec import AdapterSpec
from ..schema import Conversation, Turn, ROLE_PEER


def _raw_glob(raw_dir: str):
    base = os.path.join(raw_dir, "molweni")
    return sorted(glob.glob(os.path.join(base, "**", "*.json"), recursive=True))


def iter_conversations(raw_dir: str) -> Iterator[Conversation]:
    for path in _raw_glob(raw_dir):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        dialogues = data if isinstance(data, list) else data.get("dialogues", [])
        for d in dialogues:
            edus = d.get("edus") or d.get("utterances") or []
            speakers: list[str] = []
            turns: list[Turn] = []
            for edu in edus:
                spk = str(edu.get("speaker", "unknown"))
                text = (edu.get("text") or edu.get("utterance") or "").strip()
                if not text:
                    continue
                if spk not in speakers:
                    speakers.append(spk)
                turns.append(Turn(speaker=speakers.index(spk), text=text, role=ROLE_PEER))
            if len(turns) >= 2 and len(speakers) >= 2:
                yield Conversation(
                    source="molweni",
                    speakers=speakers,
                    turns=turns,
                    content_bearing=False,
                )


SPEC = AdapterSpec(
    source="molweni",
    content_bearing=False,
    default_weight=0.25,
    iter_conversations=iter_conversations,
    git_url="https://github.com/HIT-SCIR/Molweni",
    notes="Structure tier: multi-party Ubuntu chat (3+ speakers). No content loss.",
)
