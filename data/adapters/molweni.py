"""Molweni adapter (Ubuntu multi-party chat).

Source: HIT-SCIR/Molweni (GitHub). ~10k dialogues, avg ~3.5 speakers (2-9).
Structure-only: teaches multi-party turn-taking/silence; NOT used for content
loss (Ubuntu jargon + typos). Raw layout: train/dev/test JSON files, each a list
of dialogues with ``edus`` = [{"speaker": str, "text": str}, ...].

The exact field names should be confirmed against the downloaded copy on Rosie;
parsing here is defensive.

Grammar/spelling cleaning (2026-08-24): raw Molweni text is real Ubuntu IRC
support chat, with the typos/informality that implies. Since this source is
``content_bearing=False`` (see ``docs/RESEARCH_REFERENCE.md`` section 4.4),
that text is NEVER a content-loss supervision target -- cleaning it cannot
change what the model is trained to GENERATE. It IS still fed in as the raw
input tokens the shared hidden representation (which the activity/turn-
taking head also reads) is built from, so a conservative cleaning pass is
applied by default (``clean=True``) for that second-order input-quality
benefit; see ``data/adapters/_text_cleaning.py`` for exactly what it does
and does not touch. Pass ``clean=False`` to inspect the raw text unmodified.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterator

from .spec import AdapterSpec
from ._text_cleaning import clean_text
from ..schema import Conversation, Turn, ROLE_PEER


def _raw_glob(raw_dir: str):
    base = os.path.join(raw_dir, "molweni")
    return sorted(glob.glob(os.path.join(base, "**", "*.json"), recursive=True))


def iter_conversations(raw_dir: str, clean: bool = True) -> Iterator[Conversation]:
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
                if clean:
                    text = clean_text(text)
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
