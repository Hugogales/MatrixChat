"""Self-distilled Qwen adapter (knowledge/behavior retention anchor).

Produced by ``scripts/data_prep/distill_qwen.sbatch``: run Qwen3-4B on prompts and store
its own responses. Training MatrixQwen to reproduce these through the matrix
interface preserves Qwen's general behavior (anti-forgetting). Content-bearing,
with the assistant turn as the natural target seat.

Raw layout: JSONL at ``<raw_dir>/qwen_distill/*.jsonl`` with one object per line:
``{"prompt": str, "response": str}`` or ``{"messages": [{"role","content"}, ...]}``.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterator

from .spec import AdapterSpec
from ..schema import Conversation, Turn, ROLE_USER, ROLE_ASSISTANT


def _files(raw_dir: str):
    base = os.path.join(raw_dir, "qwen_distill")
    return sorted(glob.glob(os.path.join(base, "**", "*.jsonl"), recursive=True))


def _conv_from_messages(messages) -> Conversation | None:
    speakers: list[str] = []
    turns: list[Turn] = []
    for m in messages:
        role = m.get("role", ROLE_USER)
        text = (m.get("content") or "").strip()
        if not text:
            continue
        spk = role  # one speaker per role
        if spk not in speakers:
            speakers.append(spk)
        turns.append(Turn(speaker=speakers.index(spk), text=text, role=role))
    if len(turns) >= 2:
        return Conversation(
            source="qwen_distill", speakers=speakers, turns=turns, content_bearing=True
        )
    return None


def iter_conversations(raw_dir: str) -> Iterator[Conversation]:
    for path in _files(raw_dir):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "messages" in obj:
                    conv = _conv_from_messages(obj["messages"])
                    if conv is not None:
                        yield conv
                    continue
                prompt = (obj.get("prompt") or "").strip()
                response = (obj.get("response") or "").strip()
                if not prompt or not response:
                    continue
                yield Conversation(
                    source="qwen_distill",
                    speakers=[ROLE_USER, ROLE_ASSISTANT],
                    turns=[
                        Turn(speaker=0, text=prompt, role=ROLE_USER),
                        Turn(speaker=1, text=response, role=ROLE_ASSISTANT),
                    ],
                    content_bearing=True,
                )


SPEC = AdapterSpec(
    source="qwen_distill",
    content_bearing=True,
    default_weight=0.30,
    iter_conversations=iter_conversations,
    notes="Retention tier: self-distilled Qwen3-4B responses. Assistant = target seat.",
)
