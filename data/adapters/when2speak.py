"""When2Speak adapter (grounded-synthetic multi-party intervention timing).

Source: duke-trust-lab/When2Speak (HF, CC BY 4.0). 216,800 (context, decision)
pairs derived from 16,000 GPT-4-Turbo-generated multi-party transcripts (2-6
anonymized human speakers -- ``Speaker_0``, ``Speaker_1``, ... -- plus one
embedded AI agent, ``[AGENT]``), grounded topically in Yahoo Answers. See
``docs/RESEARCH_REFERENCE.md`` for the full evaluation writeup.

Why ``content_bearing=False``: every conversation here is LLM-generated
(GPT-4o-mini social-context annotation + GPT-4-Turbo transcript synthesis),
not human-authored. Per the project's LLM-judge-artifact finding (fluent LLM
text scores higher than disfluent human transcripts on automated metrics),
using this as a CONTENT supervision target risks teaching the model to
imitate synthetic-fluency patterns rather than real human turn-taking. It is
used exactly like Molweni/Werewolf: structure-only. The turn-taking signal
itself -- when a genuinely varied-size group (2-6 peers) yields the floor to
a distinguished participant vs. stays silent -- is real supervision the
model benefits from regardless of who authored the words.

Raw format (HF ``dialogue`` config, OpenAI chat format per row)::

    {"messages": [
        {"role": "user", "content": "Speaker_0: ..."},
        {"role": "user", "content": "Speaker_1: ..."},
        {"role": "assistant", "content": ">"}            # SILENT, or
        {"role": "assistant", "content": "<full text>"}  # SPEAK
    ]}

We use the ``dialogue`` config (not ``token``) so SPEAK turns carry the
agent's real generated utterance as raw input text, not a bare ``<``
placeholder -- consistent with Molweni's rationale that even non-supervised
raw text still shapes the shared hidden representation the activity/turn-
taking head reads (see ``data/adapters/_text_cleaning.py`` docstring).

**Window-overlap dedup (2026-08-25, revised after measuring the real HF
data)**: each row is one position of a sliding window over the original
transcript (window caps out at 9 messages), and rows for the SAME base
transcript are emitted consecutively. Measured directly against a 20,000-row
sample: **88% of consecutive rows are near-total supersets of their
neighbor** (share all but 0-2 messages), and per-transcript chain lengths
range up to 65 -- i.e. a single long synthetic conversation can flood the
dataset with dozens of near-duplicate ``Conversation``s that differ by only
one appended message, while a short one contributes just 1-2. Left
unaddressed, this doesn't just "over-represent SILENT" (the originally
documented tradeoff) -- it lets a handful of long transcripts dominate
When2Speak's entire contribution to training, independent of topic
diversity or the SPEAK/SILENT balance.

Fix: detect chain boundaries via consecutive-row content overlap (no
conversation id is released, so this is a heuristic, not exact -- a shared-
content check between adjacent rows, since genuinely unrelated transcripts
essentially never share 7+ of 9 messages verbatim), then **randomly sample
at most ``max_windows_per_chain`` rows per detected chain** (default 4,
seeded for reproducibility). Measured effect on a 20,000-row sample:
~3.1x compression (20,000 -> 6,515 rows), worst-case per-transcript
contribution capped at 4 (down from 65), and SPEAK share moves from 13.2%
(raw) to 19.7% (capped+random) -- much closer to the natural rate than
either "keep everything" (no capping, but also doesn't fix the long-chain
dominance) or "keep only the LAST window per chain" (which we measured
skews to 73.4% SPEAK -- the last captured window in a chain is not a
representative sample of that chain's decisions, so plain dedup-to-one is
worse than capped random sampling, not just smaller).
"""

from __future__ import annotations

import random
import re
from typing import Iterator, List

from .spec import AdapterSpec
from ..schema import Conversation, Turn, ROLE_PEER

DEFAULT_MAX_WINDOWS_PER_CHAIN = 4
DEFAULT_DEDUP_SEED = 0
# Two consecutive rows are the "same chain" (same underlying transcript, one
# sliding-window step apart) if they share all but a couple of messages --
# i.e. one message dropped from the front and/or one appended at the back.
# Genuinely different transcripts essentially never hit this by chance.
_CHAIN_OVERLAP_SLACK = 2

_SPEAKER_PREFIX = re.compile(r"^(Speaker_\d+):\s?(.*)$", re.DOTALL)
# A prior window's AGENT turn, reappearing as later-window CONTEXT, is
# formatted as a "user"-role message with an "Assistant: " prefix instead of
# "Speaker_N: " -- confirmed against the real downloaded data (2026-08-25):
# a role="assistant" turn's real text becomes role="user", content=
# "Assistant: <same text>" once it slides into a later window's context.
# Without this, `_split_speaker` fell through to speaker="unknown" with the
# literal "Assistant: " prefix leaked into the turn text.
_ASSISTANT_CONTEXT_PREFIX = re.compile(r"^Assistant:\s?(.*)$", re.DOTALL)
_AGENT_LABEL = "[AGENT]"
_SILENT_TOKEN = ">"


def _split_speaker(content: str) -> tuple[str, str]:
    content = content or ""
    match = _SPEAKER_PREFIX.match(content)
    if match:
        return match.group(1), match.group(2).strip()
    match = _ASSISTANT_CONTEXT_PREFIX.match(content)
    if match:
        return _AGENT_LABEL, match.group(1).strip()
    return "unknown", content.strip()


def _row_contents(row: dict) -> List[str]:
    return [m.get("content", "") for m in (row.get("messages") or [])]


def _same_chain(prev_contents: List[str], curr_contents: List[str]) -> bool:
    if not prev_contents or not curr_contents:
        return False
    shared = len(set(curr_contents) & set(prev_contents))
    # `shared > 0` guards very short windows (e.g. both length 1-2), where
    # `min_len - slack` alone can go <= 0 and would otherwise call ANY two
    # rows the same chain even with zero actual overlap.
    return shared > 0 and shared >= min(len(curr_contents), len(prev_contents)) - _CHAIN_OVERLAP_SLACK


def _iter_deduped_rows(
    ds, max_windows_per_chain: int, seed: int
) -> Iterator[dict]:
    """Group consecutive same-chain rows (see module docstring) and randomly
    sample at most ``max_windows_per_chain`` from each, so one long synthetic
    transcript can't flood the dataset with near-duplicate windows."""
    rng = random.Random(seed)
    prev_contents: List[str] = []
    chain: List[dict] = []

    def flush() -> Iterator[dict]:
        if not chain:
            return
        if len(chain) <= max_windows_per_chain:
            yield from chain
        else:
            yield from rng.sample(chain, max_windows_per_chain)

    for row in ds:
        contents = _row_contents(row)
        if chain and not _same_chain(prev_contents, contents):
            yield from flush()
            chain = []
        chain.append(row)
        prev_contents = contents
    yield from flush()


def _row_to_conversation(row: dict) -> "Conversation | None":
    messages = row.get("messages") or []
    if len(messages) < 2:
        return None
    *context, last = messages
    speakers: List[str] = []
    turns: List[Turn] = []
    for msg in context:
        if msg.get("role") != "user":
            continue
        spk, text = _split_speaker(msg.get("content", ""))
        if not text:
            continue
        if spk not in speakers:
            speakers.append(spk)
        turns.append(Turn(speaker=speakers.index(spk), text=text, role=ROLE_PEER))
    if last.get("role") == "assistant":
        label = (last.get("content") or "").strip()
        if label and label != _SILENT_TOKEN:
            if _AGENT_LABEL not in speakers:
                speakers.append(_AGENT_LABEL)
            turns.append(
                Turn(speaker=speakers.index(_AGENT_LABEL), text=label, role=ROLE_PEER)
            )
    if len(turns) >= 2 and len(speakers) >= 2:
        return Conversation(
            source="when2speak",
            speakers=speakers,
            turns=turns,
            content_bearing=False,
        )
    return None


def _iter_split(
    ds, max_windows_per_chain: int = DEFAULT_MAX_WINDOWS_PER_CHAIN, seed: int = DEFAULT_DEDUP_SEED
) -> Iterator[Conversation]:
    for row in _iter_deduped_rows(ds, max_windows_per_chain, seed):
        conv = _row_to_conversation(row)
        if conv is not None:
            yield conv


def iter_conversations(
    raw_dir: str,
    max_windows_per_chain: int = DEFAULT_MAX_WINDOWS_PER_CHAIN,
    seed: int = DEFAULT_DEDUP_SEED,
) -> Iterator[Conversation]:
    from datasets import load_dataset

    ds = load_dataset("duke-trust-lab/When2Speak", "dialogue", cache_dir=raw_dir)
    for split_name in ("train", "test", "validation"):
        if split_name in ds:
            yield from _iter_split(ds[split_name], max_windows_per_chain, seed)


SPEC = AdapterSpec(
    source="when2speak",
    source_id=6,
    content_bearing=False,
    default_weight=0.25,
    iter_conversations=iter_conversations,
    hf_repo="duke-trust-lab/When2Speak",
    hf_config="dialogue",
    notes=(
        "Structure tier: grounded-synthetic multi-party (2-6 peers + 1 agent) "
        "SPEAK/SILENT decisions. No content loss (LLM-generated text)."
    ),
)
