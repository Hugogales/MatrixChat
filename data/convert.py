"""Convert a :class:`~data.schema.Conversation` into a MatrixChat training example.

Layout (turn-major)
-------------------
Each turn occupies a contiguous block of TIME columns owned by its speaker's
row. Every other agent row holds the silence token in those columns. So with one
active speaker per column the matrix is ~1/A dense -- exactly the sparse case the
``drop_silence`` compaction in ``model/matrix_qwen.py`` was built for.

No EOS delimiter between turns. Instead, a random gap of ``[min_turn_gap,
max_turn_gap]`` all-silence columns is inserted after each turn. Turn ends are
signaled by the speaker going silent, not by an ``<|im_end|>`` token. The
variable spacing stops the model from keying on a fixed delimiter and lets it
learn timing -- interrupting (short/zero gap) vs. yielding (longer gap).

Labels
------
- CONTENT (next-token, shifted over the agent's OWN row): placed ONLY on the
  target seat's turns, and only for ``content_bearing`` sources.
  ``label[a, t] = input[a, t+1]`` across the target's turn, so the last real
  token predicts the following cell in that row -- which is silence (the turn
  ended). That teaches the agent to STOP by going silent (no EOS needed).
  The silent cell right BEFORE the turn is labeled with the turn's FIRST token
  (floor-taking): "start speaking from silence". Non-target turns and
  structure-only sources carry NO content labels.
- SILENCE: EVERY silent cell is supervised with the silence token (no
  subsampling). To avoid the model collapsing to "always silent", the silence
  token's loss is DOWN-WEIGHTED at training time via
  ``MatrixQwenConfig.silence_loss_weight`` -- not by dropping labels here.
- Everything else (non-target speakers' own words = context) is
  ``ignore_index`` (-100).

The matrix forward uses an UNSHIFTED custom cross-entropy, so this converter must
emit already-shifted labels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .schema import Conversation, ROLE_ASSISTANT

Tokenize = Callable[[str], List[int]]


@dataclass
class ConvertConfig:
    """Knobs for matrix conversion (mirror the training hyperparameters)."""

    silence_token_id: int
    eos_token_id: int  # currently unused (turns end via silence, not EOS); kept for future use
    max_agents: int = 8
    # Random inter-turn gap: number of all-silence columns inserted after each
    # turn is drawn uniformly from [min_turn_gap, max_turn_gap]. Replaces the old
    # EOS delimiter so the model learns variable turn timing (interruptions vs.
    # yielding) rather than a fixed delimiter cue.
    min_turn_gap: int = 0
    max_turn_gap: int = 3
    permute_agents: bool = True
    ignore_index: int = -100


def _select_target(conv: Conversation, appear: List[int], rng: random.Random) -> Optional[int]:
    """Pick the target seat (the speaker whose content is supervised)."""
    if not conv.content_bearing:
        return None
    if conv.target_speaker is not None and conv.target_speaker in appear:
        return conv.target_speaker
    # Prefer an explicit assistant role if present (so "assistant-only content"
    # holds for sources that have one), else rotate over speakers.
    assistants = [
        sp for sp in appear
        if any(t.speaker == sp and t.role == ROLE_ASSISTANT for t in conv.turns)
    ]
    if assistants:
        return assistants[0]
    return rng.choice(appear)


def convert_conversation(
    conv: Conversation,
    tokenize: Tokenize,
    cfg: ConvertConfig,
    source_id: int,
    rng: random.Random,
) -> Optional[Dict]:
    """Convert one conversation to a matrix example.

    Returns a dict with ``input_ids``, ``labels``, ``agent_ids`` (each ``[A, T]``
    nested lists), ``source_id`` and ``length``. Returns ``None`` if the
    conversation has no usable turns.
    """
    # Distinct speakers in first-appearance order, capped at max_agents.
    appear: List[int] = []
    for turn in conv.turns:
        if turn.speaker not in appear:
            appear.append(turn.speaker)
    if len(appear) > cfg.max_agents:
        appear = appear[: cfg.max_agents]
    keep = set(appear)
    turns = [t for t in conv.turns if t.speaker in keep]
    if not turns:
        return None

    num_agents = len(appear)

    # Map each speaker -> a matrix row, optionally permuted (so the model never
    # binds a fixed row index to a role/speaker).
    base_rows = list(range(num_agents))
    if cfg.permute_agents:
        rng.shuffle(base_rows)
    row_of = {sp: base_rows[i] for i, sp in enumerate(appear)}

    target = _select_target(conv, appear, rng)
    target_row = row_of[target] if target is not None else None

    # Tokenize each turn (NO EOS -- turns end by going silent) and draw a random
    # all-silence gap after each turn.
    turn_tokens: List[List[int]] = []
    turn_rows: List[int] = []
    turn_is_target: List[bool] = []
    for t in turns:
        ids = list(tokenize(t.text))
        if not ids:
            continue
        turn_tokens.append(ids)
        turn_rows.append(row_of[t.speaker])
        turn_is_target.append(t.speaker == target)

    if not turn_tokens:
        return None

    gaps = [rng.randint(cfg.min_turn_gap, cfg.max_turn_gap) for _ in turn_tokens]
    T = sum(len(tt) for tt in turn_tokens) + sum(gaps)
    if T == 0:
        return None

    A = num_agents
    input_ids = [[cfg.silence_token_id] * T for _ in range(A)]
    labels = [[cfg.ignore_index] * T for _ in range(A)]
    agent_ids = [[a] * T for a in range(A)]

    # Place turn tokens; gap columns are simply left as the silence default.
    turn_spans: List[tuple] = []  # (row, start, end, is_target)
    col = 0
    for ids, row, is_target, gap in zip(turn_tokens, turn_rows, turn_is_target, gaps):
        start = col
        for tok in ids:
            input_ids[row][col] = tok
            col += 1
        turn_spans.append((row, start, col, is_target))
        col += gap  # all-silence gap for every agent

    # Content labels: shift over the target's own row. The last real token maps
    # to the following cell (silence, since the turn ended) -> teaches "stop".
    for row, start, end, is_target in turn_spans:
        if not is_target or target_row is None:
            continue
        for c in range(start, end):
            if c + 1 < T:
                labels[row][c] = input_ids[row][c + 1]

    # Floor-taking: supervise the silent cell right BEFORE a target turn to emit
    # the turn's FIRST token, teaching the model to *start speaking* from silence
    # (the onset that the within-turn shift never covers). Only when that cell is
    # actually silent (skip consecutive same-speaker turns with no gap).
    for row, start, end, is_target in turn_spans:
        if not is_target or target_row is None:
            continue
        if start >= 1 and input_ids[row][start - 1] == cfg.silence_token_id:
            labels[row][start - 1] = input_ids[row][start]

    # Supervise EVERY silent cell with the silence token (no subsampling). The
    # silence token's loss is down-weighted at training time
    # (MatrixQwenConfig.silence_loss_weight) to avoid over-silent behavior.
    for a in range(A):
        row_in = input_ids[a]
        row_lb = labels[a]
        for c in range(T):
            if row_lb[c] == cfg.ignore_index and row_in[c] == cfg.silence_token_id:
                row_lb[c] = cfg.silence_token_id

    return {
        "input_ids": input_ids,
        "labels": labels,
        "agent_ids": agent_ids,
        "source_id": source_id,
        "length": T,
        "num_agents": A,
    }
