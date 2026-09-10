"""Convert a :class:`~data.schema.Conversation` into a MatrixChat training example.

Layout (turn-major)
-------------------
Each turn occupies a contiguous block of TIME columns owned by its speaker's
row. Every other agent row is INACTIVE in those columns (``input_activity_mask
= False``); their input cell holds an arbitrary ``placeholder_token_id`` that
the model wrapper always overrides with a learned "inactive" embedding (see
``model/matrix_qwen.py``). So with one active speaker per column the matrix is
~1/A dense.

No EOS delimiter between turns. Instead, a random gap of ``[min_turn_gap,
max_turn_gap]`` all-inactive columns is inserted after each turn. Turn ends are
signaled by the speaker going inactive, not by an ``<|im_end|>`` token, and are
learned by the ACTIVITY head, not the vocabulary. The variable spacing stops
the model from keying on a fixed delimiter and lets it learn timing --
interrupting (short/zero gap) vs. yielding (longer gap).

Labels
------
- ``labels`` (content, next-token, shifted over the agent's OWN row): placed
  on either only the target seat's turns (the default) or every speaking row,
  according to ``content_supervision_mode`` (and only for ``content_bearing`` sources),
  and ONLY where the agent continues speaking into the next column (i.e. never
  on the last token of a turn -- "should I stop" is an ACTIVITY decision, not a
  vocabulary one). Non-target turns and structure-only sources carry NO content
  labels.
- ``activity_labels`` (binary, next-column, EVERY agent/cell): whether the
  agent is active at column ``t+1``, predicted from context up to and
  including column ``t``. This single rule captures floor-taking (inactive ->
  active), continuing (active -> active), and stopping (active -> inactive) in
  one formula, for ALL agents (not just the target), since who-speaks-when is
  known ground truth for every source. The last column has no successor and is
  ``ignore_index``.
- Everything else in ``labels`` (non-target speakers' own words = context) is
  ``ignore_index`` (-100).

The matrix forward uses an UNSHIFTED custom cross-entropy for content, so this
converter must emit already-shifted content labels.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from .conversation_quality import attach_quality_fields
from .schema import Conversation, ROLE_ASSISTANT, TimedWord

Tokenize = Callable[[str], List[int]]

CONTENT_SUPERVISION_MODES = ("target_only", "all_speakers")


@dataclass
class ConvertConfig:
    """Knobs for matrix conversion (mirror the training hyperparameters)."""

    max_agents: int = 8
    # Dummy input id for inactive cells (always overridden by the model's
    # learned inactive-cell embedding; the value itself is otherwise unused).
    placeholder_token_id: int = 0
    # Random inter-turn gap: number of all-inactive columns inserted after each
    # turn is drawn uniformly from [min_turn_gap, max_turn_gap]. Lets the model
    # learn variable turn timing (interruptions vs. yielding) rather than a
    # fixed delimiter cue.
    min_turn_gap: int = 0
    max_turn_gap: int = 3
    permute_agents: bool = True
    ignore_index: int = -100
    # Timed-corpus conversion. Ordinary within-speech timing is compressed;
    # only cross-speaker overlap and meaningful all-speaker pauses are retained.
    pause_threshold_seconds: float = 1.0
    pause_quantum_seconds: float = 1.0
    max_pause_columns: int = 8
    max_flat_len: int = 1536
    # Timed corpora (AMI/Werewolf) are long enough to need chunking, which
    # means every chunk after the first would otherwise start "cold" -- no
    # tokens at all from the conversation that actually precedes it, even
    # though in reality it's mid-conversation. When > 0, each non-first chunk
    # is prefixed with up to this many REAL columns copied from immediately
    # before its start (real input_ids/activity, not synthetic filler) so the
    # model has genuine history to condition generation/attention on. These
    # prefix columns are CONTEXT ONLY: their content/activity labels are
    # ignore_index (they were already the label-bearing "new" portion of the
    # previous chunk, so re-labeling them here would just be redundant
    # compute, not new signal). 0 (default) reproduces the prior behavior
    # exactly -- every chunk is still built from scratch with no lookback.
    context_lookback_columns: int = 0
    # Which speaking rows receive content labels. ``target_only`` preserves
    # the original assistant/preselected/random-target behavior.
    # ``all_speakers`` supervises every speaking row in content-bearing data.
    content_supervision_mode: str = "target_only"

    def __post_init__(self) -> None:
        if self.content_supervision_mode not in CONTENT_SUPERVISION_MODES:
            choices = ", ".join(CONTENT_SUPERVISION_MODES)
            raise ValueError(
                f"content_supervision_mode must be one of: {choices}; "
                f"got {self.content_supervision_mode!r}"
            )


def _content_rows(
    conv: Conversation,
    cfg: ConvertConfig,
    num_agents: int,
    target_row: Optional[int],
) -> List[int]:
    """Return rows eligible for next-token content supervision."""
    if not conv.content_bearing:
        return []
    if cfg.content_supervision_mode == "all_speakers":
        return list(range(num_agents))
    return [target_row] if target_row is not None else []


def _count_speaker_changes(
    input_activity_mask: List[List[int]],
    start_col: int = 0,
) -> int:
    """Count ground-truth floor acquisitions by a new speaker.

    Silent columns do not reset the prior floor. For chunked examples, a
    context-only prefix may establish the prior floor, but acquisitions are
    counted only when they occur at or after ``start_col``.
    """
    if not input_activity_mask:
        return 0
    num_columns = len(input_activity_mask[0])
    previous = set()
    changes = 0
    for col in range(num_columns):
        active = {
            row
            for row, row_mask in enumerate(input_activity_mask)
            if row_mask[col]
        }
        if not active:
            continue
        if col >= start_col and previous:
            changes += len(active - previous)
        previous = active
    return changes


def _has_private_turns(conv: Conversation) -> bool:
    return any(t.visible_to is not None for t in conv.turns)


def _agent_visibility_matrix(conv: Conversation, row_of: Dict[int, int], num_agents: int) -> List[List[int]]:
    """Build ``[A, A]`` visibility: ``matrix[owner][viewer] = 1`` if ``viewer``
    may attend to ``owner``'s private cells (self is always 1; public turns
    are not represented here -- they are visible to everyone regardless)."""
    matrix = [[0] * num_agents for _ in range(num_agents)]
    for a in range(num_agents):
        matrix[a][a] = 1
    for turn in conv.turns:
        if turn.visible_to is None or turn.speaker not in row_of:
            continue
        owner_row = row_of[turn.speaker]
        for viewer_speaker in turn.visible_to:
            viewer_row = row_of.get(viewer_speaker)
            if viewer_row is not None:
                matrix[owner_row][viewer_row] = 1
    return matrix


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

    Returns a dict with ``input_ids``, ``input_activity_mask``, ``labels``,
    ``activity_labels``, ``agent_ids`` (each ``[A, T]`` nested lists),
    ``source_id``, ``length`` and ``num_agents``. Returns ``None`` if the
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

    # Tokenize each turn (NO EOS -- turns end by going inactive) and draw a
    # random all-inactive gap after each turn.
    turn_tokens: List[List[int]] = []
    turn_rows: List[int] = []
    turn_is_target: List[bool] = []
    turn_private: List[bool] = []
    for t in turns:
        ids = list(tokenize(t.text))
        if not ids:
            continue
        turn_tokens.append(ids)
        turn_rows.append(row_of[t.speaker])
        turn_is_target.append(t.speaker == target)
        turn_private.append(t.visible_to is not None)

    if not turn_tokens:
        return None

    gaps = [rng.randint(cfg.min_turn_gap, cfg.max_turn_gap) for _ in turn_tokens]
    T = sum(len(tt) for tt in turn_tokens) + sum(gaps)
    if T == 0:
        return None

    A = num_agents
    input_ids = [[cfg.placeholder_token_id] * T for _ in range(A)]
    input_activity_mask = [[0] * T for _ in range(A)]
    labels = [[cfg.ignore_index] * T for _ in range(A)]
    activity_labels = [[cfg.ignore_index] * T for _ in range(A)]
    agent_ids = [[a] * T for a in range(A)]

    # Place turn tokens; gap columns are simply left inactive.
    has_private = _has_private_turns(conv)
    is_private_mask = [[0] * T for _ in range(A)] if has_private else None
    col = 0
    for ids, row, _is_target, is_private, gap in zip(
        turn_tokens, turn_rows, turn_is_target, turn_private, gaps
    ):
        for tok in ids:
            input_ids[row][col] = tok
            input_activity_mask[row][col] = 1
            if is_private_mask is not None and is_private:
                is_private_mask[row][col] = 1
            col += 1
        col += gap  # all-inactive gap for every agent

    # Activity labels: for EVERY agent, whether it is active at t+1, predicted
    # from context up to t. This single rule covers floor-taking, continuing,
    # and stopping. The last column has no successor -> ignore_index.
    for a in range(A):
        row_mask = input_activity_mask[a]
        row_lb = activity_labels[a]
        for c in range(T - 1):
            row_lb[c] = row_mask[c + 1]

    # Content labels: shift over each eligible row, only while continuing to
    # speak into the next column. The default eligible set is the target row;
    # all_speakers broadens it without changing the next-token rule.
    for content_row in _content_rows(conv, cfg, A, target_row):
        row_mask = input_activity_mask[content_row]
        row_in = input_ids[content_row]
        row_lb = labels[content_row]
        for c in range(T - 1):
            if row_mask[c] and row_mask[c + 1]:
                row_lb[c] = row_in[c + 1]

    example = {
        "input_ids": input_ids,
        "input_activity_mask": input_activity_mask,
        "labels": labels,
        "activity_labels": activity_labels,
        "agent_ids": agent_ids,
        "source_id": source_id,
        "length": T,
        "num_agents": A,
        "content_supervision_mode": cfg.content_supervision_mode,
        "num_speaker_changes": _count_speaker_changes(input_activity_mask),
    }
    if is_private_mask is not None:
        example["is_private_mask"] = is_private_mask
        example["agent_visibility"] = _agent_visibility_matrix(conv, row_of, A)
    return attach_quality_fields(example)


@dataclass
class _TimedToken:
    speaker: int
    token_id: int
    start: float
    end: float
    ordinal: int
    private: bool = False


@dataclass
class _TimeRegion:
    start: float
    end: float
    active: Tuple[int, ...]


def _timed_word_token_ids(
    word: TimedWord,
    tokenize: Tokenize,
    has_preceding_text: bool,
) -> List[int]:
    """Tokenize one timed terminal in approximately its original context.

    Qwen's tokenizer generally encodes the leading space with the following
    word. Punctuation is joined without one, matching the AMI text builder.
    """
    prefix = "" if word.is_punctuation or not has_preceding_text else " "
    return list(tokenize(prefix + word.text))


def _collect_timed_tokens(conv: Conversation, tokenize: Tokenize) -> List[_TimedToken]:
    tokens: List[_TimedToken] = []
    ordinal = 0
    for turn in conv.turns:
        previous = False
        is_private = turn.visible_to is not None
        for word in turn.timed_words:
            ids = _timed_word_token_ids(word, tokenize, previous)
            previous = previous or bool(ids)
            for token_id in ids:
                tokens.append(
                    _TimedToken(
                        speaker=turn.speaker,
                        token_id=token_id,
                        start=float(word.start_time),
                        end=max(float(word.start_time), float(word.end_time)),
                        ordinal=ordinal,
                        private=is_private,
                    )
                )
                ordinal += 1
    return tokens


def _speech_regions(conv: Conversation) -> List[_TimeRegion]:
    """Sweep timed word intervals into constant-active-speaker regions."""
    intervals: List[Tuple[float, float, int]] = []
    points = set()
    for turn in conv.turns:
        for word in turn.timed_words:
            start = float(word.start_time)
            end = max(start, float(word.end_time))
            if end <= start:
                continue
            intervals.append((start, end, turn.speaker))
            points.add(start)
            points.add(end)
    if not intervals:
        return []

    ordered = sorted(points)
    raw: List[_TimeRegion] = []
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        active = tuple(
            sorted(
                {
                    speaker
                    for start, end, speaker in intervals
                    if start < right and end > left
                }
            )
        )
        if raw and raw[-1].active == active and abs(raw[-1].end - left) < 1e-6:
            raw[-1].end = right
        else:
            raw.append(_TimeRegion(left, right, active))
    return raw


def _assign_tokens_to_regions(
    tokens: List[_TimedToken],
    regions: List[_TimeRegion],
) -> Dict[int, Dict[int, List[_TimedToken]]]:
    """Assign every token once to its best-overlapping compatible region."""
    assigned: Dict[int, Dict[int, List[_TimedToken]]] = {}
    for token in tokens:
        best_idx = None
        best_overlap = -1.0
        for idx, region in enumerate(regions):
            if token.speaker not in region.active:
                continue
            overlap = max(0.0, min(token.end, region.end) - max(token.start, region.start))
            if token.end == token.start and region.start <= token.start <= region.end:
                overlap = 1e-9
            if overlap > 0.0 and overlap > best_overlap:
                best_idx, best_overlap = idx, overlap
        if best_idx is None:
            # Punctuation/zero-duration terminals may sit exactly at a boundary.
            candidates = [
                (abs(region.start - token.start), idx)
                for idx, region in enumerate(regions)
                if token.speaker in region.active
            ]
            if not candidates:
                continue
            best_idx = min(candidates)[1]
        assigned.setdefault(best_idx, {}).setdefault(token.speaker, []).append(token)

    for by_speaker in assigned.values():
        for speaker_tokens in by_speaker.values():
            speaker_tokens.sort(key=lambda item: (item.start, item.end, item.ordinal))
    return assigned


def _pause_columns(duration: float, cfg: ConvertConfig) -> int:
    if duration <= cfg.pause_threshold_seconds:
        return 0
    quantum = max(cfg.pause_quantum_seconds, 1e-6)
    count = int(math.ceil((duration - cfg.pause_threshold_seconds) / quantum))
    return max(1, min(cfg.max_pause_columns, count))


def _render_timed_columns(
    conv: Conversation,
    tokenize: Tokenize,
    row_of: Dict[int, int],
    cfg: ConvertConfig,
) -> Tuple[List[Dict[int, int]], List[str], List[Dict[int, bool]]]:
    tokens = _collect_timed_tokens(conv, tokenize)
    regions = _speech_regions(conv)
    assigned = _assign_tokens_to_regions(tokens, regions)
    columns: List[Dict[int, int]] = []
    kinds: List[str] = []
    privates: List[Dict[int, bool]] = []

    for idx, region in enumerate(regions):
        if not region.active:
            for _ in range(_pause_columns(region.end - region.start, cfg)):
                columns.append({})
                kinds.append("pause")
                privates.append({})
            continue

        by_speaker = assigned.get(idx, {})
        if not by_speaker:
            continue
        width = max(len(items) for items in by_speaker.values())
        kind = "overlap" if len(region.active) > 1 else "solo"
        for offset in range(width):
            column: Dict[int, int] = {}
            private_flags: Dict[int, bool] = {}
            for speaker, items in by_speaker.items():
                if offset < len(items) and speaker in row_of:
                    item = items[offset]
                    column[row_of[speaker]] = item.token_id
                    private_flags[row_of[speaker]] = item.private
            if column:
                columns.append(column)
                kinds.append(kind)
                privates.append(private_flags)
    return columns, kinds, privates


def _chunk_ranges(kinds: List[str], limit: int) -> List[Tuple[int, int]]:
    """Chunk columns, preferring pause boundaries near the size limit."""
    if not kinds:
        return []
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < len(kinds):
        hard_end = min(len(kinds), start + max(limit, 1))
        end = hard_end
        if hard_end < len(kinds):
            preferred_start = start + max(1, int(0.75 * limit))
            pauses = [
                idx + 1
                for idx in range(preferred_start, hard_end)
                if kinds[idx] == "pause"
            ]
            if pauses:
                end = pauses[-1]
        ranges.append((start, end))
        start = end
    return ranges


def convert_timed_conversation(
    conv: Conversation,
    tokenize: Tokenize,
    cfg: ConvertConfig,
    source_id: int,
    rng: random.Random,
) -> List[Dict]:
    """Convert a timed conversation to overlap/pause-aware matrix chunks."""
    appear: List[int] = []
    for turn in conv.turns:
        if turn.speaker not in appear:
            appear.append(turn.speaker)
    if len(appear) > cfg.max_agents:
        appear = appear[: cfg.max_agents]
    if not appear:
        return []

    base_rows = list(range(len(appear)))
    if cfg.permute_agents:
        rng.shuffle(base_rows)
    row_of = {speaker: base_rows[i] for i, speaker in enumerate(appear)}
    target = _select_target(conv, appear, rng)
    target_row = row_of[target] if target is not None else None

    columns, kinds, privates = _render_timed_columns(conv, tokenize, row_of, cfg)
    if not columns:
        return []
    has_private = _has_private_turns(conv)
    lookback = max(0, cfg.context_lookback_columns)
    # The lookback prefix shares the same A*T budget, so the label-bearing
    # "new" portion of each chunk shrinks to leave room for it.
    max_time = max(1, cfg.max_flat_len // len(appear) - lookback)
    ranges = _chunk_ranges(kinds, max_time)
    examples: List[Dict] = []

    for chunk_idx, (start, end) in enumerate(ranges):
        lookback_len = min(lookback, start) if chunk_idx > 0 else 0
        ctx_start = start - lookback_len
        chunk = columns[ctx_start:end]
        chunk_privates = privates[ctx_start:end]
        A, T = len(appear), len(chunk)
        input_ids = [[cfg.placeholder_token_id] * T for _ in range(A)]
        input_activity_mask = [[0] * T for _ in range(A)]
        labels = [[cfg.ignore_index] * T for _ in range(A)]
        activity_labels = [[cfg.ignore_index] * T for _ in range(A)]
        agent_ids = [[a] * T for a in range(A)]
        is_private_mask = [[0] * T for _ in range(A)] if has_private else None

        for col, values in enumerate(chunk):
            for row, token_id in values.items():
                input_ids[row][col] = token_id
                input_activity_mask[row][col] = 1
                if is_private_mask is not None and chunk_privates[col].get(row):
                    is_private_mask[row][col] = 1
        # Lookback columns (indices [0, lookback_len)) are context only -- they
        # were already the label-bearing "new" portion of the PREVIOUS chunk,
        # so no activity/content labels are drawn from them here, including
        # the transition at their last column (that stays ignore_index too).
        for row in range(A):
            for col in range(lookback_len, T - 1):
                activity_labels[row][col] = input_activity_mask[row][col + 1]

        # For timed data, supervise the content emitted in the next column even
        # when an eligible row is currently inactive. This teaches first tokens
        # after pauses and floor acquisition, matching activity-gated generation.
        for content_row in _content_rows(conv, cfg, A, target_row):
            for col in range(lookback_len, T - 1):
                if input_activity_mask[content_row][col + 1]:
                    labels[content_row][col] = input_ids[content_row][col + 1]

        num_speaker_changes = _count_speaker_changes(
            input_activity_mask, start_col=lookback_len
        )
        example = {
            "input_ids": input_ids,
            "input_activity_mask": input_activity_mask,
            "labels": labels,
            "activity_labels": activity_labels,
            "agent_ids": agent_ids,
            "source_id": source_id,
            "length": T,
            "num_agents": A,
            "dataset_split": conv.meta.get("dataset_split", "train"),
            "conversation_id": conv.meta.get("conversation_id", ""),
            "group_id": conv.meta.get("group_id", ""),
            "chunk_index": chunk_idx,
            "num_overlap_columns": sum(len(values) >= 2 for values in chunk),
            "num_pause_columns": sum(not values for values in chunk),
            "context_lookback_columns": lookback_len,
            "content_supervision_mode": cfg.content_supervision_mode,
            "num_speaker_changes": num_speaker_changes,
            "is_chain_rich": num_speaker_changes >= 2,
        }
        if is_private_mask is not None:
            example["is_private_mask"] = is_private_mask
            example["agent_visibility"] = _agent_visibility_matrix(conv, row_of, A)
        examples.append(attach_quality_fields(example, start_col=lookback_len))
    return examples


def convert_conversation_examples(
    conv: Conversation,
    tokenize: Tokenize,
    cfg: ConvertConfig,
    source_id: int,
    rng: random.Random,
) -> Iterator[Dict]:
    """Yield one untimed example or multiple bounded timed examples."""
    if conv.meta.get("timed"):
        yield from convert_timed_conversation(conv, tokenize, cfg, source_id, rng)
        return
    example = convert_conversation(conv, tokenize, cfg, source_id, rng)
    if example is not None:
        example["dataset_split"] = conv.meta.get("dataset_split", "")
        example["conversation_id"] = conv.meta.get("conversation_id", "")
        example["group_id"] = conv.meta.get("group_id", "")
        example["chunk_index"] = 0
        example["num_overlap_columns"] = 0
        example["num_pause_columns"] = sum(
            not any(example["input_activity_mask"][a][t] for a in range(example["num_agents"]))
            for t in range(example["length"])
        )
        example["context_lookback_columns"] = 0
        example["is_chain_rich"] = example["num_speaker_changes"] >= 2
        yield attach_quality_fields(example)
