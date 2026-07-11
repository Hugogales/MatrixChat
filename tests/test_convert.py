"""Tests for the Stage-2 matrix converter (data/convert.py).

Uses a trivial whitespace tokenizer (no model download) so the conversion logic
-- turn-major layout, no-EOS turn ends, shifted target-seat content labels
(last token -> silence), silence subsampling, agent permutation, source tagging
-- is verified offline. Tests pin the inter-turn gap to 0 for determinism.
"""

import random

from data.convert import ConvertConfig, convert_conversation
from data.schema import Conversation, Turn, ROLE_USER, ROLE_ASSISTANT, ROLE_PEER

SILENCE = 900
EOS = 999  # unused by the converter now; kept to exercise the config field


def make_tokenizer():
    vocab = {}

    def tok(text):
        ids = []
        for w in text.split():
            if w not in vocab:
                vocab[w] = len(vocab) + 1  # ids 1..N, below SILENCE/EOS
            ids.append(vocab[w])
        return ids

    return tok


def make_conversation(content_bearing=True):
    return Conversation(
        source="meld",
        speakers=["U", "A", "B"],
        turns=[
            Turn(speaker=0, text="hello world", role=ROLE_USER),
            Turn(speaker=1, text="hi there friend", role=ROLE_ASSISTANT),
            Turn(speaker=2, text="ok cool", role=ROLE_PEER),
        ],
        content_bearing=content_bearing,
    )


def cfg(**kw):
    base = dict(
        silence_token_id=SILENCE,
        eos_token_id=EOS,
        max_agents=8,
        min_turn_gap=0,   # deterministic layout (no gaps) for assertions
        max_turn_gap=0,
        permute_agents=False,
    )
    base.update(kw)
    return ConvertConfig(**base)


def test_layout_shapes_and_agent_ids():
    tok = make_tokenizer()
    ex = convert_conversation(make_conversation(), tok, cfg(), source_id=3, rng=random.Random(0))
    A = len(ex["input_ids"])
    T = ex["length"]
    assert A == 3
    # No EOS, no gaps: T = 2 + 3 + 2 = 7.
    assert T == 7
    assert all(len(row) == T for row in ex["input_ids"])
    assert ex["agent_ids"] == [[a] * T for a in range(A)]
    assert ex["source_id"] == 3


def test_no_eos_tokens_in_stream():
    tok = make_tokenizer()
    ex = convert_conversation(make_conversation(), tok, cfg(), source_id=0, rng=random.Random(0))
    # The EOS id must never appear in the flattened input (turns end via silence).
    assert all(EOS not in row for row in ex["input_ids"])


def test_target_seat_content_labels_shifted_to_silence_at_end():
    tok = make_tokenizer()
    ex = convert_conversation(make_conversation(), tok, cfg(), source_id=0, rng=random.Random(0))
    ii, lb = ex["input_ids"], ex["labels"]
    hi, there, friend = tok("hi")[0], tok("there")[0], tok("friend")[0]

    # Assistant is the target seat -> row 1 (no permutation), turn at cols 2..4.
    assert ii[1][2:5] == [hi, there, friend]
    # Shifted next-token labels over the row; the last token predicts SILENCE.
    assert lb[1][2] == there
    assert lb[1][3] == friend
    assert lb[1][4] == SILENCE


def test_all_silence_cells_supervised_content_context_ignored():
    tok = make_tokenizer()
    ex = convert_conversation(make_conversation(), tok, cfg(), source_id=0, rng=random.Random(0))
    ii, lb = ex["input_ids"], ex["labels"]
    A, T = len(ii), ex["length"]
    hi = tok("hi")[0]  # the target turn's first token (used for the floor-taking label)
    for a in range(A):
        for t in range(T):
            if ii[a][t] == SILENCE:
                # Every silent input cell is supervised: silence, or (for the cell
                # right before the target turn) the turn's first token (floor-taking).
                assert lb[a][t] != -100
                assert lb[a][t] in (SILENCE, hi)
            elif a == 1:
                # Target row content cells are content labels (not ignore).
                assert lb[a][t] != -100
            else:
                # Non-target speakers' own words are context (ignored).
                assert lb[a][t] == -100


def test_floor_taking_label_before_target_turn():
    """The silent cell right before the target's turn predicts the turn's first token."""
    tok = make_tokenizer()
    ex = convert_conversation(make_conversation(), tok, cfg(), source_id=0, rng=random.Random(0))
    ii, lb = ex["input_ids"], ex["labels"]
    hi = tok("hi")[0]
    # Target = assistant (row 1), turn at cols 2..4; col 1 is the pre-turn silence cell.
    assert ii[1][1] == SILENCE
    assert lb[1][1] == hi  # taught to START speaking from silence


def test_structure_only_has_no_content_labels():
    tok = make_tokenizer()
    ex = convert_conversation(
        make_conversation(content_bearing=False), tok, cfg(),
        source_id=1, rng=random.Random(0),
    )
    ii, lb = ex["input_ids"], ex["labels"]
    # No content labels; every silent cell is silence, speakers' words are ignore.
    for a in range(len(ii)):
        for t in range(ex["length"]):
            if ii[a][t] == SILENCE:
                assert lb[a][t] == SILENCE
            else:
                assert lb[a][t] == -100
    # All silent cells supervised: total cells (3x7=21) minus 7 content tokens = 14.
    n_silence = sum(v == SILENCE for row in lb for v in row)
    assert n_silence == 14


def test_random_gap_changes_length():
    tok = make_tokenizer()
    # With a positive gap range, total length exceeds the no-gap length (7).
    lengths = set()
    for seed in range(8):
        ex = convert_conversation(
            make_conversation(), tok, cfg(min_turn_gap=1, max_turn_gap=3),
            source_id=0, rng=random.Random(seed),
        )
        lengths.add(ex["length"])
    assert min(lengths) >= 7 + 3  # >=1 gap after each of 3 turns
    assert len(lengths) > 1       # gaps are actually randomized


def test_permutation_preserves_content_labeling():
    tok = make_tokenizer()
    ex = convert_conversation(
        make_conversation(), tok, cfg(permute_agents=True), source_id=0, rng=random.Random(7),
    )
    ii, lb = ex["input_ids"], ex["labels"]
    hi, there, friend = tok("hi")[0], tok("there")[0], tok("friend")[0]

    target_row, start = None, None
    for a, row in enumerate(ii):
        if hi in row:
            target_row, start = a, row.index(hi)
            break
    assert target_row is not None
    assert ii[target_row][start:start + 3] == [hi, there, friend]
    assert lb[target_row][start] == there
    assert lb[target_row][start + 1] == friend
    assert lb[target_row][start + 2] == SILENCE

    assert all(any(v != SILENCE for v in row) for row in ii)
