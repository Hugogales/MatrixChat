"""Tests for the Stage-2 matrix converter (data/convert.py).

Uses a trivial whitespace tokenizer (no model download) so the conversion logic
-- turn-major layout, no-EOS turn ends, activity mask/labels for EVERY agent,
shifted target-seat content labels gated by continued activity, agent
permutation, source tagging -- is verified offline. Tests pin the inter-turn
gap to 0 for determinism.
"""

import random

import pytest

from data.convert import ConvertConfig, convert_conversation
from data.build_dataset import parse_args
from data.schema import Conversation, Turn, ROLE_USER, ROLE_ASSISTANT, ROLE_PEER

PLACEHOLDER = 900


def make_tokenizer():
    vocab = {}

    def tok(text):
        ids = []
        for w in text.split():
            if w not in vocab:
                vocab[w] = len(vocab) + 1  # ids 1..N, below PLACEHOLDER
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
        placeholder_token_id=PLACEHOLDER,
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
    assert all(len(row) == T for row in ex["input_activity_mask"])
    assert all(len(row) == T for row in ex["activity_labels"])
    assert ex["agent_ids"] == [[a] * T for a in range(A)]
    assert ex["source_id"] == 3
    assert ex["num_agents"] == A


def test_no_placeholder_masquerading_as_content():
    """Active cells never hold the placeholder id (it's only used when inactive)."""
    tok = make_tokenizer()
    ex = convert_conversation(make_conversation(), tok, cfg(), source_id=0, rng=random.Random(0))
    ii, mask = ex["input_ids"], ex["input_activity_mask"]
    for a in range(len(ii)):
        for t in range(ex["length"]):
            if mask[a][t]:
                assert ii[a][t] != PLACEHOLDER


def test_activity_mask_matches_turn_layout():
    tok = make_tokenizer()
    ex = convert_conversation(make_conversation(), tok, cfg(), source_id=0, rng=random.Random(0))
    mask = ex["input_activity_mask"]
    # No permutation: row0=U (cols 0-1), row1=A (cols 2-4), row2=B (cols 5-6).
    assert mask[0] == [1, 1, 0, 0, 0, 0, 0]
    assert mask[1] == [0, 0, 1, 1, 1, 0, 0]
    assert mask[2] == [0, 0, 0, 0, 0, 1, 1]


def test_activity_labels_cover_every_agent_and_transition():
    """activity_labels[a][c] = whether agent a is active at c+1; last col ignored."""
    tok = make_tokenizer()
    ex = convert_conversation(make_conversation(), tok, cfg(), source_id=0, rng=random.Random(0))
    mask, al = ex["input_activity_mask"], ex["activity_labels"]
    A, T = len(mask), ex["length"]
    for a in range(A):
        for c in range(T - 1):
            assert al[a][c] == mask[a][c + 1]
        assert al[a][T - 1] == -100  # no successor column


def test_target_seat_content_labels_only_while_continuing():
    tok = make_tokenizer()
    ex = convert_conversation(make_conversation(), tok, cfg(), source_id=0, rng=random.Random(0))
    ii, lb = ex["input_ids"], ex["labels"]
    hi, there, friend = tok("hi")[0], tok("there")[0], tok("friend")[0]

    # Assistant is the target seat -> row 1 (no permutation), turn at cols 2..4.
    assert ii[1][2:5] == [hi, there, friend]
    # Shifted next-token labels WITHIN the turn only (never on the last token --
    # stopping is an activity decision now, not a content one).
    assert lb[1][2] == there
    assert lb[1][3] == friend
    assert lb[1][4] == -100

    # No content labels anywhere on non-target rows.
    for a in (0, 2):
        assert all(v == -100 for v in lb[a])


def test_all_speakers_content_labels_every_continuing_row():
    tok = make_tokenizer()
    ex = convert_conversation(
        make_conversation(),
        tok,
        cfg(content_supervision_mode="all_speakers"),
        source_id=0,
        rng=random.Random(0),
    )

    for row in range(ex["num_agents"]):
        active = [
            col
            for col, value in enumerate(ex["input_activity_mask"][row])
            if value
        ]
        assert len(active) >= 2
        for col in active[:-1]:
            assert ex["labels"][row][col] == ex["input_ids"][row][col + 1]
        assert ex["labels"][row][active[-1]] == -100
    assert ex["content_supervision_mode"] == "all_speakers"


def test_default_mode_and_cli_preserve_target_only_behavior():
    assert ConvertConfig().content_supervision_mode == "target_only"
    args = parse_args(["--mode", "convert", "--raw_dir", "/tmp/raw"])
    assert args.content_supervision_mode == "target_only"
    args = parse_args([
        "--mode", "convert",
        "--raw_dir", "/tmp/raw",
        "--content_supervision_mode", "all_speakers",
    ])
    assert args.content_supervision_mode == "all_speakers"
    with pytest.raises(ValueError, match="content_supervision_mode"):
        ConvertConfig(content_supervision_mode="invalid")


def test_structure_only_has_no_content_labels_but_has_activity_labels():
    tok = make_tokenizer()
    ex = convert_conversation(
        make_conversation(content_bearing=False), tok, cfg(),
        source_id=1, rng=random.Random(0),
    )
    lb, al = ex["labels"], ex["activity_labels"]
    # No content labels at all.
    assert all(v == -100 for row in lb for v in row)
    # But activity labels are still present for every agent/transition (known
    # ground truth "who speaks when" regardless of content_bearing).
    n_defined = sum(v != -100 for row in al for v in row)
    A, T = len(al), ex["length"]
    assert n_defined == A * (T - 1)


def test_all_speakers_does_not_enable_structure_only_content_labels():
    ex = convert_conversation(
        make_conversation(content_bearing=False),
        make_tokenizer(),
        cfg(content_supervision_mode="all_speakers"),
        source_id=1,
        rng=random.Random(0),
    )
    assert all(value == -100 for row in ex["labels"] for value in row)


def test_speaker_change_metadata_marks_chain_rich_examples():
    ex = convert_conversation(
        make_conversation(), make_tokenizer(), cfg(), source_id=0, rng=random.Random(0)
    )
    assert ex["num_speaker_changes"] == 2


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
    ii, lb, mask = ex["input_ids"], ex["labels"], ex["input_activity_mask"]
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
    assert lb[target_row][start + 2] == -100

    # All three speakers still occupy a row (permutation kept everyone active somewhere).
    assert all(any(v == 1 for v in row) for row in mask)
