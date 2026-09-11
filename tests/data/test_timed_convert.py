"""Hybrid AMI conversion: dense solo speech, overlap, and meaningful pauses."""

import random

from data.convert import ConvertConfig, convert_conversation, convert_timed_conversation
from data.schema import Conversation, TimedWord, Turn


def _tokenizer():
    vocab = {}

    def tokenize(text):
        key = text.strip()
        if not key:
            return []
        if key not in vocab:
            vocab[key] = len(vocab) + 1
        return [vocab[key]]

    return tokenize


def _cfg(**kwargs):
    values = dict(
        max_agents=8,
        placeholder_token_id=999,
        permute_agents=False,
        pause_threshold_seconds=1.0,
        pause_quantum_seconds=1.0,
        max_pause_columns=8,
        max_flat_len=1536,
    )
    values.update(kwargs)
    return ConvertConfig(**values)


def _word(text, start, end):
    return TimedWord(text=text, start_time=start, end_time=end)


def test_solo_speech_is_dense_despite_word_timing():
    conv = Conversation(
        source="ami",
        speakers=["A"],
        turns=[Turn(
            speaker=0,
            text="one two",
            start_time=0.0,
            end_time=1.0,
            timed_words=[_word("one", 0.0, 0.2), _word("two", 0.9, 1.0)],
        )],
        meta={"timed": True},
    )
    examples = convert_timed_conversation(conv, _tokenizer(), _cfg(), 5, random.Random(0))
    assert len(examples) == 1
    ex = examples[0]
    assert ex["length"] == 2
    assert ex["input_activity_mask"][0] == [1, 1]
    assert ex["num_pause_columns"] == 0


def test_long_same_speaker_pause_inserts_bounded_inactive_columns():
    conv = Conversation(
        source="ami",
        speakers=["A"],
        turns=[Turn(
            speaker=0,
            text="one two",
            start_time=0.0,
            end_time=2.7,
            timed_words=[_word("one", 0.0, 0.2), _word("two", 2.7, 2.9)],
        )],
        meta={"timed": True},
    )
    examples = convert_timed_conversation(conv, _tokenizer(), _cfg(), 5, random.Random(0))
    ex = examples[0]
    # 2.5-second all-silent gap: ceil((2.5 - 1.0) / 1.0) = 2.
    assert ex["input_activity_mask"][0] == [1, 0, 0, 1]
    assert ex["num_pause_columns"] == 2
    # The last inactive cell predicts both that A resumes and A's resumed token.
    assert ex["activity_labels"][0][2] == 1
    assert ex["labels"][0][2] == ex["input_ids"][0][3]


def test_overlap_places_speakers_in_same_column_without_solo_gaps():
    conv = Conversation(
        source="ami",
        speakers=["A", "B"],
        turns=[
            Turn(
                speaker=0,
                text="one two",
                start_time=0.0,
                end_time=2.0,
                timed_words=[_word("one", 0.0, 1.0), _word("two", 1.0, 2.0)],
            ),
            Turn(
                speaker=1,
                text="wait",
                start_time=0.5,
                end_time=1.5,
                timed_words=[_word("wait", 0.5, 1.5)],
            ),
        ],
        meta={"timed": True},
    )
    ex = convert_timed_conversation(conv, _tokenizer(), _cfg(), 5, random.Random(0))[0]
    counts = [
        sum(ex["input_activity_mask"][row][col] for row in range(2))
        for col in range(ex["length"])
    ]
    assert 2 in counts
    assert ex["num_overlap_columns"] == 1
    assert ex["num_pause_columns"] == 0
    assert all(sum(row) == 2 for row in ex["input_activity_mask"][:1])


def test_all_speakers_supervises_timed_floor_acquisition_tokens():
    conv = Conversation(
        source="ami",
        speakers=["A", "B", "C"],
        turns=[
            Turn(speaker=0, text="a1 a2", timed_words=[
                _word("a1", 0.0, 0.4), _word("a2", 0.4, 0.8),
            ]),
            Turn(speaker=1, text="b1 b2", timed_words=[
                _word("b1", 1.0, 1.4), _word("b2", 1.4, 1.8),
            ]),
            Turn(speaker=2, text="c1 c2", timed_words=[
                _word("c1", 2.0, 2.4), _word("c2", 2.4, 2.8),
            ]),
        ],
        meta={"timed": True},
    )
    ex = convert_timed_conversation(
        conv,
        _tokenizer(),
        _cfg(content_supervision_mode="all_speakers"),
        5,
        random.Random(0),
    )[0]

    # Each later speaker's first token is supervised from the preceding column,
    # when that row is still inactive (floor acquisition).
    for row in (1, 2):
        first_active = ex["input_activity_mask"][row].index(1)
        assert ex["input_activity_mask"][row][first_active - 1] == 0
        assert ex["labels"][row][first_active - 1] == ex["input_ids"][row][first_active]
        assert ex["labels"][row][first_active] == ex["input_ids"][row][first_active + 1]

    assert ex["num_speaker_changes"] == 2
    assert ex["is_chain_rich"] is True
    assert ex["content_supervision_mode"] == "all_speakers"


def test_timed_conversion_chunks_to_flat_length_limit():
    words = [_word(str(i), float(i), float(i) + 0.1) for i in range(20)]
    conv = Conversation(
        source="ami",
        speakers=["A", "B"],
        turns=[
            Turn(speaker=0, text=" ".join(str(i) for i in range(20)), timed_words=words),
            Turn(speaker=1, text="ok", timed_words=[_word("ok", 30.0, 30.2)]),
        ],
        meta={"timed": True, "conversation_id": "x"},
    )
    examples = convert_timed_conversation(
        conv, _tokenizer(), _cfg(max_flat_len=12), 5, random.Random(0)
    )
    assert len(examples) > 1
    assert all(ex["num_agents"] * ex["length"] <= 12 for ex in examples)
    assert sum(
        sum(sum(row) for row in ex["input_activity_mask"])
        for ex in examples
    ) == 21  # no token is dropped or duplicated at chunk boundaries


def test_lookback_zero_is_unchanged_from_original_chunking():
    words = [_word(str(i), float(i), float(i) + 0.1) for i in range(20)]
    conv = Conversation(
        source="ami",
        speakers=["A", "B"],
        turns=[
            Turn(speaker=0, text=" ".join(str(i) for i in range(20)), timed_words=words),
            Turn(speaker=1, text="ok", timed_words=[_word("ok", 30.0, 30.2)]),
        ],
        meta={"timed": True, "conversation_id": "x"},
    )
    baseline = convert_timed_conversation(conv, _tokenizer(), _cfg(max_flat_len=12), 5, random.Random(0))
    explicit_zero = convert_timed_conversation(
        conv, _tokenizer(), _cfg(max_flat_len=12, context_lookback_columns=0), 5, random.Random(0)
    )
    assert baseline == explicit_zero
    assert all(ex["context_lookback_columns"] == 0 for ex in baseline)


def test_lookback_prefixes_non_first_chunks_with_real_prior_columns():
    words = [_word(str(i), float(i), float(i) + 0.1) for i in range(20)]
    conv = Conversation(
        source="ami",
        speakers=["A", "B"],
        turns=[
            Turn(speaker=0, text=" ".join(str(i) for i in range(20)), timed_words=words),
            Turn(speaker=1, text="ok", timed_words=[_word("ok", 30.0, 30.2)]),
        ],
        meta={"timed": True, "conversation_id": "x"},
    )
    with_lookback = convert_timed_conversation(
        conv,
        _tokenizer(),
        _cfg(
            max_flat_len=12,
            context_lookback_columns=3,
            content_supervision_mode="all_speakers",
        ),
        5,
        random.Random(0),
    )
    assert len(with_lookback) > 1
    # First chunk has nothing to look back into (every chunk's "new" budget is
    # shrunk uniformly by the lookback reservation, chunk 0 included, so its
    # boundaries differ from the lookback=0 case -- it just never uses the
    # reserved room since there's nothing preceding it).
    assert with_lookback[0]["context_lookback_columns"] == 0

    for i in range(1, len(with_lookback)):
        chunk = with_lookback[i]
        lb = chunk["context_lookback_columns"]
        assert lb == 3  # plenty of preceding columns exist in this fixture
        # The lookback prefix's real content matches the tail of everything
        # emitted (across all agents) so far -- genuine history, not filler.
        prior_active_ids = []
        for prev in with_lookback[:i]:
            prev_lb = prev["context_lookback_columns"]
            for col in range(prev_lb, prev["length"]):
                for row in range(prev["num_agents"]):
                    if prev["input_activity_mask"][row][col]:
                        prior_active_ids.append(prev["input_ids"][row][col])
        this_prefix_ids = []
        for col in range(lb):
            for row in range(chunk["num_agents"]):
                if chunk["input_activity_mask"][row][col]:
                    this_prefix_ids.append(chunk["input_ids"][row][col])
        # (guard len==0: list[-0:] is the WHOLE list in Python, not empty)
        expected_tail = prior_active_ids[-len(this_prefix_ids):] if this_prefix_ids else []
        assert this_prefix_ids == expected_tail
        # Lookback columns carry no supervision at all.
        for col in range(lb):
            for row in range(chunk["num_agents"]):
                assert chunk["labels"][row][col] == -100
                assert chunk["activity_labels"][row][col] == -100

    # No token is dropped or duplicated in the label-bearing ("new") portion.
    total_new_active = sum(
        sum(
            sum(row[ex["context_lookback_columns"]:]) for row in ex["input_activity_mask"]
        )
        for ex in with_lookback
    )
    assert total_new_active == 21


def test_lookback_shrinks_when_not_enough_preceding_columns_exist():
    conv = Conversation(
        source="ami",
        speakers=["A", "B"],
        turns=[
            Turn(speaker=0, text="ab", timed_words=[_word("a", 0.0, 0.1), _word("b", 0.1, 0.2)]),
            Turn(speaker=1, text="ok", timed_words=[_word("ok", 5.0, 5.2)]),
        ],
        meta={"timed": True},
    )
    # A large lookback request against a conversation with very few total
    # columns should be clamped, never reach into negative indices.
    examples = convert_timed_conversation(
        conv, _tokenizer(), _cfg(max_flat_len=2, context_lookback_columns=50), 5, random.Random(0),
    )
    for i, ex in enumerate(examples):
        if i == 0:
            assert ex["context_lookback_columns"] == 0
        else:
            assert 0 <= ex["context_lookback_columns"] <= 50


def test_zero_duration_event_attaches_to_nearest_speech_not_first_region():
    conv = Conversation(
        source="ami",
        speakers=["A"],
        turns=[Turn(
            speaker=0,
            text="first [laughter] last",
            timed_words=[
                _word("first", 0.0, 0.2),
                _word("[laughter]", 5.0, 5.0),
                _word("last", 5.5, 6.0),
            ],
        )],
        meta={"timed": True},
    )
    ex = convert_timed_conversation(conv, _tokenizer(), _cfg(), 5, random.Random(0))[0]
    active_indices = [
        index for index, active in enumerate(ex["input_activity_mask"][0]) if active
    ]
    assert active_indices[0] == 0
    # Both the isolated event and following word occur after the preserved pause.
    assert active_indices[1] > 1
    assert active_indices[2] == active_indices[1] + 1


# ---------------------------------------------------------------------------
# Private turns (Turn.visible_to) -- both the timed and turn-major paths.
# ---------------------------------------------------------------------------


def test_timed_private_turn_produces_visibility_tensors_and_diagonal():
    conv = Conversation(
        source="werewolf",
        speakers=["A", "B", "C"],
        turns=[
            Turn(speaker=0, text="public intro", timed_words=[_word("public intro", 0.0, 0.5)]),
            Turn(
                speaker=0, text="secret", visible_to=[1],
                timed_words=[_word("secret", 0.5, 1.0)],
            ),
            Turn(speaker=1, text="hi", timed_words=[_word("hi", 1.0, 1.5)]),
            Turn(speaker=2, text="hey", timed_words=[_word("hey", 1.5, 2.0)]),
        ],
        meta={"timed": True},
    )
    ex = convert_timed_conversation(conv, _tokenizer(), _cfg(permute_agents=False), 4, random.Random(0))[0]
    assert "is_private_mask" in ex
    assert "agent_visibility" in ex
    vis = ex["agent_visibility"]
    for a in range(3):
        assert vis[a][a] == 1
    assert vis[0][1] == 1   # agent 1 permitted to view agent 0's private cell
    assert vis[0][2] == 0   # agent 2 NOT permitted
    # Exactly one private cell exists (the "secret" turn), on row 0.
    assert sum(sum(row) for row in ex["is_private_mask"]) == 1
    assert any(ex["is_private_mask"][0])


def test_turn_major_private_turn_produces_visibility_tensors():
    conv = Conversation(
        source="werewolf",
        speakers=["A", "B"],
        turns=[
            Turn(speaker=0, text="hello"),
            Turn(speaker=0, text="secret role", visible_to=[]),  # private, self-only
            Turn(speaker=1, text="hi"),
        ],
    )
    ex = convert_conversation(conv, _tokenizer(), _cfg(permute_agents=False, min_turn_gap=0, max_turn_gap=0), 4, random.Random(0))
    assert "is_private_mask" in ex
    vis = ex["agent_visibility"]
    assert vis[0][0] == 1
    assert vis[0][1] == 0  # no extra visibility granted
    assert sum(sum(row) for row in ex["is_private_mask"]) == 1  # "secret role" -> 1 token (this tokenizer)


def test_public_only_source_has_no_visibility_fields():
    conv = Conversation(
        source="meld",
        speakers=["A", "B"],
        turns=[Turn(speaker=0, text="hi"), Turn(speaker=1, text="hey")],
    )
    ex = convert_conversation(conv, _tokenizer(), _cfg(permute_agents=False), 1, random.Random(0))
    assert "is_private_mask" not in ex
    assert "agent_visibility" not in ex
