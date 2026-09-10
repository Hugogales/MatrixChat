"""Tests for the Bazinga! adapter (rewritten 2026-08-25 to bypass
``datasets.load_dataset``, which is broken for this source on
``datasets>=4.0`` -- see the module docstring). These exercise the raw-file
parsing and word-to-turn grouping helpers directly, without network access.
"""

from data.adapters.bazinga import (
    MOVIES,
    MULTIPLE_PERSONS,
    SERIES,
    SPEC,
    _parse_characters_txt,
    _parse_credits_txt,
    _parse_episodes_txt,
    _parse_transcript_line,
    _speaker_status_is_gold,
    _split_for_identifier,
    _words_to_turns,
)


def _word(token, speaker, start, end):
    return {
        "token": token,
        "speaker": speaker,
        "forced_alignment": {"start_time": start, "end_time": end, "confidence": 0.99},
    }


def test_contiguous_same_speaker_words_become_one_turn():
    words = [
        _word("Hello", "sheldon_cooper", 0.0, 0.3),
        _word("there", "sheldon_cooper", 0.3, 0.6),
        _word("Hi", "leonard_hofstadter", 0.7, 0.9),
    ]
    turns, speakers = _words_to_turns(words)
    assert speakers == ["sheldon_cooper", "leonard_hofstadter"]
    assert len(turns) == 2
    assert turns[0].speaker == 0
    assert turns[0].text == "Hello there"
    assert turns[0].start_time == 0.0
    assert turns[0].end_time == 0.6
    assert turns[1].speaker == 1
    assert turns[1].text == "Hi"


def test_missing_speaker_word_ends_turn_without_starting_a_new_one():
    words = [
        _word("Hello", "sheldon_cooper", 0.0, 0.3),
        {"token": "[noise]", "speaker": None, "forced_alignment": {"start_time": 0.3, "end_time": 0.4}},
        _word("Hi", "sheldon_cooper", 0.5, 0.7),
    ]
    turns, speakers = _words_to_turns(words)
    # The interruption by a speakerless word splits what would otherwise be
    # one continuous turn into two, both still attributed to the same speaker.
    assert speakers == ["sheldon_cooper"]
    assert len(turns) == 2
    assert [t.text for t in turns] == ["Hello", "Hi"]


def test_multiple_persons_speaker_ends_turn_without_becoming_a_fake_identity():
    words = [
        _word("Hello", "sheldon_cooper", 0.0, 0.3),
        _word("Surprise!", MULTIPLE_PERSONS, 0.3, 0.5),
        _word("Hi", "sheldon_cooper", 0.6, 0.8),
    ]
    turns, speakers = _words_to_turns(words)
    assert speakers == ["sheldon_cooper"]
    assert MULTIPLE_PERSONS not in speakers
    assert len(turns) == 2


def test_turn_text_is_detokenized_from_ptb_spacing():
    words = [
        _word("I", "sheldon_cooper", 0.0, 0.1),
        _word("'", "sheldon_cooper", 0.1, 0.15),
        _word("ve", "sheldon_cooper", 0.15, 0.2),
        _word("been", "sheldon_cooper", 0.2, 0.4),
        _word("thinking", "sheldon_cooper", 0.4, 0.6),
        _word(".", "sheldon_cooper", 0.6, 0.6),
    ]
    turns, _ = _words_to_turns(words)
    assert turns[0].text == "I've been thinking."


def test_missing_forced_alignment_does_not_crash():
    words = [
        {"token": "Hello", "speaker": "sheldon_cooper"},
        {"token": "there", "speaker": "sheldon_cooper", "forced_alignment": {}},
    ]
    turns, _ = _words_to_turns(words)
    assert len(turns) == 1
    assert turns[0].text == "Hello there"


def test_parse_transcript_line_full_row():
    line = "TheBigBangTheory.Season01.Episode01 sheldon_cooper 1.49 1.61 So 0.99 _ leonard_hofstadter _\n"
    parsed = _parse_transcript_line(line)
    assert parsed["token"] == "So"
    assert parsed["speaker"] == "sheldon_cooper"
    assert parsed["forced_alignment"] == {"start_time": 1.49, "end_time": 1.61, "confidence": 0.99}
    assert parsed["entity_linking"] is None
    assert parsed["addressee"] == "leonard_hofstadter"
    assert parsed["named_entity"] is None


def test_parse_transcript_line_all_speaker_maps_to_multiple_persons():
    parsed = _parse_transcript_line("id all 0.0 0.1 Surprise 0.5\n")
    assert parsed["speaker"] == MULTIPLE_PERSONS


def test_parse_transcript_line_question_mark_addressee_is_unresolved():
    parsed = _parse_transcript_line("id sheldon_cooper 0.0 0.1 hey 0.5 _ ? _\n")
    assert parsed["addressee"] is None


def test_parse_transcript_line_short_row_returns_none():
    assert _parse_transcript_line("too short\n") is None


def test_parse_episodes_txt(tmp_path):
    path = tmp_path / "episodes.txt"
    path.write_text(
        "TheBigBangTheory.Season01.Episode01,Pilot,https://imdb.com/x,\n"
        "TheBigBangTheory.Season01.Episode02,The Big Bran Hypothesis,https://imdb.com/y,\n"
    )
    identifiers = _parse_episodes_txt(str(path))
    assert identifiers == [
        "TheBigBangTheory.Season01.Episode01",
        "TheBigBangTheory.Season01.Episode02",
    ]


def test_parse_characters_txt(tmp_path):
    path = tmp_path / "characters.txt"
    path.write_text(
        "leonard_hofstadter,johnny_galecki,Leonard Hofstadter,Johnny Galecki,https://imdb.com/a\n"
        "sheldon_cooper,jim_parsons,Sheldon Cooper,Jim Parsons,https://imdb.com/b\n"
    )
    assert _parse_characters_txt(str(path)) == ["leonard_hofstadter", "sheldon_cooper"]


def test_parse_credits_txt(tmp_path):
    path = tmp_path / "credits.txt"
    path.write_text("ep01,1,0,1\nep02,0,1,0\n")
    characters = ["leonard_hofstadter", "sheldon_cooper", "penny"]
    credits = _parse_credits_txt(str(path), characters)
    assert credits["ep01"] == ["leonard_hofstadter", "penny"]
    assert credits["ep02"] == ["sheldon_cooper"]


def test_speaker_status_gold_when_credited_overlap_is_high():
    words = [_word("hi", "sheldon_cooper", 0, 1), _word("hey", "leonard_hofstadter", 1, 2)]
    assert _speaker_status_is_gold(words, ["sheldon_cooper", "leonard_hofstadter", "penny"]) is True


def test_speaker_status_not_gold_when_no_credited_overlap():
    words = [_word("hi", "background_extra_unknown3", 0, 1)]
    assert _speaker_status_is_gold(words, ["sheldon_cooper"]) is False


def test_speaker_status_not_gold_when_speakers_empty():
    assert _speaker_status_is_gold([], ["sheldon_cooper"]) is False


def test_split_for_identifier_tv_series():
    assert _split_for_identifier("X.Season01.Episode01", is_movie=False) == "test"
    assert _split_for_identifier("X.Season02.Episode01", is_movie=False) == "validation"
    assert _split_for_identifier("X.Season03.Episode01", is_movie=False) == "validation"
    assert _split_for_identifier("X.Season04.Episode01", is_movie=False) == "train"


def test_split_for_identifier_movies():
    assert _split_for_identifier("X.Episode01", is_movie=True) == "test"
    assert _split_for_identifier("X.Episode02", is_movie=True) == "validation"
    assert _split_for_identifier("X.Episode04", is_movie=True) == "train"


def test_spec_is_content_bearing_and_gated():
    assert SPEC.content_bearing is True
    assert SPEC.hf_repo == "bazinga/bazinga"
    assert SPEC.source_id == 7
    assert len(SERIES) == 13
    assert len(MOVIES) == 3
    assert len(SERIES) + len(MOVIES) == 16
