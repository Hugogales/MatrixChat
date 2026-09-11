"""Offline tests for the Werewolf-Among-Us adapter (real split-file schema)."""

import json

from data.adapters import SOURCE_IDS
from data.adapters.werewolf import (
    _build_prelude_turns,
    _PlayerContext,
    _sample_random_names,
    iter_conversations,
)


def _write_split(root, collection, split, games):
    path = root / "werewolf" / collection / "split"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{split}.json").write_text(json.dumps(games))


def _game(
    game_id="Game1",
    outer="yt1",
    player_names=("Real1", "Real2", "Real3"),
    roles=("Werewolf", "Werewolf", "Villager"),
    dialogue=None,
):
    dialogue = dialogue or [
        {"speaker": player_names[0], "timestamp": "00:03", "utterance": "Hello everyone."},
        {"speaker": player_names[1], "timestamp": "00:05", "utterance": f"I agree with {player_names[0]}."},
        {"speaker": player_names[2], "timestamp": "00:07", "utterance": "Hmm."},
    ]
    return {
        "YT_ID": outer,
        "video_name": "test",
        "Game_ID": game_id,
        "Dialogue": dialogue,
        "startTime": "00:00:00",
        "endTime": "00:10:00",
        "playerNames": list(player_names),
        "votingOutcome": [0] * len(player_names),
        "startRoles": list(roles),
        "endRoles": list(roles),
        "warning": "N/A",
    }


def test_source_id_is_pinned_to_legacy_value():
    assert SOURCE_IDS["werewolf"] == 4


def test_adapter_parses_real_split_schema(tmp_path):
    _write_split(tmp_path, "Youtube", "train", [_game()])
    convs = list(iter_conversations(str(tmp_path)))
    assert len(convs) == 1
    conv = convs[0]
    assert conv.source == "werewolf"
    assert conv.content_bearing is True
    assert conv.meta["timed"] is True
    assert conv.meta["dataset_split"] == "train"
    assert conv.meta["group_id"] == conv.meta["conversation_id"]


def test_val_split_normalized_to_validation(tmp_path):
    _write_split(tmp_path, "Youtube", "val", [_game()])
    conv = next(iter_conversations(str(tmp_path)))
    assert conv.meta["dataset_split"] == "validation"


def test_public_self_intro_states_own_randomized_name(tmp_path):
    _write_split(tmp_path, "Youtube", "train", [_game()])
    conv = next(iter_conversations(str(tmp_path)))
    intro_turns = [t for t in conv.turns if t.visible_to is None and "Hi, I'm" in t.text]
    assert len(intro_turns) == 3
    for turn in intro_turns:
        assert conv.speakers[turn.speaker] in turn.text
        # The real usernames must never leak into model-visible text.
        for real in ("Real1", "Real2", "Real3"):
            assert real not in turn.text


def test_private_role_reveal_states_own_role_and_is_not_public(tmp_path):
    _write_split(tmp_path, "Youtube", "train", [_game()])
    conv = next(iter_conversations(str(tmp_path)))
    role_turns = [t for t in conv.turns if t.meta.get("kind") == "role_reveal"]
    assert len(role_turns) == 3
    for turn in role_turns:
        assert turn.visible_to is not None  # private
        assert "Your role is" in turn.text
        assert conv.speakers[turn.speaker] in turn.text


def test_werewolf_teammates_mutually_named_by_name_and_role(tmp_path):
    _write_split(tmp_path, "Youtube", "train", [_game(roles=("Werewolf", "Werewolf", "Villager"))])
    conv = next(iter_conversations(str(tmp_path)))
    role_turns = {t.speaker: t for t in conv.turns if t.meta.get("kind") == "role_reveal"}

    ww0, ww1 = role_turns[0], role_turns[1]
    assert conv.speakers[1] in ww0.text and "Werewolf" in ww0.text
    assert conv.speakers[0] in ww1.text and "Werewolf" in ww1.text
    assert ww0.visible_to == [1]
    assert ww1.visible_to == [0]

    villager = role_turns[2]
    assert villager.visible_to == []  # no teammates, private to self only


def test_minion_learns_werewolves_one_directionally(tmp_path):
    _write_split(
        tmp_path, "Youtube", "train",
        [_game(player_names=("Real1", "Real2", "Real3"), roles=("Werewolf", "Minion", "Villager"))],
    )
    conv = next(iter_conversations(str(tmp_path)))
    role_turns = {t.speaker: t for t in conv.turns if t.meta.get("kind") == "role_reveal"}

    minion_turn = role_turns[1]
    werewolf_turn = role_turns[0]
    assert conv.speakers[0] in minion_turn.text  # minion's text names the werewolf
    assert minion_turn.visible_to == []  # but this knowledge stays private to the minion
    # The werewolf's own reveal must NOT be shared with (or mention) the minion.
    assert werewolf_turn.visible_to == []
    assert conv.speakers[1] not in werewolf_turn.text


def test_sequential_prelude_has_no_overlap_between_players():
    players = [
        _PlayerContext(index=0, real_name="Real1", random_name="Alpha", role="Villager"),
        _PlayerContext(index=1, real_name="Real2", random_name="Beta", role="Villager"),
    ]
    turns, end_time = _build_prelude_turns(players)
    intervals = [(t.start_time, t.end_time, t.speaker) for t in turns]
    intervals.sort()
    for (s1, e1, _), (s2, e2, _) in zip(intervals, intervals[1:]):
        assert e1 <= s2  # never overlapping -> no special-casing needed downstream
    assert end_time == intervals[-1][1]


def test_dialogue_mentions_of_real_names_are_substituted_consistently(tmp_path):
    _write_split(tmp_path, "Youtube", "train", [_game()])
    conv = next(iter_conversations(str(tmp_path)))
    dialogue_turns = [t for t in conv.turns if t.meta.get("kind") != "role_reveal" and "Hi, I'm" not in t.text]
    joined = " ".join(t.text for t in dialogue_turns)
    assert "Real1" not in joined and "Real2" not in joined and "Real3" not in joined
    # "I agree with Real1" should have been rewritten to the randomized name.
    assert conv.speakers[0] in joined


def test_random_names_are_unique_within_a_game_and_never_the_pool_default():
    names_a = _sample_random_names("game-A", 6)
    assert len(names_a) == len(set(names_a)) == 6


def test_random_names_differ_across_games():
    names_a = _sample_random_names("game-A", 5)
    names_b = _sample_random_names("game-B", 5)
    assert names_a != names_b


def test_random_names_reproducible_for_same_game_id():
    assert _sample_random_names("game-X", 4) == _sample_random_names("game-X", 4)


def test_avalon_only_records_without_role_fields_are_skipped(tmp_path):
    _write_split(
        tmp_path, "Ego4D", "train",
        [{"EG_ID": "e1", "Game_ID": "Game1", "Dialogue": [{"speaker": "A", "timestamp": "00:01", "utterance": "hi"}]}],
    )
    assert list(iter_conversations(str(tmp_path))) == []


def test_mismatched_role_and_player_counts_are_skipped(tmp_path):
    game = _game()
    game["startRoles"] = ["Werewolf"]  # length mismatch vs playerNames
    _write_split(tmp_path, "Youtube", "train", [game])
    assert list(iter_conversations(str(tmp_path))) == []


def test_group_id_disambiguates_colliding_yt_id_and_game_id(tmp_path):
    """YT_ID+Game_ID alone is NOT a unique key in the real corpus -- the same
    pair can legitimately label two entirely different source videos (see
    ``_game_id`` docstring). ``video_name`` must disambiguate them so a
    train/test split never accidentally shares a group_id."""
    game_a = _game(game_id="Game2", outer="part8")
    game_a["video_name"] = "Video From 2016"
    game_b = _game(game_id="Game2", outer="part8")
    game_b["video_name"] = "Unrelated Video From 2017"

    _write_split(tmp_path, "Youtube", "train", [game_a])
    _write_split(tmp_path, "Youtube", "test", [game_b])

    convs = list(iter_conversations(str(tmp_path)))
    assert len(convs) == 2
    group_ids = {c.meta["group_id"] for c in convs}
    assert len(group_ids) == 2  # must NOT collide
