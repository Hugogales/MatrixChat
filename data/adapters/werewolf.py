"""Werewolf-Among-Us adapter (social-deduction game transcripts).

Source: ``bolinlai/Werewolf-Among-Us`` (HF). 191 usable One Night Ultimate
Werewolf games (151 from YouTube, 40 from Ego4D) with per-utterance timestamps
and game-level ``playerNames``/``startRoles``/``endRoles``. A handful of Avalon
recordings also ship in the repo but lack these fields entirely and are
skipped naturally by this adapter (no special-case filtering needed).

Known data limitation
----------------------
The raw transcripts cover only the *public* day-phase discussion. Night-phase
actions (who the Seer inspected, who the Robber/Troublemaker swapped) are
never recorded, so this adapter does NOT fabricate their results. The only
private knowledge synthesized here is grounded in what a player would
actually know at the start of the night, per One Night Ultimate Werewolf
rules:

- Every player knows their own dealt role (``startRoles``).
- Werewolves see each other (a "team" that mutually knows its members).
- Masons see each other (same team mechanic).
- The Minion learns who the Werewolves are, but this is NOT reciprocal --
  Werewolves never learn the Minion's identity (Minion wins with the
  Werewolf team but plays alone at night).
- All other roles (Villager, Seer, Robber, Troublemaker, Drunk, Insomniac,
  Hunter, Tanner, Doppelganger, ...) get no extra teammate knowledge here,
  since their night actions/results are not in the data.

Structure and identity
-----------------------
Real player names are a small, recurring set of ~59 usernames across the
whole corpus. To avoid the model learning spurious name-level correlations,
every game gets its own large, randomly sampled set of replacement names
(see ``data/adapters/_names.py``, shared with the MELD adapter's per-scene
substitution); the real names are used only to align ``Dialogue.speaker``
with ``playerNames``/``startRoles``/``endRoles`` and never reach model text.

Each game gets a short, SEQUENTIAL prelude (never simultaneous, so it needs no
special-casing in the turn-taking reward): for every player, in roster order,
one PUBLIC self-introduction turn (states the player's own randomized name --
the identifier other agents can key off later) followed by one PRIVATE
role-reveal turn (states the player's own role, and -- only for Werewolf/Mason
teams, or the Minion's one-directional knowledge of the Werewolves -- the
relevant teammates by name AND role). The public ``Dialogue`` continues after
the prelude using each utterance's real timestamp, reusing the AMI hybrid
timed converter for overlap/pause-aware packing.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from .spec import AdapterSpec
from ._names import build_name_substitution_regex as _build_name_substitution_regex
from ._names import sample_random_names as _sample_random_names
from ..schema import Conversation, ROLE_PEER, TimedWord, Turn

# Roles that mutually know their teammates at night in One Night Ultimate
# Werewolf. The Minion is handled separately below (one-directional).
_MUTUAL_TEAM_ROLES = {"Werewolf", "Mason"}

_TIMESTAMP_RE = re.compile(r"^(\d+):(\d{1,2})$")

# Tiny, strictly increasing offsets so the prelude renders as a sequential run
# of solo turns immediately before the public dialogue begins.
_PRELUDE_STEP_SECONDS = 0.5


def _parse_timestamp(value: str) -> Optional[float]:
    match = _TIMESTAMP_RE.match(value.strip()) if isinstance(value, str) else None
    if not match:
        return None
    minutes, seconds = int(match.group(1)), int(match.group(2))
    return float(minutes * 60 + seconds)


def _iter_split_files(raw_dir: str) -> Iterator[Tuple[str, str, str]]:
    """Yield ``(collection, split, path)`` for every official split file."""
    base = os.path.join(raw_dir, "werewolf")
    for collection in ("Youtube", "Ego4D"):
        for split in ("train", "val", "test"):
            path = os.path.join(base, collection, "split", f"{split}.json")
            if os.path.isfile(path):
                yield collection, split, path


def _load_games(raw_dir: str) -> Iterator[Tuple[dict, str]]:
    for collection, split, path in _iter_split_files(raw_dir):
        with open(path, "r", encoding="utf-8") as f:
            try:
                games = json.load(f)
            except json.JSONDecodeError:
                continue
        for game in games:
            if isinstance(game, dict):
                yield game, split


def _game_id(game: dict) -> str:
    """A stable, collision-free identifier for one game recording.

    ``YT_ID``/``EG_ID`` alone is NOT sufficient: it is not actually a unique
    per-video key in this corpus (the same YT_ID+Game_ID pair can legitimately
    refer to two entirely different games from different source videos, e.g.
    "part8"/"Game2" appears once for a 2016 video and again for an unrelated
    2017 video). Including ``video_name`` (present for YouTube games; empty
    for Ego4D, whose ``EG_ID`` IS a real unique recording id) resolves every
    such collision -- verified against the full corpus.
    """
    outer = game.get("YT_ID") or game.get("EG_ID") or "game"
    video_name = game.get("video_name", "")
    return f"{outer}|{video_name}|{game.get('Game_ID', '?')}"


def _team_of(role: str) -> Optional[str]:
    return role if role in _MUTUAL_TEAM_ROLES else None


def _teammates(idx: int, roles: List[str]) -> List[int]:
    """Indices of players sharing a mutually-aware team with player ``idx``."""
    team = _team_of(roles[idx])
    if team is None:
        return []
    return [j for j, r in enumerate(roles) if j != idx and r == team]


@dataclass
class _PlayerContext:
    index: int
    real_name: str
    random_name: str
    role: str


def _build_prelude_turns(players: List[_PlayerContext]) -> Tuple[List[Turn], float]:
    """Sequential (public intro, private role) pair per player, in roster order."""
    roles = [p.role for p in players]
    turns: List[Turn] = []
    t = 0.0
    for player in players:
        intro_start, intro_end = t, t + _PRELUDE_STEP_SECONDS
        turns.append(
            Turn(
                speaker=player.index,
                text=f"Hi, I'm {player.random_name}.",
                role=ROLE_PEER,
                start_time=intro_start,
                end_time=intro_end,
                timed_words=[TimedWord(text=f"Hi, I'm {player.random_name}.", start_time=intro_start, end_time=intro_end)],
                visible_to=None,  # public: gives everyone a name identifier
            )
        )
        t = intro_end

        reveal = f"You are {player.random_name}. Your role is {player.role}."
        visible_to: List[int] = []
        if player.role == "Minion":
            werewolves = [j for j, r in enumerate(roles) if r == "Werewolf"]
            if werewolves:
                names = ", ".join(players[j].random_name for j in werewolves)
                verb, label = ("is", "Werewolf") if len(werewolves) == 1 else ("are", "Werewolves")
                reveal += f" {names} {verb} the {label}."
            # One-directional: the Minion knows, but is not added to the
            # werewolves' own visible_to, so they never learn the Minion.
        else:
            for j in _teammates(player.index, roles):
                reveal += f" {players[j].random_name} is also {players[j].role}."
                visible_to.append(j)

        reveal_start, reveal_end = t, t + _PRELUDE_STEP_SECONDS
        turns.append(
            Turn(
                speaker=player.index,
                text=reveal,
                role=ROLE_PEER,
                start_time=reveal_start,
                end_time=reveal_end,
                timed_words=[TimedWord(text=reveal, start_time=reveal_start, end_time=reveal_end)],
                visible_to=visible_to,
                meta={"kind": "role_reveal"},
            )
        )
        t = reveal_end
    return turns, t


def _to_conversation(game: dict, split: str) -> Optional[Conversation]:
    real_names: List[str] = list(game.get("playerNames") or [])
    roles: List[str] = list(game.get("startRoles") or [])
    dialogue = game.get("Dialogue") or []
    if len(real_names) < 2 or len(roles) != len(real_names) or not dialogue:
        return None

    game_id = _game_id(game)
    random_names = _sample_random_names(game_id, len(real_names))
    if len(random_names) < len(real_names):
        return None  # extremely defensive; the pool is far larger than any game

    players = [
        _PlayerContext(index=i, real_name=real_names[i], random_name=random_names[i], role=roles[i])
        for i in range(len(real_names))
    ]
    real_to_random = {p.real_name: p.random_name for p in players}
    name_to_index = {p.real_name: p.index for p in players}
    substitute = _build_name_substitution_regex(real_to_random)

    turns, prelude_end = _build_prelude_turns(players)

    dialogue_start = None
    for entry in dialogue:
        ts = _parse_timestamp(str(entry.get("timestamp", "")))
        if ts is not None:
            dialogue_start = ts
            break
    offset = prelude_end - (dialogue_start or 0.0)

    for entry in dialogue:
        speaker_name = str(entry.get("speaker", "")).strip()
        idx = name_to_index.get(speaker_name)
        if idx is None:
            continue
        text = str(entry.get("utterance", "")).strip()
        if not text:
            continue
        if substitute is not None:
            text = substitute(text)
        ts = _parse_timestamp(str(entry.get("timestamp", "")))
        start = (ts + offset) if ts is not None else prelude_end
        end = start + 0.4  # no per-word timing available; treat as one terminal
        turns.append(
            Turn(
                speaker=idx,
                text=text,
                role=ROLE_PEER,
                start_time=start,
                end_time=end,
                timed_words=[TimedWord(text=text, start_time=start, end_time=end)],
                visible_to=None,
            )
        )

    if len(turns) < 2:
        return None

    return Conversation(
        source="werewolf",
        speakers=[p.random_name for p in players],
        turns=turns,
        content_bearing=True,
        meta={
            "timed": True,
            "conversation_id": game_id,
            "group_id": game_id,
            "dataset_split": "validation" if split == "val" else split,
        },
    )


def iter_conversations(raw_dir: str) -> Iterator[Conversation]:
    for game, split in _load_games(raw_dir):
        conv = _to_conversation(game, split)
        if conv is not None:
            yield conv


SPEC = AdapterSpec(
    source="werewolf",
    source_id=4,
    content_bearing=True,
    default_weight=0.10,
    iter_conversations=iter_conversations,
    hf_repo="bolinlai/Werewolf-Among-Us",
    hf_snapshot=True,  # file-based repo: snapshot into raw_dir/werewolf, parse files
    # The repo's full mp4/zip video+audio assets total ~9.9GB and are never
    # read by this adapter; restrict the snapshot to the transcript/role JSON
    # and human-readable txt/md files.
    hf_allow_patterns=["*.json", "*.txt", "*.md"],
    notes=(
        "Social-deduction games (191 usable, 151 YouTube + 40 Ego4D). Public day "
        "dialogue is content-bearing and timed; each player gets a sequential "
        "public self-intro + private role-reveal prelude (own role, and "
        "teammates for Werewolf/Mason, or one-directional Werewolf knowledge for "
        "the Minion) with per-game randomized names -- see module docstring."
    ),
)
