"""Bazinga! adapter (multi-party TV/movie dialogue, word-level timed).

Source: ``bazinga/bazinga`` on Hugging Face -- **gated**: requires an
approved access request on the dataset page plus an authenticated session
(``huggingface-cli login`` or an ``HF_TOKEN``/``HUGGING_FACE_HUB_TOKEN`` env
var) before any file can be downloaded. 16 TV/movie series (*24, Battlestar
Galactica, Breaking Bad, Buffy the Vampire Slayer, ER, Friends, Game of
Thrones, Homeland, Lost, Six Feet Under, The Big Bang Theory, The Office, The
Walking Dead, Harry Potter, Star Wars, The Lord of the Rings*), 400+ hours of
speech, 8M+ tokens, manually fan-transcribed and forced-aligned to word-level
timestamps with speaker/addressee/entity-linking annotations [26].

**License caution** (same category as MELD's TV-content caveat, §4.1 of
``docs/RESEARCH_REFERENCE.md``): the paper states manual transcripts were
scraped from fan sites "and are shared for research purposes only" --
research-use-only, not a permissive open license.

``content_bearing=True``: unlike Molweni/When2Speak, this is professionally
written, human-performed dialogue (fan-transcribed from real scripts/audio,
not LLM-generated or informal chat), so its text is a legitimate content
target -- see the priority-scoring writeup in ``docs/RESEARCH_REFERENCE.md``
§4.7.

**IMPORTANT (2026-08-25, confirmed against a real download once access was
granted): ``datasets.load_dataset("bazinga/bazinga", ...)`` does NOT work on
current ``datasets`` versions** (>=4.0, confirmed on 5.0.0) -- HF removed
support for script-based (``bazinga.py``) dataset loaders entirely
(``RuntimeError: Dataset scripts are no longer supported``). This is a
library-version incompatibility, not an access/auth problem: the gated
access check itself succeeds (the loading script and raw files download
fine), the *builder script execution* is what's blocked.

This adapter therefore bypasses ``datasets`` entirely and downloads/parses
the raw per-series files directly via ``huggingface_hub.hf_hub_download``,
reimplementing ``bazinga.py``'s own ``_generate_examples`` logic (fetched and
read directly from the HF repo to confirm field-for-field parity):

    data/{series}/episodes.txt     # "{identifier},{title},{imdb_url},"  one per line
    data/{series}/characters.txt   # "{character_id},{actor_id},{Character Name},{Actor Name},{imdb_url}"
    data/{series}/credits.txt      # "{identifier},{bool per character, comma-sep}" (numpy-loadtxt style)
    data/{series}/{identifier}.txt # one row per WORD, space-separated:
        "{identifier} {speaker} {start_time} {end_time} {word} {confidence} [{entity_linking} [{addressee} [{named_entity}]]]"
        -- trailing 3 fields are optional (older/partial annotations have fewer
        columns); "_" means the field is annotated-but-empty, "?" means
        annotation attempted but inconclusive, missing entirely means
        "NotAvailable". ``speaker == "all"`` denotes simultaneous/crowd speech
        (mapped to ``MULTIPLE_PERSONS`` below, same as the original loader).

Words are grouped into ``Turn``s by contiguous same-speaker runs; a missing
OR ``MULTIPLE_PERSONS`` speaker on a word ends the current turn without
starting a new one (crowd/overlap speech isn't attributable to one identity,
so it isn't turned into a fake single-speaker turn). An episode is skipped
entirely if its speaker annotation doesn't clear the same "GoldStandard" bar
``bazinga.py`` uses: at least one non-``MULTIPLE_PERSONS``/non-"unknown"*
speaker present, no literal ``"not_available"`` speaker, and >70% of the
episode's distinct speakers appearing in that episode's credited-character
list (a proxy for "this episode's speaker labels were actually vetted").

**Known limitation**: the transcript text itself is PTB/word-tokenized
(each word its own row, joined with a plain space), so
``_words_to_turns`` runs ``detokenize_ptb`` (spacing fixes only, not the
Molweni adapter's spellchecker -- this is professionally scripted dialogue,
already well-spelled) on each turn's joined text. Spaced hyphens
("twenty - six", "Port - au - Prince") are deliberately left alone -- PTB
tokenization can't be distinguished here from a genuine spaced em-dash
pause without more context, and this is content_bearing=True text, so a
wrong guess would be worse than leaving it as-is.

Train/validation/test split mirrors ``bazinga.py`` exactly: for TV series,
``Season01`` -> test, ``Season02``/``Season03`` -> validation, else -> train;
for movies, ``Episode01`` -> test, ``Episode02``/``Episode03`` -> validation,
else -> train. Exposed via ``meta["dataset_split"]`` (informational; this
project's own train/eval split is applied later in the pipeline, not here).
"""

from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional, Tuple

from ._text_cleaning import detokenize_ptb
from .spec import AdapterSpec
from ..schema import Conversation, ROLE_PEER, TimedWord, Turn

# The 16 series named in the paper (Lerner et al., LREC 2022, §3), split into
# the TV-series vs. movie identifier-format groups ``bazinga.py`` uses for
# train/val/test filtering (Season/Episode vs. bare Episode numbering).
SERIES = [
    "24",
    "BattlestarGalactica",
    "BreakingBad",
    "BuffyTheVampireSlayer",
    "ER",
    "Friends",
    "GameOfThrones",
    "Homeland",
    "Lost",
    "SixFeetUnder",
    "TheBigBangTheory",
    "TheOffice",
    "TheWalkingDead",
]
MOVIES = ["HarryPotter", "StarWars", "TheLordOfTheRings"]

MULTIPLE_PERSONS = "multiple_persons"
_PUNCT_ONLY = re.compile(r"^[^\w]+$")


def _download(repo_filename: str, raw_dir: str) -> Optional[str]:
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            "bazinga/bazinga", repo_filename, repo_type="dataset", cache_dir=raw_dir
        )
    except Exception as exc:  # gated/unauthenticated access, missing file, network, etc.
        print(f"[bazinga] failed to download {repo_filename!r}: {exc}")
        return None


def _parse_episodes_txt(path: str) -> List[str]:
    identifiers = []
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            identifiers.append(line.split(",")[0])
    return identifiers


def _parse_characters_txt(path: str) -> List[str]:
    characters = []
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            characters.append(line.split(",")[0])
    return characters


def _parse_credits_txt(path: str, characters: List[str]) -> Dict[str, List[str]]:
    credits: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            identifier, flags = parts[0], parts[1:]
            credits[identifier] = [
                characters[i] for i, flag in enumerate(flags) if flag == "1" and i < len(characters)
            ]
    return credits


def _resolve_optional_field(raw: str) -> Optional[str]:
    if raw in ("_", "?"):
        return None
    return raw


def _parse_transcript_line(line: str) -> Optional[dict]:
    parts = line.strip().split(" ")
    if len(parts) < 6:
        return None
    _identifier, speaker, start_time, end_time, word, confidence, *rest = parts
    if speaker == "all":
        speaker = MULTIPLE_PERSONS

    entity_linking = _resolve_optional_field(rest[0]) if len(rest) > 0 else None
    if entity_linking and "@" in entity_linking:
        entity_linking = MULTIPLE_PERSONS
    addressee = _resolve_optional_field(rest[1]) if len(rest) > 1 else None
    if addressee and "@" in addressee:
        addressee = MULTIPLE_PERSONS
    named_entity = _resolve_optional_field(rest[2]) if len(rest) > 2 else None

    try:
        start = float(start_time)
        end = float(end_time)
        confidence_f = float(confidence)
    except ValueError:
        return None

    return {
        "token": word,
        "speaker": speaker,
        "forced_alignment": {"start_time": start, "end_time": end, "confidence": confidence_f},
        "entity_linking": entity_linking,
        "named_entity": named_entity,
        "addressee": addressee,
    }


def _speaker_status_is_gold(words: List[dict], credited_characters: List[str]) -> bool:
    speakers = {
        w["speaker"]
        for w in words
        if w["speaker"] and w["speaker"] != MULTIPLE_PERSONS and "unknown" not in w["speaker"]
    }
    if not speakers or "not_available" in speakers:
        return False
    overlap = len(speakers & set(credited_characters)) / len(speakers)
    return overlap > 0.7


def _split_for_identifier(identifier: str, is_movie: bool) -> str:
    if is_movie:
        if ".Episode01" in identifier:
            return "test"
        if ".Episode02" in identifier or ".Episode03" in identifier:
            return "validation"
        return "train"
    if ".Season01.Episode" in identifier:
        return "test"
    if ".Season02.Episode" in identifier or ".Season03.Episode" in identifier:
        return "validation"
    return "train"


def _words_to_turns(words: List[dict]) -> Tuple[List[Turn], List[str]]:
    speakers: List[str] = []
    turns: List[Turn] = []
    current_speaker: Optional[str] = None
    current_words: List[TimedWord] = []

    def flush() -> None:
        if current_speaker is None or not current_words:
            return
        if current_speaker not in speakers:
            speakers.append(current_speaker)
        # The transcripts are PTB/word-tokenized (each word is its own row,
        # joined with a plain space), so contractions/punctuation come out
        # spaced ("did n't", "I ' ve") -- since content_bearing=True here
        # (unlike Molweni), fix that before it becomes a content-loss
        # target. Only the safe, unambiguous `detokenize_ptb` runs (NOT the
        # spellchecker in `clean_text`) -- this is professionally scripted
        # dialogue already, and spellcheck previously proved to mangle real
        # words (see Molweni's sys/cpu/aux/lol/ati regressions); it would be
        # even riskier against character/place names here.
        text = detokenize_ptb(" ".join(w.text for w in current_words if w.text)).strip()
        if not text:
            return
        turns.append(
            Turn(
                speaker=speakers.index(current_speaker),
                text=text,
                role=ROLE_PEER,
                start_time=current_words[0].start_time,
                end_time=current_words[-1].end_time,
                timed_words=list(current_words),
            )
        )

    for word in words:
        spk = word.get("speaker")
        token = str(word.get("token") or "").strip()
        # A missing OR MULTIPLE_PERSONS speaker ends the current turn without
        # starting a new one -- crowd/overlap speech isn't one identity.
        if not spk or spk == MULTIPLE_PERSONS or not token:
            flush()
            current_speaker, current_words = None, []
            continue
        if spk != current_speaker:
            flush()
            current_speaker, current_words = spk, []
        fa = word.get("forced_alignment") or {}
        try:
            start = float(fa["start_time"])
            end = float(fa["end_time"])
        except (KeyError, TypeError, ValueError):
            start = end = current_words[-1].end_time if current_words else 0.0
        current_words.append(
            TimedWord(
                text=token,
                start_time=start,
                end_time=max(start, end),
                is_punctuation=bool(_PUNCT_ONLY.match(token)),
            )
        )
    flush()
    return turns, speakers


def _iter_series(series_name: str, raw_dir: str) -> Iterator[Conversation]:
    is_movie = series_name in MOVIES
    episodes_path = _download(f"data/{series_name}/episodes.txt", raw_dir)
    if not episodes_path:
        return
    characters_path = _download(f"data/{series_name}/characters.txt", raw_dir)
    credits_path = _download(f"data/{series_name}/credits.txt", raw_dir)
    characters = _parse_characters_txt(characters_path) if characters_path else []
    credits = _parse_credits_txt(credits_path, characters) if credits_path else {}

    for identifier in _parse_episodes_txt(episodes_path):
        transcript_path = _download(f"data/{series_name}/{identifier}.txt", raw_dir)
        if not transcript_path:
            continue
        words = []
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                parsed = _parse_transcript_line(line)
                if parsed:
                    words.append(parsed)
        if not _speaker_status_is_gold(words, credits.get(identifier, [])):
            continue
        turns, speakers = _words_to_turns(words)
        if len(turns) < 2 or len(speakers) < 2:
            continue
        yield Conversation(
            source="bazinga",
            speakers=speakers,
            turns=turns,
            content_bearing=True,
            meta={
                "timed": True,
                "conversation_id": identifier,
                "series": series_name,
                "dataset_split": _split_for_identifier(identifier, is_movie),
            },
        )


def iter_conversations(raw_dir: str, series: Optional[List[str]] = None) -> Iterator[Conversation]:
    for series_name in (series or SERIES):
        yield from _iter_series(series_name, raw_dir)


SPEC = AdapterSpec(
    source="bazinga",
    source_id=7,
    content_bearing=True,
    default_weight=0.30,
    iter_conversations=iter_conversations,
    hf_repo="bazinga/bazinga",
    notes=(
        "Gated (requires an approved HF access request + auth token). 16 "
        "TV/movie series, word-timed, human fan-transcribed. Research-use-"
        "only per source acknowledgment -- see docs/RESEARCH_REFERENCE.md "
        "sec 4.7. Bypasses `datasets.load_dataset` (broken on datasets>=4.0 "
        "since HF removed script-based loaders) and downloads/parses the "
        "raw per-series .txt files directly via huggingface_hub."
    ),
)
