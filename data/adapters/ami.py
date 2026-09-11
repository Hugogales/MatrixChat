"""AMI Meeting Corpus adapter with word-level timing and dialogue-act turns.

The official manual AMI annotations use stand-off NXT XML: dialogue acts point
to ranges in per-speaker word files.  This adapter resolves those ranges while
retaining every terminal's forced-alignment timestamps so the converter can
preserve overlap and meaningful all-speaker pauses.
"""

from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from .spec import AdapterSpec
from ..schema import Conversation, ROLE_PEER, TimedWord, Turn

NITE = "http://nite.sourceforge.net/"
NITE_ID = f"{{{NITE}}}id"
_ID_RE = re.compile(r"id\(([^)]+)\)")


@dataclass
class _MeetingInfo:
    observation: str
    speakers: List[str]
    speaker_meta: Dict[str, dict]
    split: str
    group_id: str
    meeting_type: str


def _corpus_root(raw_dir: str) -> str:
    """Find the extracted AMI annotation root (wrapped or unwrapped)."""
    candidates = [
        os.path.join(raw_dir, "ami"),
        os.path.join(raw_dir, "ami", "ami_public_manual_1.6.2"),
        os.path.join(raw_dir, "ami_public_manual_1.6.2"),
    ]
    for root in candidates:
        if os.path.isdir(os.path.join(root, "words")) and os.path.isdir(
            os.path.join(root, "dialogueActs")
        ):
            return root
    raise FileNotFoundError(
        f"Could not find extracted AMI words/ and dialogueActs/ under {raw_dir!r}"
    )


def _scenario_group(observation: str) -> str:
    """Group scenario a/b/c/d observations to prevent split leakage."""
    if re.match(r"^(ES|IS|TS)\d{4}[a-d]$", observation):
        return observation[:-1]
    return observation


def _meeting_split(attrs: dict) -> str:
    if attrs.get("visibility") == "unseen":
        return "test"
    seen_type = attrs.get("seen_type", "")
    if seen_type == "development":
        return "validation"
    return "train"


def _load_meetings(root: str) -> Dict[str, _MeetingInfo]:
    path = os.path.join(root, "corpusResources", "meetings.xml")
    tree = ET.parse(path)
    meetings: Dict[str, _MeetingInfo] = {}
    for node in tree.getroot():
        if node.tag.rsplit("}", 1)[-1] != "meeting":
            continue
        observation = node.attrib.get("observation")
        if not observation:
            continue
        speaker_meta: Dict[str, dict] = {}
        speakers: List[str] = []
        for sp in node:
            if sp.tag.rsplit("}", 1)[-1] != "speaker":
                continue
            local = sp.attrib.get("nxt_agent")
            if not local:
                continue
            speakers.append(local)
            speaker_meta[local] = {
                "global_name": sp.attrib.get("global_name"),
                "role": sp.attrib.get("role"),
                "channel": sp.attrib.get("channel"),
            }
        speakers.sort()
        meetings[observation] = _MeetingInfo(
            observation=observation,
            speakers=speakers,
            speaker_meta=speaker_meta,
            split=_meeting_split(node.attrib),
            group_id=_scenario_group(observation),
            meeting_type=node.attrib.get("type", "unknown"),
        )
    return meetings


def _event_text(element: ET.Element) -> str:
    tag = element.tag.rsplit("}", 1)[-1]
    if tag == "w":
        return (element.text or "").strip()
    if tag == "vocalsound":
        kind = element.attrib.get("type", "vocal_noise")
        # Keep interpretable conversational events; generic "other" noises are
        # annotation artifacts and should not become vocabulary targets.
        # "[laughter]" was dropped (not just filtered like "other"): in
        # practice the model latched onto it as a cheap, content-free way for
        # multiple agents to become "active" at once, producing degenerate
        # "[laughter] [laughter] [laughter]" spam in generation instead of
        # real dialogue (see e.g. logs/demo_handoff_variation_result.json).
        return {
            "cough": "[cough]",
            "breath": "[breath]",
        }.get(kind, "")
    # Nonwords and transform errors contain no useful lexical target.
    return ""


def _load_words(path: str) -> tuple[List[str], Dict[str, TimedWord]]:
    ids: List[str] = []
    words: Dict[str, TimedWord] = {}
    for element in ET.parse(path).getroot():
        word_id = element.attrib.get(NITE_ID)
        if not word_id:
            continue
        # Keep every terminal ID in document order even when its text is
        # intentionally discarded. Dialogue-act ranges may begin/end on a
        # generic sound but still contain useful words after/before it.
        ids.append(word_id)
        text = _event_text(element)
        if not text:
            continue
        try:
            start = float(element.attrib["starttime"])
            end = float(element.attrib["endtime"])
        except (KeyError, ValueError):
            continue
        words[word_id] = TimedWord(
            text=text,
            start_time=start,
            end_time=max(start, end),
            is_punctuation=element.attrib.get("punc") == "true",
            meta={"xml_tag": element.tag.rsplit("}", 1)[-1]},
        )
    return ids, words


def _expand_href(href: str, ordered_ids: List[str]) -> List[str]:
    refs = _ID_RE.findall(href)
    if not refs:
        return []
    if len(refs) == 1:
        return refs if refs[0] in ordered_ids else []
    index = {word_id: i for i, word_id in enumerate(ordered_ids)}
    if refs[0] not in index or refs[-1] not in index:
        return []
    lo, hi = index[refs[0]], index[refs[-1]]
    if hi < lo:
        lo, hi = hi, lo
    return ordered_ids[lo : hi + 1]


def _join_words(words: List[TimedWord]) -> str:
    text = ""
    for word in words:
        if word.is_punctuation:
            text = text.rstrip() + word.text
        else:
            text += (" " if text else "") + word.text
    return text.strip()


def _dialogue_act_type(dact: ET.Element) -> Optional[str]:
    for child in dact:
        if child.tag == f"{{{NITE}}}pointer" and child.attrib.get("role") == "da-aspect":
            refs = _ID_RE.findall(child.attrib.get("href", ""))
            return refs[0] if refs else None
    return None


def _load_speaker_turns(root: str, meeting: str, speaker_idx: int, speaker: str) -> List[Turn]:
    word_path = os.path.join(root, "words", f"{meeting}.{speaker}.words.xml")
    da_path = os.path.join(root, "dialogueActs", f"{meeting}.{speaker}.dialog-act.xml")
    if not os.path.isfile(word_path) or not os.path.isfile(da_path):
        return []

    ordered_ids, word_by_id = _load_words(word_path)
    turns: List[Turn] = []
    for ordinal, dact in enumerate(ET.parse(da_path).getroot()):
        if dact.tag.rsplit("}", 1)[-1] != "dact":
            continue
        span_ids: List[str] = []
        for child in dact:
            if child.tag == f"{{{NITE}}}child":
                span_ids.extend(_expand_href(child.attrib.get("href", ""), ordered_ids))
        timed_words = [word_by_id[word_id] for word_id in span_ids if word_id in word_by_id]
        if not timed_words:
            continue
        text = _join_words(timed_words)
        if not text or all(word.is_punctuation for word in timed_words):
            continue
        turns.append(
            Turn(
                speaker=speaker_idx,
                text=text,
                role=ROLE_PEER,
                start_time=min(word.start_time for word in timed_words),
                end_time=max(word.end_time for word in timed_words),
                timed_words=timed_words,
                meta={
                    "dialogue_act_id": dact.attrib.get(NITE_ID),
                    "dialogue_act_type": _dialogue_act_type(dact),
                    "source_ordinal": ordinal,
                },
            )
        )
    return turns


def iter_conversations(raw_dir: str) -> Iterator[Conversation]:
    root = _corpus_root(raw_dir)
    meetings = _load_meetings(root)
    available = {
        os.path.basename(path).split(".", 1)[0]
        for path in glob.glob(os.path.join(root, "dialogueActs", "*.dialog-act.xml"))
    }
    for observation in sorted(available):
        info = meetings.get(observation)
        if info is None or len(info.speakers) < 2:
            continue
        turns: List[Turn] = []
        for idx, speaker in enumerate(info.speakers):
            turns.extend(_load_speaker_turns(root, observation, idx, speaker))
        turns.sort(
            key=lambda turn: (
                float(turn.start_time or 0.0),
                float(turn.end_time or 0.0),
                info.speakers[turn.speaker],
                int(turn.meta.get("source_ordinal", 0)),
            )
        )
        if len(turns) < 2:
            continue
        yield Conversation(
            source="ami",
            speakers=info.speakers,
            turns=turns,
            content_bearing=True,
            meta={
                "conversation_id": observation,
                "group_id": info.group_id,
                "dataset_split": info.split,
                "meeting_type": info.meeting_type,
                "speaker_meta": info.speaker_meta,
                "timed": True,
            },
        )


SPEC = AdapterSpec(
    source="ami",
    source_id=5,
    content_bearing=True,
    default_weight=0.60,
    iter_conversations=iter_conversations,
    archive_url=(
        "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
        "ami_public_manual_1.6.2.zip"
    ),
    notes="AMI meetings: timed dialogue acts, overlap/pauses, CC BY 4.0.",
)
