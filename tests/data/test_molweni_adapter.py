"""Tests for the Molweni adapter, focused on the 2026-08-24 grammar/spelling
cleaning pass (data/adapters/_text_cleaning.py) applied by default."""

import json

from data.adapters.molweni import SPEC, iter_conversations


def _write_raw(tmp_path, dialogues):
    base = tmp_path / "molweni"
    base.mkdir()
    (base / "train.json").write_text(json.dumps(dialogues), encoding="utf-8")
    return str(tmp_path)


def test_cleans_text_by_default(tmp_path):
    dialogues = [{
        "edus": [
            {"speaker": "alice", "text": "i dont think thats right"},
            {"speaker": "bob", "text": "sudo apt-get install nvidia-driver"},
        ]
    }]
    raw_dir = _write_raw(tmp_path, dialogues)
    convs = list(iter_conversations(raw_dir))
    assert len(convs) == 1
    texts = [t.text for t in convs[0].turns]
    assert texts[0] == "i don't think that's right"
    assert texts[1] == "sudo apt-get install nvidia-driver"  # jargon untouched


def test_clean_false_preserves_raw_text(tmp_path):
    dialogues = [{
        "edus": [
            {"speaker": "alice", "text": "i dont think thats right"},
            {"speaker": "bob", "text": "yeah ok"},
        ]
    }]
    raw_dir = _write_raw(tmp_path, dialogues)
    convs = list(iter_conversations(raw_dir, clean=False))
    assert convs[0].turns[0].text == "i dont think thats right"


def test_conversation_is_not_content_bearing():
    """Cleaning Molweni's text cannot affect the content loss -- it is never
    a supervision target, only structural (turn-taking) input; this is the
    key caveat behind the cleaning pass (see molweni.py's module docstring)."""
    assert SPEC.content_bearing is False
