"""Offline tests for the official AMI NXT XML adapter."""

from pathlib import Path

from data.adapters import SOURCE_IDS
from data.adapters.ami import iter_conversations


def _write_fixture(root: Path):
    (root / "corpusResources").mkdir(parents=True)
    (root / "words").mkdir()
    (root / "dialogueActs").mkdir()
    (root / "corpusResources" / "meetings.xml").write_text(
        """<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <meeting observation="ES2002a" type="scenario" visibility="seen" seen_type="training">
    <speaker nxt_agent="A" global_name="person-a" role="PM" channel="0"/>
    <speaker nxt_agent="B" global_name="person-b" role="UI" channel="1"/>
  </meeting>
</nite:root>"""
    )
    (root / "words" / "ES2002a.A.words.xml").write_text(
        """<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <vocalsound nite:id="x0" starttime="0.0" endtime="0.0" type="other"/>
  <w nite:id="a0" starttime="0.0" endtime="0.4">Hello</w>
  <w nite:id="a1" starttime="0.4" endtime="0.4" punc="true">,</w>
  <w nite:id="a2" starttime="0.4" endtime="1.2">there</w>
</nite:root>"""
    )
    (root / "dialogueActs" / "ES2002a.A.dialog-act.xml").write_text(
        """<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <dact nite:id="da-a">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_4)"/>
    <nite:child href="ES2002a.A.words.xml#id(x0)..id(a2)"/>
  </dact>
</nite:root>"""
    )
    (root / "words" / "ES2002a.B.words.xml").write_text(
        """<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <w nite:id="b0" starttime="0.8" endtime="1.1">Hi</w>
  <w nite:id="b1" starttime="1.1" endtime="1.1" punc="true">.</w>
</nite:root>"""
    )
    (root / "dialogueActs" / "ES2002a.B.dialog-act.xml").write_text(
        """<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <dact nite:id="da-b">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_2)"/>
    <nite:child href="ES2002a.B.words.xml#id(b0)..id(b1)"/>
  </dact>
</nite:root>"""
    )


def test_ami_adapter_resolves_ranges_timing_and_metadata(tmp_path):
    root = tmp_path / "ami"
    _write_fixture(root)

    conversations = list(iter_conversations(str(tmp_path)))
    assert len(conversations) == 1
    conv = conversations[0]
    assert conv.source == "ami"
    assert conv.content_bearing is True
    assert conv.speakers == ["A", "B"]
    assert conv.meta["conversation_id"] == "ES2002a"
    assert conv.meta["group_id"] == "ES2002"
    assert conv.meta["dataset_split"] == "train"
    assert conv.meta["timed"] is True

    assert [turn.text for turn in conv.turns] == ["Hello, there", "Hi."]
    assert conv.turns[0].start_time == 0.0
    assert conv.turns[0].end_time == 1.2
    assert conv.turns[0].meta["dialogue_act_type"] == "ami_da_4"
    assert conv.turns[1].speaker == 1
    assert conv.meta["speaker_meta"]["A"]["global_name"] == "person-a"


def test_ami_source_id_is_append_only_and_legacy_ids_do_not_shift():
    assert SOURCE_IDS["conversation_chronicles"] == 0
    assert SOURCE_IDS["meld"] == 1
    assert SOURCE_IDS["ami"] == 5


def _write_vocalsound_fixture(root: Path):
    """Same shape as _write_fixture, but agent A's words include a laugh AND
    a cough vocalsound bracketing "Hello, there"."""
    (root / "corpusResources").mkdir(parents=True)
    (root / "words").mkdir()
    (root / "dialogueActs").mkdir()
    (root / "corpusResources" / "meetings.xml").write_text(
        """<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <meeting observation="ES2002a" type="scenario" visibility="seen" seen_type="training">
    <speaker nxt_agent="A" global_name="person-a" role="PM" channel="0"/>
    <speaker nxt_agent="B" global_name="person-b" role="UI" channel="1"/>
  </meeting>
</nite:root>"""
    )
    (root / "words" / "ES2002a.A.words.xml").write_text(
        """<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <vocalsound nite:id="x0" starttime="0.0" endtime="0.0" type="laugh"/>
  <w nite:id="a0" starttime="0.1" endtime="0.4">Hello</w>
  <w nite:id="a1" starttime="0.4" endtime="0.4" punc="true">,</w>
  <w nite:id="a2" starttime="0.4" endtime="1.2">there</w>
  <vocalsound nite:id="x1" starttime="1.2" endtime="1.2" type="cough"/>
</nite:root>"""
    )
    (root / "dialogueActs" / "ES2002a.A.dialog-act.xml").write_text(
        """<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <dact nite:id="da-a">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_4)"/>
    <nite:child href="ES2002a.A.words.xml#id(x0)..id(x1)"/>
  </dact>
</nite:root>"""
    )
    (root / "words" / "ES2002a.B.words.xml").write_text(
        """<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <w nite:id="b0" starttime="1.3" endtime="1.6">Hi</w>
  <w nite:id="b1" starttime="1.6" endtime="1.6" punc="true">.</w>
</nite:root>"""
    )
    (root / "dialogueActs" / "ES2002a.B.dialog-act.xml").write_text(
        """<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <dact nite:id="da-b">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_2)"/>
    <nite:child href="ES2002a.B.words.xml#id(b0)..id(b1)"/>
  </dact>
</nite:root>"""
    )


def test_ami_adapter_drops_laughter_but_keeps_cough(tmp_path):
    """Regression test: "[laughter]" is intentionally dropped (it became a
    cheap, content-free way for agents to "speak" -- degenerate
    "[laughter] [laughter]" spam in generation) while other interpretable
    vocal events like cough are still kept."""
    root = tmp_path / "ami"
    _write_vocalsound_fixture(root)

    conversations = list(iter_conversations(str(tmp_path)))
    assert len(conversations) == 1
    text = conversations[0].turns[0].text
    assert "[laughter]" not in text
    assert "[cough]" in text
    assert text == "Hello, there [cough]"
