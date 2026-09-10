"""CPU-only tests for XML-tagged turn round-robin and pass-the-baton."""

from types import SimpleNamespace

import torch

from evaluation.paired_continuation import stable_id
from model.generation_xml_turns import (
    _extract_content,
    generate_xml_turn_protocol,
    matrix_prefix_to_xml,
    parse_agent_choice,
)


class CharTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(int(value)) for value in ids if int(value) >= 32)


class SuffixCausalLM:
    """Emit the next character of a turn based on the decoded suffix.

    Forwards during tag injection are ignored; only the last logits are sampled.
    """

    def __init__(self, turns: dict[str, str], routing_reply: str = ""):
        self.turns = dict(turns)
        self.routing_reply = routing_reply
        self.tokenizer = CharTokenizer()
        self.vocab_size = 256
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def _predict(self, text: str) -> int:
        if self.routing_reply and "</routing>" in text:
            after = text.rsplit("</routing>", 1)[-1].lstrip("\n")
            reply = self.routing_reply
            if reply.startswith(after) and after != reply:
                return ord(reply[len(after)])
        last = -1
        chosen = None
        for tag, turn in self.turns.items():
            index = text.rfind(tag)
            if index >= last:
                last = index
                chosen = (tag, turn)
        if chosen is None or last < 0:
            return int(CharTokenizer.eos_token_id)
        tag, turn = chosen
        after = text[last + len(tag) :]
        if "<agent" in after:
            return int(CharTokenizer.eos_token_id)
        if turn.startswith(after) and after != turn:
            return ord(turn[len(after)])
        return int(CharTokenizer.eos_token_id)

    def __call__(self, input_ids, **kwargs):
        del kwargs
        text = self.tokenizer.decode(input_ids[0].tolist())
        nxt = self._predict(text)
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self.vocab_size), -10.0, dtype=torch.float32)
        logits[:, -1, int(nxt)] = 10.0
        return SimpleNamespace(logits=logits)


def _decode_ids_row(tokenizer, tokens, speak):
    ids = [int(token) for token, active in zip(tokens.tolist(), speak.tolist()) if active]
    return tokenizer.decode(ids)


def test_extract_content_stops_at_foreign_xml():
    content, done, own = _extract_content("Yes.</agent1>\n<agent1>um", agent_index=2)
    assert content == "Yes."
    assert done
    assert own is False
    content, done, own = _extract_content("Hi</agent1>", agent_index=0)
    assert content == "Hi"
    assert done
    assert own is True


def test_parse_agent_choice_reads_first_valid_index():
    assert parse_agent_choice("2", num_agents=4, default=0) == 1
    assert parse_agent_choice("agent 3", num_agents=4, default=0) == 2
    assert parse_agent_choice("<agent4>", num_agents=4, default=0) == 3
    assert parse_agent_choice("9", num_agents=4, default=1) == 1
    assert parse_agent_choice("", num_agents=4, default=-1) == -1


def test_prefix_xml_wraps_sole_speaker_runs():
    tokenizer = CharTokenizer()
    ids = torch.tensor([[ord("H"), ord("i"), 0], [0, 0, ord("Y")]], dtype=torch.long)
    activity = torch.tensor([[True, True, False], [False, False, True]])
    text = matrix_prefix_to_xml(ids, activity, tokenizer)
    assert text == "<agent1>Hi</agent1>\n<agent2>Y</agent2>"


def test_xml_round_robin_writes_full_turns_then_next_agent():
    tokenizer = CharTokenizer()
    input_ids = torch.tensor([[[ord("A")], [ord("B")]]], dtype=torch.long)
    activity = torch.tensor([[[True], [True]]])
    model = SuffixCausalLM(
        {
            "<agent1>": "Hi</agent1>",
            "<agent2>": "Yo</agent2>",
        }
    )
    tokens, speak, probs = generate_xml_turn_protocol(
        model,
        input_ids,
        max_new_tokens=4,
        tokenizer=tokenizer,
        protocol="xml_round_robin",
        input_activity_mask=activity,
        temperature=0.0,
        instruction="",
        return_probs=True,
    )
    assert speak[0, 0, :2].tolist() == [True, True]
    assert speak[0, 1, 2:4].tolist() == [True, True]
    assert _decode_ids_row(tokenizer, tokens[0, 0], speak[0, 0]) == "Hi"
    assert _decode_ids_row(tokenizer, tokens[0, 1], speak[0, 1]) == "Yo"
    assert torch.allclose(probs[speak], torch.ones_like(probs[speak]))
    assert "<" not in _decode_ids_row(tokenizer, tokens[0, 0], speak[0, 0])


def test_xml_round_robin_strips_foreign_agent_tags_from_matrix():
    tokenizer = CharTokenizer()
    input_ids = torch.tensor([[[ord("A")], [ord("B")]]], dtype=torch.long)
    activity = torch.tensor([[[True], [True]]])
    model = SuffixCausalLM(
        {
            "<agent1>": "Yes.</agent2>",
            "<agent2>": "Ok</agent2>",
        }
    )
    tokens, speak, _probs = generate_xml_turn_protocol(
        model,
        input_ids,
        max_new_tokens=6,
        tokenizer=tokenizer,
        protocol="xml_round_robin",
        input_activity_mask=activity,
        temperature=0.0,
        instruction="",
        return_probs=True,
    )
    decoded0 = _decode_ids_row(tokenizer, tokens[0, 0], speak[0, 0])
    decoded1 = _decode_ids_row(tokenizer, tokens[0, 1], speak[0, 1])
    assert decoded0 == "Yes."
    assert decoded1 == "Ok"
    assert "<" not in decoded0 + decoded1
    assert "</" not in decoded0 + decoded1


def test_pass_the_baton_hidden_routing_is_not_written_to_matrix():
    tokenizer = CharTokenizer()
    input_ids = torch.tensor([[[ord("A")], [ord("B")]]], dtype=torch.long)
    activity = torch.tensor([[[True], [True]]])
    model = SuffixCausalLM(
        {
            "<agent1>": "Hi</agent1>",
            "<agent2>": "Yo</agent2>",
        },
        routing_reply="2",
    )
    tokens, speak, _probs = generate_xml_turn_protocol(
        model,
        input_ids,
        max_new_tokens=4,
        tokenizer=tokenizer,
        protocol="pass_the_baton",
        input_activity_mask=activity,
        temperature=0.0,
        instruction="",
        return_probs=True,
    )
    decoded0 = _decode_ids_row(tokenizer, tokens[0, 0], speak[0, 0])
    decoded1 = _decode_ids_row(tokenizer, tokens[0, 1], speak[0, 1])
    assert decoded0 == "Hi"
    assert decoded1 == "Yo"
    visible = decoded0 + decoded1
    assert "routing" not in visible
    assert "Who should speak next" not in visible
    assert "<agent" not in visible
    assert "</agent" not in visible


def test_xml_protocol_identity_differs_from_token_round_robin():
    base = {
        "temperature": 1.0,
        "activity_threshold": 0.5,
        "placeholder_token_id": 0,
        "fractions": [0.5],
        "generation_seeds": [0],
        "protocol": "round_robin",
        "round_robin_start": "next_after_cut_reference",
    }
    xml = {
        **base,
        "protocol": "xml_round_robin",
        "turn_format": "xml_agent_tags",
        "hidden_routing": False,
    }
    baton = {
        **base,
        "protocol": "pass_the_baton",
        "turn_format": "xml_agent_tags",
        "hidden_routing": True,
    }
    assert stable_id("generation", base) != stable_id("generation", xml)
    assert stable_id("generation", xml) != stable_id("generation", baton)
