"""CPU-only tests for the vanilla Qwen round-robin generation protocol."""

from types import SimpleNamespace

import torch

from evaluation.paired_continuation import stable_id
from model.generation_round_robin import (
    generate_round_robin_vanilla,
    linearize_active_tokens,
    round_robin_start_agents,
)


class MockCausalLM:
    """Deterministic 1D LM: always peaks on a configured token id."""

    def __init__(self, token_id: int = 7, vocab: int = 32):
        self.token_id = token_id
        self.vocab_size = vocab
        self.training = False
        self.calls: list[torch.Tensor] = []

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def __call__(self, input_ids, **kwargs):
        del kwargs
        self.calls.append(input_ids.detach().cpu().clone())
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self.vocab_size), -10.0, dtype=torch.float32)
        logits[:, -1, self.token_id] = 10.0
        return SimpleNamespace(logits=logits)


def test_linearize_is_column_major_active_only():
    ids = torch.tensor([[[1, 2, 3], [10, 20, 30]]])
    mask = torch.tensor([[[True, True, True], [False, True, False]]])
    context = linearize_active_tokens(ids, mask)[0].tolist()
    assert context == [1, 2, 20, 3]


def test_start_agent_is_next_after_last_sole_speaker():
    mask = torch.tensor(
        [[[True, True, True], [True, True, False]]],
        dtype=torch.bool,
    )
    assert int(round_robin_start_agents(mask)[0]) == 1
    mask = torch.tensor(
        [[[True, False], [False, True]]],
        dtype=torch.bool,
    )
    assert int(round_robin_start_agents(mask)[0]) == 0
    mask = torch.tensor(
        [[[True, True], [True, True]]],
        dtype=torch.bool,
    )
    assert int(round_robin_start_agents(mask)[0]) == 0


def test_round_robin_two_agents_four_columns_forced_pattern():
    model = MockCausalLM(token_id=7)
    input_ids = torch.tensor([[[1, 2, 3], [10, 20, 30]]], dtype=torch.long)
    activity = torch.tensor([[[True, True, True], [False, True, False]]])
    tokens, speak, probs = generate_round_robin_vanilla(
        model,
        input_ids,
        max_new_tokens=4,
        input_activity_mask=activity,
        temperature=0.0,
        placeholder_token_id=0,
        return_probs=True,
    )
    assert tuple(tokens.shape) == (1, 2, 4)
    assert tuple(speak.shape) == (1, 2, 4)
    assert speak.dtype == torch.bool
    assert speak[0].tolist() == [
        [False, True, False, True],
        [True, False, True, False],
    ]
    assert int(speak.sum()) == 4
    assert torch.all(speak.sum(dim=1) == 1)
    assert torch.all(tokens[0, 1, 0] == 7)
    assert torch.all(tokens[0, 0, 1] == 7)
    assert torch.all(tokens[~speak] == 0)
    assert torch.allclose(probs[speak], torch.ones_like(probs[speak]))
    assert torch.allclose(probs[~speak], torch.zeros_like(probs[~speak]))
    assert [int(call.shape[-1]) for call in model.calls] == [4, 5, 6, 7]


def test_round_robin_protocol_identity_differs_from_matrix():
    base = {
        "temperature": 1.0,
        "activity_threshold": 0.5,
        "placeholder_token_id": 0,
        "fractions": [0.5],
        "generation_seeds": [0],
    }
    matrix_id = stable_id("generation", base)
    rr_id = stable_id(
        "generation",
        {
            **base,
            "protocol": "round_robin",
            "round_robin_start": "next_after_cut_reference",
        },
    )
    assert matrix_id != rr_id
