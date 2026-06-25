"""Behavioral correctness tests for the matrix-causal attention mask.

These go beyond shape checks: they verify the wrapper is actually causal at the
column level and does not leak information across agents within the same column.
"""

import torch

from model.matrix_qwen import MatrixQwenForCausalLM


def _model(tiny_model, matrix_config):
    m = MatrixQwenForCausalLM(tiny_model, matrix_config)
    return m.eval()


def test_logits_all_finite(tiny_model, matrix_config):
    model = _model(tiny_model, matrix_config)
    b, a, t = 1, 3, 4
    ids = torch.randint(1, model.vocab_size, (b, a, t))
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    assert torch.isfinite(logits).all()


def test_no_future_leakage(tiny_model, matrix_config):
    """Changing a token in a future column must not change earlier-column logits."""
    model = _model(tiny_model, matrix_config)
    b, a, t = 1, 2, 4
    ids = torch.randint(1, model.vocab_size, (b, a, t))

    ids_future = ids.clone()
    ids_future[0, 0, t - 1] = (ids[0, 0, t - 1] + 5) % model.vocab_size

    with torch.no_grad():
        base = model(input_ids=ids).logits
        changed = model(input_ids=ids_future).logits

    # Column 0 logits must be unaffected by a change in the final column.
    diff = (base[0, :, 0, :] - changed[0, :, 0, :]).abs().max().item()
    assert diff == 0.0


def test_no_same_column_cross_agent_leakage(tiny_model, matrix_config):
    """Agent 0's logits at time t must not depend on agent 1's token at time t."""
    model = _model(tiny_model, matrix_config)
    b, a, t = 1, 2, 4
    ids = torch.randint(1, model.vocab_size, (b, a, t))

    ids_other = ids.clone()
    ids_other[0, 1, 1] = (ids[0, 1, 1] + 7) % model.vocab_size

    with torch.no_grad():
        base = model(input_ids=ids).logits
        changed = model(input_ids=ids_other).logits

    diff = (base[0, 0, 1, :] - changed[0, 0, 1, :]).abs().max().item()
    assert diff == 0.0
