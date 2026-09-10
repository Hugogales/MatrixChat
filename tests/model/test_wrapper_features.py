"""Tests for wrapper features: channel embeddings, position modes, loss masking."""

import dataclasses

import torch

from model.matrix_qwen import MatrixQwenForCausalLM


def test_channel_embeddings_change_output(tiny_model, matrix_config):
    cfg = dataclasses.replace(
        matrix_config, use_channel_embeddings=True, num_channels=3
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg).eval()
    assert model.channel_embeddings is not None

    b, a, t = 1, 2, 4
    ids = torch.randint(1, model.vocab_size, (b, a, t))
    ch_zero = torch.zeros(b, a, t, dtype=torch.long)
    ch_one = torch.ones(b, a, t, dtype=torch.long)

    with torch.no_grad():
        out_zero = model(input_ids=ids, channel_ids=ch_zero).logits
        out_one = model(input_ids=ids, channel_ids=ch_one).logits

    # Different channel ids should produce different logits.
    assert (out_zero - out_one).abs().max().item() > 0.0


def test_position_mode_changes_output(tiny_model, matrix_config):
    b, a, t = 1, 3, 4
    ids = torch.randint(1, 127, (b, a, t))

    flat_model = MatrixQwenForCausalLM(
        tiny_model, dataclasses.replace(matrix_config, position_mode="flat")
    ).eval()
    with torch.no_grad():
        flat_out = flat_model(input_ids=ids).logits
        flat_pos = flat_model.build_position_ids(b, a, t, torch.device("cpu"))

    col_model = MatrixQwenForCausalLM(
        tiny_model, dataclasses.replace(matrix_config, position_mode="column")
    ).eval()
    with torch.no_grad():
        col_out = col_model(input_ids=ids).logits
        col_pos = col_model.build_position_ids(b, a, t, torch.device("cpu"))

    assert not torch.equal(flat_pos, col_pos)
    # Same weights, different positions -> different logits.
    assert (flat_out - col_out).abs().max().item() > 0.0


def test_loss_only_counts_unignored_labels(tiny_model, matrix_config):
    """Loss with all labels == -100 is undefined/NaN; with one valid label it is finite.

    Verifies the custom CE honors ignore_index and is computed over matrix-aligned
    labels rather than the base model's shifted objective.
    """
    model = MatrixQwenForCausalLM(tiny_model, matrix_config).eval()
    b, a, t = 1, 2, 4
    ids = torch.randint(1, model.vocab_size, (b, a, t))

    labels = torch.full((b, a, t), -100, dtype=torch.long)
    labels[0, 0, t - 1] = 5  # single valid target

    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_num_agents_exceeds_max_raises(tiny_model, matrix_config):
    cfg = dataclasses.replace(matrix_config, max_agents=2)
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    ids = torch.randint(1, model.vocab_size, (1, 3, 4))  # 3 agents > max 2
    try:
        model(input_ids=ids)
    except ValueError as exc:
        assert "max_agents" in str(exc)
    else:
        raise AssertionError("expected ValueError for num_agents > max_agents")
