import torch

from model.matrix_qwen import MatrixQwenForCausalLM


def test_matrix_forward_shape(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    b, a, t = 2, 3, 5
    vocab = model.vocab_size

    input_ids = torch.randint(0, vocab, (b, a, t))
    out = model(input_ids=input_ids)

    assert out.logits.shape == (b, a, t, vocab)
    assert out.flat_logits.shape == (b, a * t, vocab)
    assert out.position_ids.shape == (b, a * t)
    assert out.flat_input_ids.shape == (b, a * t)


def test_variable_num_agents(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    vocab = model.vocab_size

    for a in (1, 5):
        assert a <= matrix_config.max_agents
        b, t = 2, 4
        input_ids = torch.randint(0, vocab, (b, a, t))
        out = model(input_ids=input_ids)
        assert out.logits.shape == (b, a, t, vocab)
        assert out.flat_logits.shape == (b, a * t, vocab)


def test_flatten_unflatten_roundtrip(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    b, a, t = 2, 3, 4
    x = torch.arange(b * a * t).reshape(b, a, t)

    flat = model.flatten_matrix(x)
    assert flat.shape == (b, t * a)
    # Column-major: first A entries are time-0 across all agents.
    assert flat[0, 0].item() == x[0, 0, 0].item()
    assert flat[0, 1].item() == x[0, 1, 0].item()  # agent 1, time 0
    assert flat[0, a].item() == x[0, 0, 1].item()  # agent 0, time 1
