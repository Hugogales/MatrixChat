import torch

from model.matrix_qwen import MatrixQwenForCausalLM
from model.generation import generate_next_tokens_for_all_agents


def test_generation_next_tokens(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.eval()
    vocab = model.vocab_size

    b, a, t = 2, 3, 5
    input_ids = torch.randint(0, vocab, (b, a, t))

    next_tokens = generate_next_tokens_for_all_agents(
        model, input_ids, query_token_id=matrix_config.query_token_id, temperature=0.0
    )

    assert next_tokens.shape == (b, a)
    assert next_tokens.dtype == torch.long
    assert int(next_tokens.min()) >= 0
    assert int(next_tokens.max()) < vocab


def test_generation_sampling(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.eval()
    vocab = model.vocab_size

    b, a, t = 1, 2, 4
    input_ids = torch.randint(0, vocab, (b, a, t))

    next_tokens = generate_next_tokens_for_all_agents(
        model, input_ids, query_token_id=matrix_config.query_token_id, temperature=1.0
    )

    assert next_tokens.shape == (b, a)
    assert int(next_tokens.min()) >= 0
    assert int(next_tokens.max()) < vocab
