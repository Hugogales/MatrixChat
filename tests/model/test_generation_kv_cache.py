import torch

from model.generation import generate_matrix
from model.matrix_qwen import MatrixQwenForCausalLM


def test_generate_matrix_kv_cache_matches_uncached(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.eval()
    vocab = model.vocab_size

    b, a, t = 1, 2, 4
    input_ids = torch.randint(1, vocab, (b, a, t))
    activity = torch.ones_like(input_ids, dtype=torch.bool)

    cached = generate_matrix(
        model,
        input_ids,
        max_new_tokens=6,
        input_activity_mask=activity,
        temperature=0.0,
        placeholder_token_id=0,
        use_kv_cache=True,
    )
    uncached = generate_matrix(
        model,
        input_ids,
        max_new_tokens=6,
        input_activity_mask=activity,
        temperature=0.0,
        placeholder_token_id=0,
        use_kv_cache=False,
    )
    assert torch.equal(cached[0], uncached[0])
    assert torch.equal(cached[1], uncached[1])


def test_generate_matrix_kv_cache_return_probs(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.eval()
    vocab = model.vocab_size
    input_ids = torch.randint(1, vocab, (1, 2, 3))

    cached = generate_matrix(
        model,
        input_ids,
        max_new_tokens=3,
        temperature=0.0,
        placeholder_token_id=0,
        return_probs=True,
        use_kv_cache=True,
    )
    uncached = generate_matrix(
        model,
        input_ids,
        max_new_tokens=3,
        temperature=0.0,
        placeholder_token_id=0,
        return_probs=True,
        use_kv_cache=False,
    )
    assert torch.equal(cached[0], uncached[0])
    assert torch.equal(cached[1], uncached[1])
    assert torch.allclose(cached[2], uncached[2])
