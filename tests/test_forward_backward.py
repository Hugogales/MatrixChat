import torch

from model.matrix_qwen import MatrixQwenForCausalLM
from training.toy_batch import make_toy_matrix_batch


def test_forward_backward(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.train()
    vocab = model.vocab_size

    b, a, t = 2, 3, 5
    input_ids, labels = make_toy_matrix_batch(
        batch_size=b,
        num_agents=a,
        seq_len=t,
        vocab_size=vocab,
        query_token_id=matrix_config.query_token_id,
    )

    out = model(input_ids=input_ids, labels=labels)

    assert out.loss is not None
    assert torch.isfinite(out.loss)

    out.loss.backward()

    grad = model.agent_embeddings.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.norm().item() > 0.0
