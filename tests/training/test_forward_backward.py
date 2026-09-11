import torch

from model.matrix_qwen import MatrixQwenForCausalLM
from training.toy_batch import make_toy_matrix_batch


def test_forward_backward(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.train()
    vocab = model.vocab_size

    b, a, t = 2, 3, 5
    input_ids, labels, activity_mask, activity_labels = make_toy_matrix_batch(
        batch_size=b,
        num_agents=a,
        seq_len=t,
        vocab_size=vocab,
        query_token_id=matrix_config.query_token_id,
    )

    out = model(
        input_ids=input_ids, labels=labels,
        input_activity_mask=activity_mask, activity_labels=activity_labels,
    )

    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert out.content_loss is not None
    assert out.activity_loss is not None
    assert out.turn_reward is not None

    out.loss.backward()

    grad = model.agent_embeddings.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.norm().item() > 0.0

    # The activity head must also receive gradient now that the toy batch
    # exercises the activity BCE + turn-taking reward paths.
    assert model.activity_head.weight.grad is not None
    assert torch.isfinite(model.activity_head.weight.grad).all()
