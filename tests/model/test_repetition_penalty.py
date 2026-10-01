import torch

from model.matrix_qwen import MatrixQwenForCausalLM


def test_recent_token_repetition_penalty_targets_non_target_history_only():
    # At t=1 the human target is token 3; token 2 is the preceding same-row
    # token and should be discouraged, while token 3 remains ordinary CE work.
    input_ids = torch.tensor([[[1, 2, 3, 4]]])
    labels = torch.tensor([[[-100, 3, -100, -100]]])
    active = torch.ones_like(input_ids, dtype=torch.bool)
    neutral = torch.zeros((1, 1, 4, 5))
    repeat = neutral.clone()
    repeat[0, 0, 1, 2] = 10.0
    target = neutral.clone()
    target[0, 0, 1, 3] = 10.0

    neutral_loss = MatrixQwenForCausalLM.recent_token_repetition_loss(
        neutral, input_ids, labels, active, window=2
    )
    repeat_loss = MatrixQwenForCausalLM.recent_token_repetition_loss(
        repeat, input_ids, labels, active, window=2
    )
    target_loss = MatrixQwenForCausalLM.recent_token_repetition_loss(
        target, input_ids, labels, active, window=2
    )

    assert repeat_loss > neutral_loss
    assert target_loss < repeat_loss
    assert torch.isfinite(repeat_loss)


def test_recent_token_repetition_penalty_never_returns_nan_for_nonfinite_logits():
    input_ids = torch.tensor([[[1, 2]]])
    labels = torch.tensor([[[-100, 3]]])
    active = torch.ones_like(input_ids, dtype=torch.bool)
    logits = torch.full((1, 1, 2, 5), float("nan"))

    loss = MatrixQwenForCausalLM.recent_token_repetition_loss(
        logits, input_ids, labels, active, window=2
    )

    assert torch.isfinite(loss)
