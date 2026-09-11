import torch

from scripts.eval.knowledge_retention_probe import target_mask


def test_target_mask_scores_only_completion_tokens():
    # prompt tokens occupy indices 0,1,2. Prediction column 2 targets the
    # first completion token at index 3; columns 0 and 1 remain context-only.
    mask = target_mask(sequence_length=6, prompt_length=3)
    assert mask.tolist() == [False, False, True, True, True]


def test_target_mask_handles_single_token_prompt():
    mask = target_mask(sequence_length=4, prompt_length=1)
    assert mask.tolist() == [True, True, True]
    assert mask.dtype == torch.bool
