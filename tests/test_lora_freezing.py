import importlib

import pytest

from model.matrix_qwen import MatrixQwenForCausalLM
from model.lora_utils import (
    count_trainable_parameters,
    freeze_base_model,
    unfreeze_first_n_layers,
    unfreeze_last_n_layers,
)


def _base_param_count_trainable(model):
    return sum(p.numel() for p in model.base_model.parameters() if p.requires_grad)


def test_freeze_keeps_agent_embeddings_trainable(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    freeze_base_model(model)

    # Base frozen.
    assert _base_param_count_trainable(model) == 0
    # Agent embeddings still trainable.
    assert model.agent_embeddings.weight.requires_grad is True
    assert count_trainable_parameters(model) == model.agent_embeddings.weight.numel()


def test_unfreeze_last_n_layers(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    freeze_base_model(model)
    assert _base_param_count_trainable(model) == 0

    unfrozen = unfreeze_last_n_layers(model, 1)
    assert unfrozen == 1
    # Some base params are now trainable.
    assert _base_param_count_trainable(model) > 0


def test_unfreeze_first_n_layers(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    freeze_base_model(model)
    assert _base_param_count_trainable(model) == 0

    unfrozen = unfreeze_first_n_layers(model, 1)
    assert unfrozen == 1
    trainable_after_first = _base_param_count_trainable(model)
    assert trainable_after_first > 0

    # Unfreezing the first layer does NOT unfreeze the whole model: with a 2-layer
    # tiny model, one unfrozen layer is strictly fewer params than both.
    all_base = sum(p.numel() for p in model.base_model.parameters())
    assert trainable_after_first < all_base


def test_lora_apply_or_skip(tiny_model, matrix_config):
    """If PEFT is installed, LoRA should apply; otherwise it should raise clearly."""
    from types import SimpleNamespace

    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    args = SimpleNamespace(
        lora_enable=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    from model.lora_utils import apply_lora_if_enabled

    if importlib.util.find_spec("peft") is None:
        with pytest.raises(ImportError):
            apply_lora_if_enabled(model, args)
        pytest.skip("peft not installed; verified clean error path.")
    else:
        model = apply_lora_if_enabled(model, args)
        # Agent embeddings remain trainable regardless of LoRA.
        assert model.agent_embeddings.weight.requires_grad is True
        # At least some trainable params exist (LoRA adapters + embeddings).
        assert count_trainable_parameters(model) > 0
