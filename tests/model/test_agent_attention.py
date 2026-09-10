import dataclasses
from types import SimpleNamespace

import torch

from model.matrix_qwen import MatrixQwenForCausalLM
from training.checkpoint import apply_checkpoint_to_model, save_checkpoint
from training.toy_batch import create_tiny_qwen3_model


def _inputs(vocab_size, agents=3, steps=4):
    torch.manual_seed(11)
    input_ids = torch.randint(1, vocab_size, (1, agents, steps))
    activity = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, activity


def test_qkv_conditioning_changes_logits_when_agent_ids_change(tiny_model, matrix_config):
    cfg = dataclasses.replace(
        matrix_config,
        position_mode="column",
        use_agent_embeddings=False,
        agent_attention_mode="qkv",
        agent_attention_init_std=0.01,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg).eval()
    input_ids, activity = _inputs(model.vocab_size)
    default_ids = model._default_agent_ids(1, 3, 4, input_ids.device)
    all_zero_ids = torch.zeros_like(default_ids)

    with torch.no_grad():
        distinct = model(
            input_ids, agent_ids=default_ids, input_activity_mask=activity
        ).flat_logits
        collapsed = model(
            input_ids, agent_ids=all_zero_ids, input_activity_mask=activity
        ).flat_logits

    assert not torch.allclose(distinct, collapsed)


def test_qkv_conditioner_receives_finite_gradients(tiny_model, matrix_config):
    cfg = dataclasses.replace(
        matrix_config,
        agent_attention_mode="qkv",
        agent_attention_init_std=0.01,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    input_ids, activity = _inputs(model.vocab_size)
    labels = input_ids.roll(-1, dims=-1)
    labels[:, :, -1] = -100
    activity_labels = activity.long()
    activity_labels[:, :, -1] = -100

    output = model(
        input_ids,
        labels=labels,
        input_activity_mask=activity,
        activity_labels=activity_labels,
    )
    output.loss.backward()

    grads = [
        parameter.grad
        for parameter in model.agent_attention.parameters()
        if parameter.requires_grad
    ]
    assert grads
    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert sum(float(grad.norm()) for grad in grads) > 0


def test_gated_qkv_conditioning_trains_gate(tiny_model, matrix_config):
    cfg = dataclasses.replace(
        matrix_config,
        agent_attention_mode="qkv_gated",
        agent_attention_init_std=0.01,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    input_ids, activity = _inputs(model.vocab_size)
    output = model(input_ids, input_activity_mask=activity)
    output.flat_logits.square().mean().backward()

    gate_grads = [
        layer.gate.weight.grad
        for layer in model.agent_attention.layers
    ]
    assert all(grad is not None for grad in gate_grads)
    assert all(torch.isfinite(grad).all() for grad in gate_grads)
    assert sum(float(grad.norm()) for grad in gate_grads) > 0


def test_simplex_agent_codes_are_fixed_and_equidistant(tiny_model, matrix_config):
    cfg = dataclasses.replace(
        matrix_config,
        agent_attention_mode="qkv",
        agent_attention_representation="simplex",
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    conditioner = model.agent_attention
    codes = conditioner._simplex_codes
    similarities = codes @ codes.T
    off_diagonal = similarities[~torch.eye(len(codes), dtype=torch.bool)]

    assert conditioner.agent_representations is None
    assert torch.allclose(similarities.diag(), torch.ones(len(codes)))
    assert torch.allclose(
        off_diagonal,
        torch.full_like(off_diagonal, -1.0 / (len(codes) - 1)),
        atol=1e-6,
    )


def test_same_agent_attention_bias_changes_logits_and_receives_gradient(
    tiny_model, matrix_config
):
    cfg = dataclasses.replace(
        matrix_config,
        position_mode="column",
        use_agent_embeddings=False,
        agent_same_attention_bias=True,
        agent_same_attention_bias_init=0.0,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    input_ids, activity = _inputs(model.vocab_size)

    neutral = model(input_ids, input_activity_mask=activity).flat_logits.detach()
    with torch.no_grad():
        model.agent_same_attention_bias.layer_bias.fill_(2.0)
    biased = model(input_ids, input_activity_mask=activity).flat_logits
    assert not torch.allclose(neutral, biased)

    biased.square().mean().backward()
    gradient = model.agent_same_attention_bias.layer_bias.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_same_agent_bias_supports_gradient_checkpointing(tiny_model, matrix_config):
    tiny_model.gradient_checkpointing_enable()
    cfg = dataclasses.replace(
        matrix_config,
        position_mode="column",
        agent_same_attention_bias=True,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    model.train()
    input_ids, activity = _inputs(model.vocab_size)
    labels = input_ids.roll(-1, dims=-1)
    labels[:, :, -1] = -100

    output = model(input_ids, labels=labels, input_activity_mask=activity)
    output.loss.backward()

    gradient = model.agent_same_attention_bias.layer_bias.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_relation_bias_same_diff_changes_logits_and_receives_gradient(
    tiny_model, matrix_config
):
    cfg = dataclasses.replace(
        matrix_config,
        position_mode="column",
        use_agent_embeddings=False,
        agent_relation_bias_mode="same_diff",
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    input_ids, activity = _inputs(model.vocab_size)

    neutral = model(input_ids, input_activity_mask=activity).flat_logits.detach()
    with torch.no_grad():
        model.agent_relation_bias.same_bias.fill_(2.0)
        model.agent_relation_bias.diff_bias.fill_(-2.0)
    biased = model(input_ids, input_activity_mask=activity).flat_logits
    assert not torch.allclose(neutral, biased)

    biased.square().mean().backward()
    same_grad = model.agent_relation_bias.same_bias.grad
    diff_grad = model.agent_relation_bias.diff_bias.grad
    assert same_grad is not None and diff_grad is not None
    assert torch.isfinite(same_grad).all() and torch.isfinite(diff_grad).all()
    assert same_grad.abs().sum() > 0
    assert diff_grad.abs().sum() > 0


def test_relation_bias_bilinear_changes_logits_and_receives_gradient(
    tiny_model, matrix_config
):
    cfg = dataclasses.replace(
        matrix_config,
        position_mode="column",
        use_agent_embeddings=False,
        agent_relation_bias_mode="bilinear",
        agent_relation_bias_dim=6,
        agent_relation_bias_init_std=0.05,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    input_ids, activity = _inputs(model.vocab_size)

    output = model(input_ids, input_activity_mask=activity)
    output.flat_logits.square().mean().backward()

    grads = [
        parameter.grad
        for parameter in model.agent_relation_bias.parameters()
        if parameter.requires_grad
    ]
    assert grads
    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert sum(float(grad.norm()) for grad in grads) > 0


def test_relation_bias_bilinear_simplex_codes_are_fixed_and_equidistant(
    tiny_model, matrix_config
):
    cfg = dataclasses.replace(
        matrix_config,
        agent_relation_bias_mode="bilinear",
        agent_relation_bias_representation="simplex",
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    relation_bias = model.agent_relation_bias
    codes = relation_bias._simplex_codes
    similarities = codes @ codes.T
    off_diagonal = similarities[~torch.eye(len(codes), dtype=torch.bool)]

    assert relation_bias.agent_representations is None
    assert torch.allclose(similarities.diag(), torch.ones(len(codes)))
    assert torch.allclose(
        off_diagonal,
        torch.full_like(off_diagonal, -1.0 / (len(codes) - 1)),
        atol=1e-6,
    )


def test_relation_bias_supports_gradient_checkpointing(tiny_model, matrix_config):
    tiny_model.gradient_checkpointing_enable()
    cfg = dataclasses.replace(
        matrix_config,
        position_mode="column",
        agent_relation_bias_mode="bilinear",
        agent_relation_bias_dim=6,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    model.train()
    input_ids, activity = _inputs(model.vocab_size)
    labels = input_ids.roll(-1, dims=-1)
    labels[:, :, -1] = -100

    output = model(input_ids, labels=labels, input_activity_mask=activity)
    output.loss.backward()

    grads = [
        parameter.grad
        for parameter in model.agent_relation_bias.parameters()
        if parameter.requires_grad
    ]
    assert grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_relation_bias_round_trips_through_checkpoint(tmp_path, matrix_config):
    cfg = dataclasses.replace(
        matrix_config,
        agent_relation_bias_mode="bilinear",
        agent_relation_bias_dim=6,
        agent_relation_bias_init_std=0.05,
    )
    torch.manual_seed(44)
    first = MatrixQwenForCausalLM(create_tiny_qwen3_model(), cfg).eval()
    with torch.no_grad():
        first.agent_relation_bias.agent_representations.weight.add_(0.25)

    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(
        first,
        SimpleNamespace(
            lora_enable=False,
            agent_attention_mode="none",
            agent_relation_bias_mode="bilinear",
            agent_relation_bias_representation="learned",
            agent_relation_bias_dim=6,
            agent_relation_bias_init_std=0.05,
        ),
        str(checkpoint_dir),
    )

    torch.manual_seed(999)
    restored = MatrixQwenForCausalLM(create_tiny_qwen3_model(), cfg).eval()
    apply_checkpoint_to_model(restored, str(checkpoint_dir))

    input_ids, activity = _inputs(first.vocab_size)
    with torch.no_grad():
        expected = first(input_ids, input_activity_mask=activity).flat_logits
        actual = restored(input_ids, input_activity_mask=activity).flat_logits
    assert torch.allclose(expected, actual)


def test_dynamic_agent_state_changes_logits_and_receives_gradient(tiny_model, matrix_config):
    cfg = dataclasses.replace(
        matrix_config,
        position_mode="column",
        use_agent_embeddings=False,
        agent_dynamic_state_mode="gru",
        agent_dynamic_state_dim=8,
        agent_dynamic_state_init_std=0.05,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    input_ids, activity = _inputs(model.vocab_size)

    output = model(input_ids, input_activity_mask=activity)
    output.flat_logits.square().mean().backward()

    grads = [
        parameter.grad
        for parameter in model.agent_dynamic_state.parameters()
        if parameter.requires_grad
    ]
    assert grads
    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert sum(float(grad.norm()) for grad in grads) > 0


def test_dynamic_agent_state_is_causal_along_time(tiny_model, matrix_config):
    """Changing a LATER column must not change an EARLIER column's dynamic state."""
    cfg = dataclasses.replace(
        matrix_config,
        position_mode="column",
        use_agent_embeddings=False,
        agent_dynamic_state_mode="gru",
        agent_dynamic_state_dim=8,
        agent_dynamic_state_init_std=0.05,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg).eval()
    input_ids, activity = _inputs(model.vocab_size, agents=2, steps=6)

    modified = input_ids.clone()
    modified[:, :, -1] = (modified[:, :, -1] + 1) % model.vocab_size

    with torch.no_grad():
        token_embeds = model.base_model.get_input_embeddings()(input_ids)
        token_embeds_mod = model.base_model.get_input_embeddings()(modified)
        state = model.agent_dynamic_state(token_embeds)
        state_mod = model.agent_dynamic_state(token_embeds_mod)

    assert torch.allclose(state[:, :, :-1], state_mod[:, :, :-1])
    assert not torch.allclose(state[:, :, -1], state_mod[:, :, -1])


def test_dynamic_agent_state_supports_gradient_checkpointing(tiny_model, matrix_config):
    tiny_model.gradient_checkpointing_enable()
    cfg = dataclasses.replace(
        matrix_config,
        position_mode="column",
        agent_dynamic_state_mode="gru",
        agent_dynamic_state_dim=8,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    model.train()
    input_ids, activity = _inputs(model.vocab_size)
    labels = input_ids.roll(-1, dims=-1)
    labels[:, :, -1] = -100

    output = model(input_ids, labels=labels, input_activity_mask=activity)
    output.loss.backward()

    grads = [
        parameter.grad
        for parameter in model.agent_dynamic_state.parameters()
        if parameter.requires_grad
    ]
    assert grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_dynamic_agent_state_round_trips_through_checkpoint(tmp_path, matrix_config):
    cfg = dataclasses.replace(
        matrix_config,
        agent_dynamic_state_mode="gru",
        agent_dynamic_state_dim=8,
        agent_dynamic_state_init_std=0.05,
    )
    torch.manual_seed(44)
    first = MatrixQwenForCausalLM(create_tiny_qwen3_model(), cfg).eval()
    with torch.no_grad():
        first.agent_dynamic_state.output_proj.weight.add_(0.1)

    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(
        first,
        SimpleNamespace(
            lora_enable=False,
            agent_attention_mode="none",
            agent_dynamic_state_mode="gru",
            agent_dynamic_state_dim=8,
            agent_dynamic_state_init_std=0.05,
        ),
        str(checkpoint_dir),
    )

    torch.manual_seed(999)
    restored = MatrixQwenForCausalLM(create_tiny_qwen3_model(), cfg).eval()
    apply_checkpoint_to_model(restored, str(checkpoint_dir))

    input_ids, activity = _inputs(first.vocab_size)
    with torch.no_grad():
        expected = first(input_ids, input_activity_mask=activity).flat_logits
        actual = restored(input_ids, input_activity_mask=activity).flat_logits
    assert torch.allclose(expected, actual)


def test_joint_row_and_identity_permutation_is_equivariant(tiny_model, matrix_config):
    cfg = dataclasses.replace(
        matrix_config,
        position_mode="column",
        agent_attention_mode="qkv_gated",
        agent_attention_init_std=0.01,
        agent_same_attention_bias=True,
        agent_relation_bias_mode="bilinear",
        agent_relation_bias_dim=6,
        agent_relation_bias_init_std=0.05,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg).eval()
    input_ids, activity = _inputs(model.vocab_size)
    agent_ids = model._default_agent_ids(1, 3, 4, input_ids.device)
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)

    with torch.no_grad():
        original = model(
            input_ids, agent_ids=agent_ids, input_activity_mask=activity
        ).logits
        permuted = model(
            input_ids[:, permutation],
            agent_ids=agent_ids[:, permutation],
            input_activity_mask=activity[:, permutation],
        ).logits[:, inverse]

    assert torch.allclose(original, permuted, atol=2e-5, rtol=2e-5)


def test_attention_agent_state_round_trips_through_checkpoint(tmp_path, matrix_config):
    cfg = dataclasses.replace(
        matrix_config,
        agent_attention_mode="qkv_gated",
        agent_attention_init_std=0.01,
    )
    torch.manual_seed(44)
    first = MatrixQwenForCausalLM(create_tiny_qwen3_model(), cfg).eval()
    with torch.no_grad():
        first.agent_attention.agent_representations.weight.add_(0.25)

    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(
        first,
        SimpleNamespace(
            lora_enable=False,
            agent_attention_mode="qkv_gated",
            agent_attention_representation="learned",
            agent_attention_dim=32,
            agent_attention_init_std=0.01,
        ),
        str(checkpoint_dir),
    )

    torch.manual_seed(999)
    restored = MatrixQwenForCausalLM(create_tiny_qwen3_model(), cfg).eval()
    apply_checkpoint_to_model(restored, str(checkpoint_dir))

    input_ids, activity = _inputs(first.vocab_size)
    with torch.no_grad():
        expected = first(input_ids, input_activity_mask=activity).flat_logits
        actual = restored(input_ids, input_activity_mask=activity).flat_logits
    assert torch.allclose(expected, actual)
