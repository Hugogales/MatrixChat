import torch

from model.matrix_qwen import MatrixQwenForCausalLM
from model.generation import (
    generate_matrix,
    generate_next_tokens_for_all_agents,
    generation_topology_summary,
)


def test_generation_next_tokens(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.eval()
    vocab = model.vocab_size

    b, a, t = 2, 3, 5
    input_ids = torch.randint(1, vocab, (b, a, t))

    next_tokens, speak = generate_next_tokens_for_all_agents(model, input_ids, temperature=0.0)

    assert next_tokens.shape == (b, a)
    assert next_tokens.dtype == torch.long
    assert speak.shape == (b, a)
    assert speak.dtype == torch.bool
    assert int(next_tokens.min()) >= 0
    assert int(next_tokens.max()) < vocab
    # Yielding agents must emit exactly the placeholder id, never a sampled token.
    assert torch.all(next_tokens[~speak] == 0)


def test_generation_sampling(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.eval()
    vocab = model.vocab_size

    b, a, t = 1, 2, 4
    input_ids = torch.randint(1, vocab, (b, a, t))

    next_tokens, speak = generate_next_tokens_for_all_agents(model, input_ids, temperature=1.0)

    assert next_tokens.shape == (b, a)
    assert speak.shape == (b, a)
    assert int(next_tokens.min()) >= 0
    assert int(next_tokens.max()) < vocab


def test_generation_activity_threshold_gates_speaking(tiny_model, matrix_config):
    """A threshold of 0 forces every agent to speak; 1.01 forces every agent to yield."""
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.eval()
    vocab = model.vocab_size
    b, a, t = 1, 2, 4
    input_ids = torch.randint(1, vocab, (b, a, t))

    _, speak_all = generate_next_tokens_for_all_agents(model, input_ids, activity_threshold=-1.0)
    assert bool(speak_all.all())

    _, speak_none = generate_next_tokens_for_all_agents(model, input_ids, activity_threshold=2.0)
    assert not bool(speak_none.any())


def test_generate_matrix_shapes_and_yield_placeholder(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.eval()
    vocab = model.vocab_size
    b, a, t = 1, 2, 3
    input_ids = torch.randint(1, vocab, (b, a, t))

    tokens, speak = generate_matrix(model, input_ids, max_new_tokens=5, placeholder_token_id=0)

    assert tokens.shape == (b, a, 5)
    assert speak.shape == (b, a, 5)
    assert tokens.dtype == torch.long
    assert speak.dtype == torch.bool
    assert torch.all(tokens[~speak] == 0)


def test_generate_matrix_return_probs_is_opt_in_and_matches_speak_decision(tiny_model, matrix_config):
    """return_probs=False (default) keeps the old 2-tuple contract; True adds
    a third P(speak) tensor consistent with the thresholded speak decision."""
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    model.eval()
    vocab = model.vocab_size
    b, a, t = 1, 2, 3
    input_ids = torch.randint(1, vocab, (b, a, t))

    default_result = generate_matrix(model, input_ids, max_new_tokens=4, placeholder_token_id=0)
    assert len(default_result) == 2

    tokens, speak, probs = generate_matrix(
        model, input_ids, max_new_tokens=4, placeholder_token_id=0,
        activity_threshold=0.5, return_probs=True,
    )
    assert probs.shape == speak.shape == tokens.shape
    assert probs.dtype == torch.float32
    assert bool((probs >= 0).all()) and bool((probs <= 1).all())
    # The threshold decision must be consistent with the returned probabilities.
    assert torch.equal(speak, probs > 0.5)


def test_generation_topology_summary_uses_realized_previous_owner():
    probabilities = torch.tensor([[
        [0.8, 0.2],  # agent 0: continue strongly, then yield
        [0.2, 0.8],  # agent 1: yield, then take over
    ]])
    speak = torch.tensor([[
        [True, False],
        [False, True],
    ]])
    initial = torch.tensor([[True, False]])

    summary = generation_topology_summary(probabilities, speak, initial)

    assert summary["columns"] == 2
    assert abs(summary["soft"]["p0"] - 0.16) < 1e-6
    assert abs(summary["soft"]["p_same"] - 0.34) < 1e-6
    assert abs(summary["soft"]["p_handoff"] - 0.34) < 1e-6
    assert abs(summary["soft"]["p_overlap"] - 0.16) < 1e-6
    assert abs(summary["soft_unique_owner"]["same"] - 0.5) < 1e-6
    assert abs(summary["soft_unique_owner"]["handoff"] - 0.5) < 1e-6
    assert summary["realized"] == {
        "silence": 0.0,
        "same": 0.5,
        "handoff": 0.5,
        "overlap": 0.0,
    }
