import torch

from model.masks import build_matrix_causal_mask


def _is_allowed(mask, q, k):
    """A position is allowed when its additive mask value is 0."""
    return mask[0, 0, q, k].item() == 0.0


def test_matrix_causal_mask_shape():
    a, t, b = 2, 3, 4
    mask = build_matrix_causal_mask(b, a, t, device=torch.device("cpu"))
    s = a * t
    assert mask.shape == (b, 1, s, s)


def test_matrix_causal_mask_no_same_column():
    # A=2, T=3. Column-major flat index i -> time=i//2, agent=i%2.
    #   i: 0(t0,a0) 1(t0,a1) 2(t1,a0) 3(t1,a1) 4(t2,a0) 5(t2,a1)
    a, t = 2, 3
    mask = build_matrix_causal_mask(1, a, t, device=torch.device("cpu"), allow_same_column=False)

    q = 2  # query at t=1, a=0
    # Can attend previous column t=0 (keys 0, 1).
    assert _is_allowed(mask, q, 0)
    assert _is_allowed(mask, q, 1)
    # Can attend itself (diagonal) -- key 2 is the query's own slot.
    assert _is_allowed(mask, q, 2)
    # Cannot attend the OTHER agent in its own column t=1 (key 3).
    assert not _is_allowed(mask, q, 3)
    # Cannot attend future column t=2 (keys 4, 5).
    assert not _is_allowed(mask, q, 4)
    assert not _is_allowed(mask, q, 5)


def test_matrix_causal_mask_first_column_has_self():
    # The first column must NOT be fully masked (otherwise softmax degenerates
    # into uniform attention over the whole sequence and leaks the future).
    a, t = 2, 3
    mask = build_matrix_causal_mask(1, a, t, device=torch.device("cpu"), allow_same_column=False)

    # t=0 cell (key 0) can attend itself but nothing else.
    assert _is_allowed(mask, 0, 0)        # self
    assert not _is_allowed(mask, 0, 1)    # other agent, same column
    assert not _is_allowed(mask, 0, 2)    # future
    # Every query row has at least one allowed key.
    s = a * t
    for q in range(s):
        assert any(_is_allowed(mask, q, k) for k in range(s))


def test_matrix_causal_mask_allow_same_column():
    a, t = 2, 3
    mask = build_matrix_causal_mask(1, a, t, device=torch.device("cpu"), allow_same_column=True)

    q = 2  # query at t=1, a=0
    # Previous column allowed.
    assert _is_allowed(mask, q, 0)
    assert _is_allowed(mask, q, 1)
    # Same column now allowed.
    assert _is_allowed(mask, q, 2)
    assert _is_allowed(mask, q, 3)
    # Future column still masked.
    assert not _is_allowed(mask, q, 4)
    assert not _is_allowed(mask, q, 5)


# ---------------------------------------------------------------------------
# Private-cell attention ACL (agent_visibility), see data.schema.Turn.visible_to
# ---------------------------------------------------------------------------


def test_private_mask_blocks_non_permitted_viewer():
    # A=3, T=2. Agent 0's cell at t=0 is private, visible only to agent 1.
    a, t = 3, 2
    private_mask = torch.zeros(1, t * a, dtype=torch.bool)
    private_mask[0, 0] = True  # flat idx 0 == (t=0, agent=0)
    agent_visibility = torch.zeros(1, a, a, dtype=torch.bool)
    for i in range(a):
        agent_visibility[0, i, i] = True
    agent_visibility[0, 0, 1] = True  # agent 1 may view agent 0's private cells

    mask = build_matrix_causal_mask(
        1, a, t, device=torch.device("cpu"), allow_same_column=True,
        private_mask=private_mask, agent_visibility=agent_visibility,
    )
    # Query at t=1 for each agent (flat idx = 1*3 + agent), key = (t=0, agent=0) = flat 0.
    assert _is_allowed(mask, 1 * a + 0, 0)   # agent 0 (self) always allowed
    assert _is_allowed(mask, 1 * a + 1, 0)   # agent 1 (permitted) allowed
    assert not _is_allowed(mask, 1 * a + 2, 0)  # agent 2 (not permitted) blocked


def test_private_mask_diagonal_always_visible_to_self():
    a, t = 2, 1
    private_mask = torch.ones(1, t * a, dtype=torch.bool)  # everything private
    agent_visibility = torch.zeros(1, a, a, dtype=torch.bool)  # nobody sees anybody, not even self by table
    mask = build_matrix_causal_mask(
        1, a, t, device=torch.device("cpu"),
        private_mask=private_mask, agent_visibility=agent_visibility,
    )
    # The diagonal (an agent attending to its own current-column slot) must
    # remain allowed regardless of the visibility table, exactly like the
    # existing inactive_mask exception -- otherwise that query row would have
    # zero allowed keys and softmax would degenerate.
    assert _is_allowed(mask, 0, 0)
    assert _is_allowed(mask, 1, 1)


def test_public_cells_unaffected_by_private_mechanism():
    a, t = 2, 2
    private_mask = torch.zeros(1, t * a, dtype=torch.bool)  # nothing private
    agent_visibility = torch.zeros(1, a, a, dtype=torch.bool)  # would block everything if it mattered
    mask_with_private_args = build_matrix_causal_mask(
        1, a, t, device=torch.device("cpu"), allow_same_column=True,
        private_mask=private_mask, agent_visibility=agent_visibility,
    )
    mask_without = build_matrix_causal_mask(1, a, t, device=torch.device("cpu"), allow_same_column=True)
    assert torch.equal(mask_with_private_args, mask_without)


def test_private_mask_end_to_end_zero_direct_influence(tiny_model):
    """No amount of perturbing a private cell's token may change the logits of
    an agent that both (a) has no attention permission to it, AND (b) has no
    other legitimate channel (the owner never speaks publicly afterward) --
    this isolates the direct-attention leak from legitimate indirect
    influence (an owner's own later PUBLIC speech naturally reflects what it
    privately knows, and that is intentionally still observable by others)."""
    from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM

    torch.manual_seed(0)
    model = MatrixQwenForCausalLM(
        tiny_model, MatrixQwenConfig(max_agents=8, position_mode="column")
    )
    model.eval()

    a, t = 2, 6
    vocab = model.vocab_size
    input_ids = torch.randint(1, vocab, (1, a, t))
    activity_mask = torch.zeros(1, a, t, dtype=torch.bool)
    activity_mask[0, 0, 0:2] = True   # agent 0: private, then yields for good
    activity_mask[0, 1, :] = True     # agent 1: active throughout
    private_mask = torch.zeros(1, a, t, dtype=torch.bool)
    private_mask[0, 0, 0:2] = True
    agent_visibility = torch.zeros(1, a, a, dtype=torch.bool)
    agent_visibility[0, 0, 0] = True
    agent_visibility[0, 1, 1] = True  # agent 1 NOT permitted to see agent 0's private cells

    with torch.no_grad():
        out1 = model(
            input_ids=input_ids, input_activity_mask=activity_mask,
            input_private_mask=private_mask, agent_visibility=agent_visibility,
        )
    perturbed = input_ids.clone()
    perturbed[0, 0, 0:2] = (perturbed[0, 0, 0:2] + 1) % vocab
    with torch.no_grad():
        out2 = model(
            input_ids=perturbed, input_activity_mask=activity_mask,
            input_private_mask=private_mask, agent_visibility=agent_visibility,
        )

    assert torch.equal(out1.logits[:, 1], out2.logits[:, 1])
