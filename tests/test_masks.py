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
