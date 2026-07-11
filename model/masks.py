"""Attention mask builders for MatrixChat.

The matrix is flattened in column-major order, so a flat index ``i`` decodes as::

    time  = i // num_agents
    agent = i %  num_agents

This module builds an additive 4D attention mask of shape ``[B, 1, S, S]`` where
``S = num_agents * seq_len``.
"""

from __future__ import annotations

import torch


def build_matrix_causal_mask(
    batch_size: int,
    num_agents: int,
    seq_len: int,
    device,
    dtype: torch.dtype = torch.float32,
    allow_same_column: bool = False,
    silence_mask: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Build an additive matrix-causal attention mask.

    For a query cell ``q = (tq, aq)`` and key cell ``k = (tk, ak)``:

    - if ``allow_same_column is False``: allowed iff ``tk < tq`` OR ``k is q itself``
      (previous columns, plus the cell's own slot). Other agents in the *same*
      current column are blocked, which avoids same-step teacher-forcing leakage
      across agents.
    - if ``allow_same_column is True``:  allowed iff ``tk <= tq`` (the whole
      current column is visible).

    Self-attention (the diagonal) is always allowed even when
    ``allow_same_column is False``. This is a deliberate deviation from a literal
    "block the entire current column" rule: without it, the first time column
    (``t=0``) would have zero allowed keys, producing an all-masked softmax row
    that degenerates into uniform attention over the whole sequence (including
    future tokens) and leaks future information forward. Allowing the diagonal
    keeps every row well-defined and fully causal.

    Silence (invisible cells)
    -------------------------
    ``silence_mask`` is an optional boolean tensor of shape ``[batch_size, S]``
    (column-major flattened, ``True`` where a cell is silent). A silent cell is
    made **invisible**: no query may attend to it as a *key*, so other agents
    cannot see a silent agent's slot at all. The only exception is the diagonal
    (a silent cell may still attend to itself) so its own softmax row stays
    well-defined and never degenerates -- its output logits are simply discarded
    by the caller. Note this also hides a silent cell from the *same* agent's
    later columns, since a silence slot carries no token information.

    Allowed positions are ``0.0``; disallowed positions are a large negative
    value (``-1e4`` for fp16/bf16 safety, otherwise ``finfo.min``).

    Returns a tensor of shape ``[batch_size, 1, S, S]`` with ``S = num_agents * seq_len``.
    """
    seq = num_agents * seq_len

    # Decode flat indices into (time, agent) -- column-major flattening.
    flat_index = torch.arange(seq, device=device)
    time = flat_index // num_agents   # [S]
    agent = flat_index % num_agents   # [S]

    tq = time.view(seq, 1)   # query time, [S, 1]
    tk = time.view(1, seq)   # key time,   [1, S]
    aq = agent.view(seq, 1)  # query agent
    ak = agent.view(1, seq)  # key agent

    if allow_same_column:
        base_allowed = tk <= tq
    else:
        is_self = (tk == tq) & (ak == aq)
        base_allowed = (tk < tq) | is_self  # [S, S]

    # Use a finite, dtype-safe negative for half precision.
    if dtype in (torch.float16, torch.bfloat16):
        neg = torch.tensor(-1e4, dtype=dtype, device=device)
    else:
        neg = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)
    zero = torch.tensor(0.0, dtype=dtype, device=device)

    if silence_mask is None:
        mask_2d = torch.where(base_allowed, zero, neg)  # [S, S]
        return mask_2d.view(1, 1, seq, seq).expand(batch_size, 1, seq, seq).contiguous()

    # Per-batch: block any key that is a silent cell, except the diagonal.
    diag = torch.eye(seq, dtype=torch.bool, device=device)              # [S, S]
    key_silent = silence_mask.to(torch.bool).to(device).view(batch_size, 1, seq)  # [B, 1, S]
    block_silent = key_silent & (~diag).unsqueeze(0)                    # [B, S, S]
    allowed = base_allowed.unsqueeze(0) & (~block_silent)              # [B, S, S]
    mask_2d = torch.where(allowed, zero, neg)                          # [B, S, S]
    return mask_2d.view(batch_size, 1, seq, seq).contiguous()


def build_matrix_mask_from_coords(
    time: torch.Tensor,
    agent: torch.Tensor,
    valid: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    allow_same_column: bool = False,
) -> torch.Tensor:
    """Build an additive matrix-causal mask from explicit per-token coordinates.

    Unlike :func:`build_matrix_causal_mask`, this does NOT assume a dense
    rectangular ``A*T`` layout. It is used by the silence-dropping (compaction)
    path, where each batch element keeps a different subset of cells padded to a
    common length ``L``.

    Parameters
    ----------
    time, agent:
        Long tensors ``[B, L]`` giving each (possibly compacted) token's original
        time column and agent row.
    valid:
        Bool tensor ``[B, L]``; ``True`` for real tokens, ``False`` for padding.
        Padding cells are blocked as keys (so nothing attends to them) but the
        diagonal is always kept so every query row is well-defined.
    allow_same_column:
        Same semantics as :func:`build_matrix_causal_mask`.

    Returns an additive mask ``[B, 1, L, L]``.
    """
    batch_size, length = time.shape
    device = time.device

    tq = time.unsqueeze(2)   # [B, L, 1]
    tk = time.unsqueeze(1)   # [B, 1, L]
    aq = agent.unsqueeze(2)
    ak = agent.unsqueeze(1)

    if allow_same_column:
        base = tk <= tq
    else:
        base = (tk < tq) | ((tk == tq) & (ak == aq))

    key_valid = valid.to(torch.bool).unsqueeze(1)                       # [B, 1, L]
    diag = torch.eye(length, dtype=torch.bool, device=device).unsqueeze(0)  # [1, L, L]
    allowed = (base & key_valid) | diag                                # [B, L, L]

    if dtype in (torch.float16, torch.bfloat16):
        neg = torch.tensor(-1e4, dtype=dtype, device=device)
    else:
        neg = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)
    zero = torch.tensor(0.0, dtype=dtype, device=device)

    mask_2d = torch.where(allowed, zero, neg)                          # [B, L, L]
    return mask_2d.view(batch_size, 1, length, length).contiguous()
