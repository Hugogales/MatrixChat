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
        allowed = tk <= tq
    else:
        is_self = (tk == tq) & (ak == aq)
        allowed = (tk < tq) | is_self

    # Use a finite, dtype-safe negative for half precision.
    if dtype in (torch.float16, torch.bfloat16):
        neg = torch.tensor(-1e4, dtype=dtype, device=device)
    else:
        neg = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)

    zero = torch.tensor(0.0, dtype=dtype, device=device)
    mask_2d = torch.where(allowed, zero, neg)  # [S, S]

    mask = mask_2d.view(1, 1, seq, seq).expand(batch_size, 1, seq, seq).contiguous()
    return mask
