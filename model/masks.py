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
    inactive_mask: "torch.Tensor | None" = None,
    private_mask: "torch.Tensor | None" = None,
    agent_visibility: "torch.Tensor | None" = None,
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

    Inactive cells (invisible)
    ---------------------------
    ``inactive_mask`` is an optional boolean tensor of shape ``[batch_size, S]``
    (column-major flattened, ``True`` where a cell is inactive/yielding -- i.e.
    ``input_activity_mask`` inverted). An inactive cell is made **invisible**: no
    query may attend to it as a *key*, so other agents cannot see an inactive
    agent's slot at all. The only exception is the diagonal (an inactive cell may
    still attend to itself) so its own softmax row stays well-defined and never
    degenerates -- its output logits are simply discarded by the caller. Note
    this also hides an inactive cell from the *same* agent's later columns, since
    an inactive slot carries no token information.

    Private cells (per-agent visibility ACL)
    -----------------------------------------
    ``private_mask`` is an optional boolean tensor of shape ``[batch_size, S]``
    (column-major flattened, ``True`` where that cell belongs to a PRIVATE turn
    -- see ``data.schema.Turn.visible_to``). ``agent_visibility`` is an optional
    boolean tensor of shape ``[batch_size, num_agents, num_agents]`` where
    ``agent_visibility[b, owner, viewer]`` is ``True`` iff agent ``viewer`` may
    attend to agent ``owner``'s private cells (the diagonal -- an agent viewing
    its own private cells -- should always be ``True``).

    A private key cell is blocked for any query whose agent is not permitted by
    ``agent_visibility``; public cells (``private_mask[k] == False``) are
    completely unaffected by this mechanism, exactly like the existing
    ``inactive_mask`` blocking, this exception is on top of (not instead of)
    the causal/inactive rules above -- a private cell can still be invisible for
    ordinary causal/inactive reasons even to a permitted viewer.

    This guarantees no agent can attend DIRECTLY to another agent's private
    cells. It deliberately does NOT (and cannot, without hiding the speaker's
    entire subsequent identity) prevent an unauthorized viewer from later
    observing the OWNER's own public speech, even though that speech's hidden
    states were computed from context that includes the owner's own private
    cells (an owner always sees its own past, private or not). That is exactly
    how information asymmetry should work in a social-deduction setting: your
    role card itself is unreadable to others, but your subsequent public
    behavior -- which your role legitimately informs -- is fair game to
    observe and reason about. Verified in ``tests/test_masks.py``.

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

    if inactive_mask is None and private_mask is None:
        mask_2d = torch.where(base_allowed, zero, neg)  # [S, S]
        return mask_2d.view(1, 1, seq, seq).expand(batch_size, 1, seq, seq).contiguous()

    # Per-batch: block any key that is an inactive cell, except the diagonal.
    diag = torch.eye(seq, dtype=torch.bool, device=device)              # [S, S]
    allowed = base_allowed.unsqueeze(0).expand(batch_size, seq, seq)    # [B, S, S]

    if inactive_mask is not None:
        key_inactive = inactive_mask.to(torch.bool).to(device).view(batch_size, 1, seq)  # [B, 1, S]
        block_inactive = key_inactive & (~diag).unsqueeze(0)             # [B, S, S]
        allowed = allowed & (~block_inactive)

    if private_mask is not None and agent_visibility is not None:
        key_private = private_mask.to(torch.bool).to(device).view(batch_size, 1, seq)  # [B, 1, S]
        vis = agent_visibility.to(torch.bool).to(device)                # [B, A, A]
        owner_idx = ak.expand(seq, seq)                                  # [Sq, Sk] -> key's agent
        viewer_idx = aq.expand(seq, seq)                                 # [Sq, Sk] -> query's agent
        vis_qk = vis[:, owner_idx, viewer_idx]                           # [B, Sq, Sk]
        block_private = key_private & (~vis_qk) & (~diag).unsqueeze(0)   # never block self
        allowed = allowed & (~block_private)

    mask_2d = torch.where(allowed, zero, neg)                           # [B, S, S]
    return mask_2d.view(batch_size, 1, seq, seq).contiguous()


def build_matrix_causal_mask_new_column_queries(
    batch_size: int,
    num_agents: int,
    seq_len: int,
    device,
    dtype: torch.dtype = torch.float32,
    allow_same_column: bool = False,
    inactive_mask: "torch.Tensor | None" = None,
    private_mask: "torch.Tensor | None" = None,
    agent_visibility: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Attention mask for only the newest column's query rows.

    Returns ``[B, 1, num_agents, num_agents * seq_len]`` -- the last
    ``num_agents`` query rows of the full matrix-causal mask at ``seq_len``.
    Used by incremental KV-cache generation to avoid rebuilding the full
    ``S x S`` mask on every column step.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    full = build_matrix_causal_mask(
        batch_size=batch_size,
        num_agents=num_agents,
        seq_len=seq_len,
        device=device,
        dtype=dtype,
        allow_same_column=allow_same_column,
        inactive_mask=inactive_mask,
        private_mask=private_mask,
        agent_visibility=agent_visibility,
    )
    flat = num_agents * seq_len
    return full[:, :, flat - num_agents : flat, :]
