"""Simple multi-agent generation helper for MatrixChat.

Predicts exactly one next token for every agent row in a single forward pass by
appending a query column and reading the logits at the appended slots.
Multi-token autoregressive generation is intentionally out of scope for now.
"""

from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def generate_next_tokens_for_all_agents(
    model,
    input_ids: torch.LongTensor,        # [B, A, T]
    query_token_id: int,
    channel_ids: Optional[torch.LongTensor] = None,
    temperature: float = 0.0,
) -> torch.LongTensor:
    """Return the next token for every agent: ``[B, A]``.

    Steps:
      1. Append a new time column ``T`` filled with ``query_token_id`` for each agent.
      2. Run a single MatrixQwen forward pass.
      3. Read logits at the appended query column: ``logits[:, :, -1, :]``.
      4. Greedy argmax if ``temperature == 0``, otherwise sample.
    """
    if input_ids.dim() != 3:
        raise ValueError(f"input_ids must be [B, A, T], got shape {tuple(input_ids.shape)}")

    b, a, t = input_ids.shape
    device = input_ids.device

    query_col = torch.full((b, a, 1), query_token_id, dtype=input_ids.dtype, device=device)
    new_input_ids = torch.cat([input_ids, query_col], dim=2)  # [B, A, T+1]

    new_channel_ids = None
    if channel_ids is not None:
        # Reuse the final channel column for the appended query column.
        channel_col = channel_ids[:, :, -1:].clone()
        new_channel_ids = torch.cat([channel_ids, channel_col], dim=2)

    out = model(input_ids=new_input_ids, channel_ids=new_channel_ids)
    last_logits = out.logits[:, :, -1, :]  # [B, A, V]

    if temperature == 0.0:
        next_tokens = last_logits.argmax(dim=-1)  # [B, A]
    else:
        probs = torch.softmax(last_logits / temperature, dim=-1)  # [B, A, V]
        flat_probs = probs.reshape(b * a, -1)
        sampled = torch.multinomial(flat_probs, num_samples=1)  # [B*A, 1]
        next_tokens = sampled.reshape(b, a)

    return next_tokens.long()
