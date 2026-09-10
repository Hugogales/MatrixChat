"""DialogueRNN-inspired dynamic per-agent state for MatrixChat.

Every other agent-identity mechanism in ``model/agent_attention.py`` is
*static*: one vector (or simplex code) per agent row, fixed for the whole
example. This module instead maintains one recurrent state PER AGENT ROW
that accumulates causally across that row's own time axis, following
Majumder et al.'s DialogueRNN (AAAI 2019, per-party GRU state updated from
each party's own utterances) rather than a fixed seat-indexed code.

Kept intentionally simple relative to DialogueRNN itself: DialogueRNN
updates its party state from pooled utterance-level context (and a separate
global/emotion state); this version runs directly on token embeddings at
the input, one GRU lane per agent row, so it plugs into the matrix input
pathway without needing an utterance-boundary abstraction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicAgentState(nn.Module):
    """Per-agent causal GRU state, injected additively into the input.

    ``forward`` takes ``[B, A, T, input_dim]`` cell-level embeddings (already
    masked to zero at inactive/silent cells if desired by the caller) and
    returns ``[B, A, T, hidden_size]``: at ``(a, t)``, a projection of the
    GRU state accumulated from agent ``a``'s own cells at times ``<= t``
    only -- ``nn.GRU`` is inherently causal along its sequence axis, so no
    extra masking is needed to prevent future leakage.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_size: int,
        *,
        output_init_std: float = 1e-3,
    ):
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        self.state_dim = state_dim
        self.gru = nn.GRU(input_dim, state_dim, batch_first=True)
        self.output_proj = nn.Linear(state_dim, hidden_size)
        # Small init: the GRU itself starts with generic PyTorch defaults
        # (not near-zero), but the OUTPUT projection is small so the initial
        # contribution to the model is a gentle perturbation, consistent
        # with every other new interface module in this project.
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=output_init_std)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, cell_embeds: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_state(cell_embeds)
        return output

    def forward_with_state(
        self,
        cell_embeds: torch.Tensor,
        hidden: "torch.Tensor | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the GRU over ``[B, A, T, H]`` and return outputs plus final hidden."""
        b, a, t, h = cell_embeds.shape
        flat = cell_embeds.reshape(b * a, t, h)
        gru_out, hidden = self.gru(flat.float(), hidden)
        gru_out = gru_out.reshape(b, a, t, self.state_dim)
        return self.output_proj(gru_out), hidden

    def forward_step(
        self,
        column_embeds: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one matrix column ``[B, A, H]`` and return ``[B, A, hidden_size]``."""
        b, a, h = column_embeds.shape
        flat = column_embeds.reshape(b * a, 1, h)
        gru_out, hidden = self.gru(flat.float(), hidden)
        return self.output_proj(gru_out.squeeze(1)), hidden
