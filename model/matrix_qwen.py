"""Matrix-structured wrapper around a Hugging Face Qwen3 causal LM.

The wrapper adds an ``agents`` dimension to the model. Inputs are shaped
``[B, A, T]`` (batch, agents, time) and the model produces ``[B, A, T, V]`` logits
in a single forward pass.

Internally the matrix is flattened column-major into a Qwen-compatible sequence::

    flat = input_ids.permute(0, 2, 1).reshape(B, T * A)

so the flat order is::

    (t=0,a=0), (t=0,a=1), ..., (t=0,a=A-1),
    (t=1,a=0), (t=1,a=1), ..., (t=1,a=A-1),
    ...

This is a first, testable prototype -- not the final architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import build_matrix_causal_mask


@dataclass
class MatrixQwenConfig:
    """Configuration for the matrix wrapper (kept small and explicit)."""

    base_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_agents: int = 8
    use_agent_embeddings: bool = True
    use_channel_embeddings: bool = False
    num_channels: int = 1
    position_mode: str = "flat"  # "flat" or "column"
    query_token_id: int = 0
    attn_mask_mode: str = "matrix_causal"  # "none" or "matrix_causal"
    allow_same_column: bool = False


@dataclass
class MatrixCausalLMOutput:
    """Structured output of :meth:`MatrixQwenForCausalLM.forward`."""

    loss: Optional[torch.Tensor]
    logits: torch.Tensor          # [B, A, T, V]
    flat_logits: torch.Tensor     # [B, A*T, V]
    position_ids: torch.Tensor    # [B, A*T]
    flat_input_ids: torch.Tensor  # [B, A*T]


class MatrixQwenForCausalLM(nn.Module):
    """Wrap a Qwen3 causal LM with an agent dimension and matrix interface.

    Parameters
    ----------
    base_model:
        A Hugging Face ``Qwen3ForCausalLM`` / ``AutoModelForCausalLM`` instance.
    matrix_config:
        A :class:`MatrixQwenConfig`.
    """

    def __init__(self, base_model: nn.Module, matrix_config: MatrixQwenConfig):
        super().__init__()
        self.base_model = base_model
        self.matrix_config = matrix_config

        hidden_size = self._infer_hidden_size(base_model)
        self.hidden_size = hidden_size

        # Learned agent embeddings (always trainable, even with a frozen base).
        if matrix_config.use_agent_embeddings:
            self.agent_embeddings = nn.Embedding(matrix_config.max_agents, hidden_size)
            nn.init.normal_(self.agent_embeddings.weight, mean=0.0, std=0.02)
        else:
            self.agent_embeddings = None

        # Optional channel embeddings.
        if matrix_config.use_channel_embeddings:
            self.channel_embeddings = nn.Embedding(matrix_config.num_channels, hidden_size)
            nn.init.normal_(self.channel_embeddings.weight, mean=0.0, std=0.02)
        else:
            self.channel_embeddings = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_hidden_size(base_model: nn.Module) -> int:
        config = getattr(base_model, "config", None)
        if config is not None and getattr(config, "hidden_size", None) is not None:
            return config.hidden_size
        # Fall back to the input embedding dimension.
        emb = base_model.get_input_embeddings()
        return emb.embedding_dim

    @property
    def vocab_size(self) -> int:
        return self.base_model.get_input_embeddings().num_embeddings

    @staticmethod
    def flatten_matrix(x: torch.Tensor) -> torch.Tensor:
        """Flatten ``[B, A, T]`` -> ``[B, T*A]`` in column-major order."""
        b, a, t = x.shape
        return x.permute(0, 2, 1).reshape(b, t * a)

    @staticmethod
    def unflatten_logits(flat_logits: torch.Tensor, num_agents: int, seq_len: int) -> torch.Tensor:
        """Unflatten ``[B, T*A, V]`` -> ``[B, A, T, V]`` (inverse of flatten)."""
        b, _, v = flat_logits.shape
        return flat_logits.reshape(b, seq_len, num_agents, v).permute(0, 2, 1, 3)

    def build_position_ids(self, batch_size: int, num_agents: int, seq_len: int, device) -> torch.Tensor:
        """Build flat position ids ``[B, A*T]`` for the chosen position mode.

        flat:   ``[0, 1, 2, ..., A*T-1]``
        column: ``[0]*A, [1]*A, ..., [T-1]*A`` (all same-time agents share a position)
        """
        seq = num_agents * seq_len
        idx = torch.arange(seq, device=device)
        mode = self.matrix_config.position_mode
        if mode == "flat":
            pos = idx
        elif mode == "column":
            pos = idx // num_agents
        else:
            raise ValueError(f"Unknown position_mode: {mode!r} (expected 'flat' or 'column')")
        return pos.unsqueeze(0).expand(batch_size, seq)

    def _default_agent_ids(self, batch_size: int, num_agents: int, seq_len: int, device) -> torch.Tensor:
        """Agent ids ``[B, A, T]`` where row ``a`` holds the constant value ``a``."""
        agent = torch.arange(num_agents, device=device).view(1, num_agents, 1)
        return agent.expand(batch_size, num_agents, seq_len)

    def get_input_embeddings_with_agents(
        self,
        flat_input_ids: torch.Tensor,
        flat_agent_ids: torch.Tensor,
        flat_channel_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ``token_embeds + agent_embeds (+ channel_embeds)``."""
        token_embeds = self.base_model.get_input_embeddings()(flat_input_ids)
        inputs_embeds = token_embeds

        if self.agent_embeddings is not None:
            inputs_embeds = inputs_embeds + self.agent_embeddings(flat_agent_ids)

        if self.channel_embeddings is not None:
            if flat_channel_ids is None:
                flat_channel_ids = torch.zeros_like(flat_input_ids)
            inputs_embeds = inputs_embeds + self.channel_embeddings(flat_channel_ids)

        return inputs_embeds

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,                 # [B, A, T]
        labels: Optional[torch.LongTensor] = None,   # [B, A, T], -100 ignored
        agent_ids: Optional[torch.LongTensor] = None,    # [B, A, T]
        channel_ids: Optional[torch.LongTensor] = None,  # [B, A, T]
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> MatrixCausalLMOutput:
        if input_ids.dim() != 3:
            raise ValueError(f"input_ids must be [B, A, T], got shape {tuple(input_ids.shape)}")

        b, a, t = input_ids.shape
        device = input_ids.device

        if a > self.matrix_config.max_agents:
            raise ValueError(
                f"num_agents ({a}) exceeds max_agents ({self.matrix_config.max_agents})"
            )

        if agent_ids is None:
            agent_ids = self._default_agent_ids(b, a, t, device)

        flat_input_ids = self.flatten_matrix(input_ids)            # [B, T*A]
        flat_agent_ids = self.flatten_matrix(agent_ids)            # [B, T*A]
        flat_channel_ids = self.flatten_matrix(channel_ids) if channel_ids is not None else None

        inputs_embeds = self.get_input_embeddings_with_agents(
            flat_input_ids, flat_agent_ids, flat_channel_ids
        )

        position_ids = self.build_position_ids(b, a, t, device)    # [B, T*A]

        if attention_mask is None and self.matrix_config.attn_mask_mode == "matrix_causal":
            attention_mask = build_matrix_causal_mask(
                batch_size=b,
                num_agents=a,
                seq_len=t,
                device=device,
                dtype=inputs_embeds.dtype,
                allow_same_column=self.matrix_config.allow_same_column,
            )

        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        flat_logits = outputs.logits  # [B, T*A, V]
        v = flat_logits.shape[-1]

        logits = self.unflatten_logits(flat_logits, num_agents=a, seq_len=t)  # [B, A, T, V]

        loss = None
        if labels is not None:
            # Custom CE over matrix-aligned labels (NOT the base model's shifted loss).
            flat_labels = self.flatten_matrix(labels)  # [B, T*A]
            loss = F.cross_entropy(
                flat_logits.reshape(-1, v),
                flat_labels.reshape(-1),
                ignore_index=-100,
            )

        return MatrixCausalLMOutput(
            loss=loss,
            logits=logits,
            flat_logits=flat_logits,
            position_ids=position_ids,
            flat_input_ids=flat_input_ids,
        )
