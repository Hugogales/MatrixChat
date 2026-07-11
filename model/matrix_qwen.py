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

from .masks import build_matrix_causal_mask, build_matrix_mask_from_coords


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
    # If set, cells holding this token id are treated as "silence" and made
    # invisible to all other cells in the attention mask. None disables silence.
    silence_token_id: Optional[int] = None
    # Loss weight for the silence token in the custom cross-entropy. Silence is
    # supervised on EVERY silent cell (see data/convert.py), so it is abundant;
    # a weight < 1.0 down-weights it to prevent over-silent behavior. 1.0 = no
    # change. Only applied when silence_token_id is set.
    silence_loss_weight: float = 1.0
    # Give the silence token a LEARNED input representation. The silence id is a
    # reserved vocab token whose base embedding is frozen and meaningless; when
    # this is True (and silence_token_id is set), silent cells use a trainable
    # `silence_embedding` vector instead, so silence context is informative.
    learned_silence_embedding: bool = True
    # If True (and silence_token_id is set), silent cells are physically DROPPED
    # from the flattened stream before the base model (compaction) instead of
    # masked, saving the per-token compute. Logits are scattered back into the
    # [B, A, T, V] grid. Produces identical logits at kept cells as masking.
    drop_silence: bool = False


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

        # Learned silence embedding (trainable, replaces the frozen reserved-token
        # embedding at silent cells so silence context is meaningful).
        if matrix_config.silence_token_id is not None and matrix_config.learned_silence_embedding:
            self.silence_embedding = nn.Embedding(1, hidden_size)
            nn.init.normal_(self.silence_embedding.weight, mean=0.0, std=0.02)
        else:
            self.silence_embedding = None

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
        """Compute ``token_embeds + agent_embeds (+ channel_embeds)``.

        The agent/channel embedding tables are kept in fp32 (good for stable
        optimization) while the base model may run in bf16/fp16. Cast the added
        embeddings to the token-embedding dtype so the sum -- and therefore the
        tensor handed to the base model -- always matches the base model's dtype.
        """
        token_embeds = self.base_model.get_input_embeddings()(flat_input_ids)

        # Replace the (frozen, meaningless) reserved-token embedding at silent
        # cells with the learned silence embedding.
        if self.silence_embedding is not None:
            silent = (flat_input_ids == self.matrix_config.silence_token_id).unsqueeze(-1)
            sil_vec = self.silence_embedding.weight[0].to(token_embeds.dtype)
            token_embeds = torch.where(silent, sil_vec, token_embeds)

        inputs_embeds = token_embeds

        if self.agent_embeddings is not None:
            agent_embeds = self.agent_embeddings(flat_agent_ids).to(token_embeds.dtype)
            inputs_embeds = inputs_embeds + agent_embeds

        if self.channel_embeddings is not None:
            if flat_channel_ids is None:
                flat_channel_ids = torch.zeros_like(flat_input_ids)
            channel_embeds = self.channel_embeddings(flat_channel_ids).to(token_embeds.dtype)
            inputs_embeds = inputs_embeds + channel_embeds

        return inputs_embeds

    # ------------------------------------------------------------------
    # Silence compaction (drop silent cells, scatter logits back)
    # ------------------------------------------------------------------
    def _forward_compacted(
        self,
        inputs_embeds: torch.Tensor,   # [B, S, H]
        position_ids: torch.Tensor,    # [B, S]
        flat_input_ids: torch.Tensor,  # [B, S]
        num_agents: int,
        seq_len: int,
        silence_token_id: int,
    ) -> torch.Tensor:
        """Run the base model over only the non-silent cells, then scatter the
        logits back into the full ``[B, S, V]`` grid (silent slots stay zero).

        Surviving cells keep their original ``position_ids`` (so RoPE is
        unchanged) and their original ``(time, agent)`` coordinates (so the
        matrix-causal rule is preserved). Each batch element keeps a different
        number of cells, so they are packed to the front and padded to the batch
        max length; padding is blocked by the attention mask.
        """
        b, s, h = inputs_embeds.shape
        device = inputs_embeds.device

        keep = flat_input_ids != silence_token_id              # [B, S]
        max_k = int(keep.sum(dim=1).max().item())
        max_k = max(max_k, 1)  # guard against an all-silent batch

        flat_idx = torch.arange(s, device=device)
        time_full = (flat_idx // num_agents).unsqueeze(0).expand(b, s)
        agent_full = (flat_idx % num_agents).unsqueeze(0).expand(b, s)

        # Destination slot (front-packed) for each kept cell.
        dest = keep.long().cumsum(dim=1) - 1                   # [B, S]
        src_b, src_i = keep.nonzero(as_tuple=True)             # kept (batch, flat) indices
        d = dest[src_b, src_i]

        comp_embeds = inputs_embeds.new_zeros(b, max_k, h)
        comp_pos = position_ids.new_zeros(b, max_k)
        comp_time = torch.zeros(b, max_k, dtype=torch.long, device=device)
        comp_agent = torch.zeros(b, max_k, dtype=torch.long, device=device)
        comp_valid = torch.zeros(b, max_k, dtype=torch.bool, device=device)

        comp_embeds[src_b, d] = inputs_embeds[src_b, src_i]
        comp_pos[src_b, d] = position_ids[src_b, src_i]
        comp_time[src_b, d] = time_full[src_b, src_i]
        comp_agent[src_b, d] = agent_full[src_b, src_i]
        comp_valid[src_b, d] = True

        attention_mask = build_matrix_mask_from_coords(
            time=comp_time,
            agent=comp_agent,
            valid=comp_valid,
            dtype=comp_embeds.dtype,
            allow_same_column=self.matrix_config.allow_same_column,
        )

        outputs = self.base_model(
            inputs_embeds=comp_embeds,
            position_ids=comp_pos,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        comp_logits = outputs.logits  # [B, max_k, V]
        v = comp_logits.shape[-1]

        flat_logits = comp_logits.new_zeros(b, s, v)
        flat_logits[src_b, src_i] = comp_logits[src_b, d]
        return flat_logits

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

        silence_token_id = self.matrix_config.silence_token_id
        use_drop = (
            attention_mask is None
            and self.matrix_config.attn_mask_mode == "matrix_causal"
            and self.matrix_config.drop_silence
            and silence_token_id is not None
        )

        if use_drop:
            # Compaction path: physically drop silent cells before the base model.
            flat_logits = self._forward_compacted(
                inputs_embeds, position_ids, flat_input_ids, a, t, silence_token_id
            )
        else:
            if attention_mask is None and self.matrix_config.attn_mask_mode == "matrix_causal":
                silence_mask = None
                if silence_token_id is not None:
                    silence_mask = flat_input_ids == silence_token_id  # [B, T*A]
                attention_mask = build_matrix_causal_mask(
                    batch_size=b,
                    num_agents=a,
                    seq_len=t,
                    device=device,
                    dtype=inputs_embeds.dtype,
                    allow_same_column=self.matrix_config.allow_same_column,
                    silence_mask=silence_mask,
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
            weight = None
            sid = self.matrix_config.silence_token_id
            slw = self.matrix_config.silence_loss_weight
            if sid is not None and slw != 1.0 and 0 <= sid < v:
                # Down-weight the (abundant) silence token so it does not dominate.
                weight = torch.ones(v, dtype=flat_logits.dtype, device=flat_logits.device)
                weight[sid] = slw
            loss = F.cross_entropy(
                flat_logits.reshape(-1, v),
                flat_labels.reshape(-1),
                weight=weight,
                ignore_index=-100,
            )

        return MatrixCausalLMOutput(
            loss=loss,
            logits=logits,
            flat_logits=flat_logits,
            position_ids=position_ids,
            flat_input_ids=flat_input_ids,
        )
