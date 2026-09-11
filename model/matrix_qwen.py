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

Turn-taking (activity) head
----------------------------
There is no vocabulary "silence token". Instead every cell has TWO heads on its
final hidden state:

- ``activity_head``: a scalar ``P(this agent speaks at the NEXT column)``.
- the base model's vocabulary head: ``P(token | speaking)``.

The normalized per-cell policy is ``P(yield) = 1 - P(speak)`` and
``P(token v) = P(speak) * P_vocab(v | speak)``. Cells that are not currently
active (``input_activity_mask == False``) use a trainable ``inactive_embedding``
instead of looking up a (frozen, semantically-empty) placeholder token id, so
"being silent" has a meaningful, learned input representation.

This is a first, testable prototype -- not the final architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent_attention import AgentQKVConditioner, AgentRelationBias, AgentSameAttentionBias
from .dynamic_agent_state import DynamicAgentState
from .masks import build_matrix_causal_mask, build_matrix_causal_mask_new_column_queries
from .generation_cache import MatrixGenerationCache
from .turn_reward import (
    FloorControlConfig,
    TurnRewardConfig,
    floor_control_reward,
    same_handoff_cross_entropy,
    turn_taking_reward,
)


@dataclass
class MatrixQwenConfig:
    """Configuration for the matrix wrapper (kept small and explicit)."""

    base_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_agents: int = 8
    use_agent_embeddings: bool = True
    # Optional attention-level identity path. ``none`` preserves the historical
    # input-only additive embedding behavior. ``qkv`` and ``qkv_gated`` inject
    # projected agent representations into every Q/K/V projection output.
    agent_attention_mode: str = "none"  # "none", "qkv", or "qkv_gated"
    agent_attention_representation: str = "learned"  # "learned" or "simplex"
    agent_attention_dim: int = 32
    agent_attention_init_std: float = 1e-3
    # Optional permutation-equivariant relation cue in attention logits.
    agent_same_attention_bias: bool = False
    agent_same_attention_bias_init: float = 0.0
    # Optional GENERALIZED relation cue, independent of the legacy
    # same-agent-only bias above (see model/agent_attention.py's
    # AgentRelationBias). "none" disables it (default; identical to
    # historical behavior). "same_diff" learns separate same-agent and
    # different-agent scalars per layer. "bilinear" learns an
    # AgentFormer-inspired identity-conditioned attention bias.
    agent_relation_bias_mode: str = "none"  # "none", "same_diff", or "bilinear"
    agent_relation_bias_representation: str = "learned"  # "learned" or "simplex"
    agent_relation_bias_dim: int = 8
    agent_relation_bias_init_std: float = 1e-3
    agent_relation_bias_same_init: float = 0.0
    agent_relation_bias_diff_init: float = 0.0
    # Optional DialogueRNN-inspired dynamic per-agent state (see
    # model/dynamic_agent_state.py): a causal per-agent-row GRU over that
    # row's own (activity-masked) token embeddings, injected additively into
    # the input alongside the static agent_embeddings above. "none" disables
    # it (default; identical to historical behavior).
    agent_dynamic_state_mode: str = "none"  # "none" or "gru"
    agent_dynamic_state_dim: int = 64
    agent_dynamic_state_init_std: float = 1e-3
    use_channel_embeddings: bool = False
    num_channels: int = 1
    position_mode: str = "flat"  # "flat" or "column"
    query_token_id: int = 0
    attn_mask_mode: str = "matrix_causal"  # "none" or "matrix_causal"
    allow_same_column: bool = False
    # Dummy input id used at inactive (yielding) cells. Its embedding is always
    # overridden by the learned `inactive_embedding`, so the value itself is
    # arbitrary as long as it is a valid vocab index.
    placeholder_token_id: int = 0
    # Stage-A loss weights: total = lambda_content * content_CE
    #                                + lambda_activity * activity_BCE
    #                                - lambda_reward * mean(turn_taking_reward).
    lambda_content: float = 1.0
    lambda_activity: float = 1.0
    lambda_reward: float = 0.05
    # Proper-scoring identity-transition objective. It only compares SAME
    # versus HANDOFF on clean unique-owner -> unique-owner human transitions,
    # masking silence/overlap entirely. Zero preserves historical behavior.
    lambda_same_handoff: float = 0.0
    # Class weight for the activity BCE's "speak" (positive) label, in
    # [0, 1]. 0.5 = balanced (equal weight on speak vs. silence mistakes,
    # the long-standing default). >0.5 penalizes "should have spoken but
    # predicted silence" mistakes MORE than the reverse, directly pushing
    # predicted P(speak) up -- a different lever from lambda_reward, which
    # only nudges a soft differentiable proxy rather than the supervised
    # classification objective itself.
    activity_pos_weight: float = 0.5
    # Which floor-control mechanism computes turn_reward_mean (see
    # model/turn_reward.py's module docstring for the full rationale).
    # "linear": the original hand-tuned additive shape-parameter reward
    #   (speak_grace/tau, silence_grace/tau, overlap_*, handoff_bonus_weight
    #   below). Kept as the default for exact backward compatibility with
    #   every existing recipe/checkpoint/search history.
    # "balanced_ce": the 2026-08-24 rearchitecture -- a balanced 4-way
    #   (silence/same/handoff/overlap) proper-scoring cross-entropy with NO
    #   hand-tuned shape parameters. See floor_control_reward()'s docstring
    #   for the exploit-resistance analysis motivating this option.
    turn_reward_mode: str = "linear"  # "linear" or "balanced_ce"
    # Turn-taking reward shape (see model/turn_reward.py). Only used when
    # turn_reward_mode == "linear".
    speak_grace: float = 50.0
    speak_tau: float = 2500.0
    silence_grace: float = 1.0
    silence_tau: float = 20.0
    overlap_base_weight: float = 0.35
    overlap_max_weight: float = 1.5
    overlap_grace: float = 1.0
    overlap_tau: float = 3.0
    handoff_bonus_weight: float = 0.0
    # Reference/baseline category rates for turn_reward_mode == "balanced_ce"
    # (see FloorControlConfig). Only affects the logged reward's zero point,
    # never what is optimized.
    floor_control_ref_silence: float = 0.0784
    floor_control_ref_same: float = 0.7805
    floor_control_ref_handoff: float = 0.0675
    floor_control_ref_overlap: float = 0.0735
    # Per-category combination weights for turn_reward_mode == "balanced_ce"
    # (see FloorControlConfig.weight_* docstring). All 1.0 (default)
    # reproduces the original unweighted mean exactly.
    floor_control_weight_silence: float = 1.0
    floor_control_weight_same: float = 1.0
    floor_control_weight_handoff: float = 1.0
    floor_control_weight_overlap: float = 1.0
    floor_control_adaptive_weight_alpha: float = 0.0
    floor_control_adaptive_prior_strength: float = 32.0


@dataclass
class MatrixCausalLMOutput:
    """Structured output of :meth:`MatrixQwenForCausalLM.forward`."""

    loss: Optional[torch.Tensor]
    logits: torch.Tensor          # [B, A, T, V]
    flat_logits: torch.Tensor     # [B, A*T, V]
    position_ids: torch.Tensor    # [B, A*T]
    flat_input_ids: torch.Tensor  # [B, A*T]
    activity_logits: Optional[torch.Tensor] = None  # [B, A, T]; P(speak at t+1)
    content_loss: Optional[torch.Tensor] = None
    activity_loss: Optional[torch.Tensor] = None
    turn_reward: Optional[torch.Tensor] = None  # mean reward (higher is better)
    turn_reward_stats: Optional[dict] = None
    same_handoff_loss: Optional[torch.Tensor] = None
    same_handoff_stats: Optional[dict] = None


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

        if matrix_config.agent_attention_mode == "none":
            self.agent_attention = None
        else:
            self.agent_attention = AgentQKVConditioner(
                base_model,
                max_agents=matrix_config.max_agents,
                mode=matrix_config.agent_attention_mode,
                representation=matrix_config.agent_attention_representation,
                representation_dim=matrix_config.agent_attention_dim,
                init_std=matrix_config.agent_attention_init_std,
            )
            self.agent_attention.install_hooks(base_model)
        self.agent_same_attention_bias = (
            AgentSameAttentionBias(
                base_model,
                init=matrix_config.agent_same_attention_bias_init,
            )
            if matrix_config.agent_same_attention_bias
            else None
        )
        if self.agent_same_attention_bias is not None:
            self.agent_same_attention_bias.install_hooks(base_model)

        if matrix_config.agent_relation_bias_mode == "none":
            self.agent_relation_bias = None
        else:
            self.agent_relation_bias = AgentRelationBias(
                base_model,
                mode=matrix_config.agent_relation_bias_mode,
                max_agents=matrix_config.max_agents,
                representation=matrix_config.agent_relation_bias_representation,
                representation_dim=matrix_config.agent_relation_bias_dim,
                init_std=matrix_config.agent_relation_bias_init_std,
                same_init=matrix_config.agent_relation_bias_same_init,
                diff_init=matrix_config.agent_relation_bias_diff_init,
            )
            self.agent_relation_bias.install_hooks(base_model)

        if matrix_config.agent_dynamic_state_mode == "none":
            self.agent_dynamic_state = None
        else:
            self.agent_dynamic_state = DynamicAgentState(
                input_dim=hidden_size,
                state_dim=matrix_config.agent_dynamic_state_dim,
                hidden_size=hidden_size,
                output_init_std=matrix_config.agent_dynamic_state_init_std,
            )

        # Optional channel embeddings.
        if matrix_config.use_channel_embeddings:
            self.channel_embeddings = nn.Embedding(matrix_config.num_channels, hidden_size)
            nn.init.normal_(self.channel_embeddings.weight, mean=0.0, std=0.02)
        else:
            self.channel_embeddings = None

        # Learned representation for inactive (yielding) cells -- replaces
        # looking up the (frozen, meaningless) placeholder token embedding.
        self.inactive_embedding = nn.Embedding(1, hidden_size)
        nn.init.normal_(self.inactive_embedding.weight, mean=0.0, std=0.02)

        # Binary speak/yield head, applied to the final hidden state of every
        # cell (always present -- this is the turn-taking "action" head).
        self.activity_head = nn.Linear(hidden_size, 1)

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

    @staticmethod
    def unflatten_scalar(flat: torch.Tensor, num_agents: int, seq_len: int) -> torch.Tensor:
        """Unflatten a per-cell scalar ``[B, T*A]`` -> ``[B, A, T]`` (no vocab dim)."""
        b = flat.shape[0]
        return flat.reshape(b, seq_len, num_agents).permute(0, 2, 1)

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
        flat_activity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ``token_embeds + agent_embeds (+ channel_embeds)``.

        The agent/channel embedding tables are kept in fp32 (good for stable
        optimization) while the base model may run in bf16/fp16. Cast the added
        embeddings to the token-embedding dtype so the sum -- and therefore the
        tensor handed to the base model -- always matches the base model's dtype.

        ``flat_activity_mask`` (optional, ``True`` where a cell is active/speaking):
        when given, inactive cells use the learned ``inactive_embedding`` instead
        of the base model's embedding of ``flat_input_ids`` at that position (the
        placeholder id there is otherwise meaningless). ``None`` means "every
        cell is active" (no substitution) -- the default, fully backward
        compatible behavior for callers that don't model turn-taking.
        """
        token_embeds = self.base_model.get_input_embeddings()(flat_input_ids)

        if flat_activity_mask is not None:
            inactive = (~flat_activity_mask.bool()).unsqueeze(-1)
            inactive_vec = self.inactive_embedding.weight[0].to(token_embeds.dtype)
            token_embeds = torch.where(inactive, inactive_vec, token_embeds)

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

    def install_agent_attention_hooks(self) -> None:
        """Reattach Q/K/V hooks after PEFT replaces projection modules."""
        if self.agent_attention is not None:
            self.agent_attention.install_hooks(self.base_model)
        if self.agent_same_attention_bias is not None:
            self.agent_same_attention_bias.install_hooks(self.base_model)
        if self.agent_relation_bias is not None:
            self.agent_relation_bias.install_hooks(self.base_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,                        # [B, A, T]
        labels: Optional[torch.LongTensor] = None,           # [B, A, T], -100 ignored, content targets
        agent_ids: Optional[torch.LongTensor] = None,        # [B, A, T]
        channel_ids: Optional[torch.LongTensor] = None,      # [B, A, T]
        input_activity_mask: Optional[torch.Tensor] = None,  # [B, A, T] bool/long; None = all active
        activity_labels: Optional[torch.LongTensor] = None,  # [B, A, T] in {0,1,-100}; speak-next label
        input_private_mask: Optional[torch.Tensor] = None,   # [B, A, T] bool; True = private cell
        agent_visibility: Optional[torch.Tensor] = None,     # [B, A, A] bool; [owner, viewer]
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
        flat_activity_mask = (
            self.flatten_matrix(input_activity_mask) if input_activity_mask is not None else None
        )
        flat_private_mask = (
            self.flatten_matrix(input_private_mask) if input_private_mask is not None else None
        )

        inputs_embeds = self.get_input_embeddings_with_agents(
            flat_input_ids, flat_agent_ids, flat_channel_ids, flat_activity_mask
        )
        if self.agent_dynamic_state is not None:
            token_embeds_matrix = self.base_model.get_input_embeddings()(input_ids)  # [B, A, T, H]
            if input_activity_mask is not None:
                token_embeds_matrix = token_embeds_matrix * input_activity_mask.unsqueeze(-1).to(
                    token_embeds_matrix.dtype
                )
            dynamic_state = self.agent_dynamic_state(token_embeds_matrix)  # [B, A, T, H]
            dynamic_state_flat = dynamic_state.permute(0, 2, 1, 3).reshape(b, t * a, self.hidden_size)
            inputs_embeds = inputs_embeds + dynamic_state_flat.to(inputs_embeds.dtype)
        if self.agent_attention is not None:
            # This state intentionally remains set after forward: gradient
            # checkpointing recomputes decoder layers during backward, after
            # the base-model forward has returned.
            self.agent_attention.set_agent_ids(flat_agent_ids)
        if self.agent_same_attention_bias is not None:
            self.agent_same_attention_bias.set_agent_ids(flat_agent_ids)
        if self.agent_relation_bias is not None:
            self.agent_relation_bias.set_agent_ids(flat_agent_ids)

        position_ids = self.build_position_ids(b, a, t, device)    # [B, T*A]

        if attention_mask is None and self.matrix_config.attn_mask_mode == "matrix_causal":
            inactive_mask = (~flat_activity_mask.bool()) if flat_activity_mask is not None else None
            attention_mask = build_matrix_causal_mask(
                batch_size=b,
                num_agents=a,
                seq_len=t,
                device=device,
                dtype=inputs_embeds.dtype,
                allow_same_column=self.matrix_config.allow_same_column,
                inactive_mask=inactive_mask,
                private_mask=flat_private_mask,
                agent_visibility=agent_visibility,
            )
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        flat_logits = outputs.logits            # [B, T*A, V]
        flat_hidden = outputs.hidden_states[-1]  # [B, T*A, H] -- what feeds the vocab head
        v = flat_logits.shape[-1]

        logits = self.unflatten_logits(flat_logits, num_agents=a, seq_len=t)  # [B, A, T, V]

        # activity_head stays fp32 (like the other wrapper params, for stable
        # optimization) while the base model may run in bf16/fp16; cast the
        # (possibly lower-precision) hidden state up to the head's dtype so
        # F.linear never sees a dtype mismatch.
        flat_activity_logits = self.activity_head(
            flat_hidden.to(self.activity_head.weight.dtype)
        ).squeeze(-1)   # [B, T*A]
        activity_logits = self.unflatten_scalar(flat_activity_logits, num_agents=a, seq_len=t)  # [B, A, T]

        content_loss = None
        if labels is not None:
            # Custom CE over matrix-aligned labels (NOT the base model's shifted loss).
            # Only ever placed on cells where the next column speaks (see data/convert.py).
            flat_labels = self.flatten_matrix(labels)  # [B, T*A]
            if (flat_labels != -100).any():
                content_loss = F.cross_entropy(
                    flat_logits.reshape(-1, v),
                    flat_labels.reshape(-1),
                    ignore_index=-100,
                )
            else:
                # No valid content targets in this batch (e.g. the target's turns
                # are all single tokens, so no internal continuation exists to
                # supervise). F.cross_entropy on an all-ignored batch is NaN, so
                # define this case as a proper zero instead -- no content signal
                # this batch, not a corrupted one (a NaN would otherwise poison
                # any running average, e.g. across a validation set).
                content_loss = flat_logits.new_zeros(())

        activity_loss = None
        turn_reward_mean = None
        turn_reward_stats = None
        same_handoff_loss = None
        same_handoff_stats = None
        if activity_labels is not None:
            flat_activity_labels = self.flatten_matrix(activity_labels)  # [B, T*A]
            valid = flat_activity_labels != -100
            pos = valid & (flat_activity_labels == 1)
            neg = valid & (flat_activity_labels == 0)

            bce_all = F.binary_cross_entropy_with_logits(
                flat_activity_logits.float(),
                flat_activity_labels.clamp(min=0).float(),
                reduction="none",
            )
            pos_loss = bce_all[pos].mean() if pos.any() else None
            neg_loss = bce_all[neg].mean() if neg.any() else None
            w = self.matrix_config.activity_pos_weight
            if pos_loss is not None and neg_loss is not None:
                # w=0.5 (default) balances the speak-class and yield-class BCE
                # equally, so the (usually much larger) yield class does not
                # dominate and push the model toward an always-silent policy.
                # w>0.5 additionally biases the decision boundary toward
                # speaking, penalizing missed-speak mistakes harder.
                activity_loss = w * pos_loss + (1.0 - w) * neg_loss
            elif pos_loss is not None:
                activity_loss = pos_loss
            elif neg_loss is not None:
                activity_loss = neg_loss

            if input_activity_mask is not None:
                if self.matrix_config.turn_reward_mode == "balanced_ce":
                    floor_cfg = FloorControlConfig(
                        ref_silence=self.matrix_config.floor_control_ref_silence,
                        ref_same=self.matrix_config.floor_control_ref_same,
                        ref_handoff=self.matrix_config.floor_control_ref_handoff,
                        ref_overlap=self.matrix_config.floor_control_ref_overlap,
                        weight_silence=self.matrix_config.floor_control_weight_silence,
                        weight_same=self.matrix_config.floor_control_weight_same,
                        weight_handoff=self.matrix_config.floor_control_weight_handoff,
                        weight_overlap=self.matrix_config.floor_control_weight_overlap,
                        adaptive_weight_alpha=(
                            self.matrix_config.floor_control_adaptive_weight_alpha
                        ),
                        adaptive_prior_strength=(
                            self.matrix_config.floor_control_adaptive_prior_strength
                        ),
                    )
                    turn_reward_mean, turn_reward_stats = floor_control_reward(
                        activity_logits, input_activity_mask, cfg=floor_cfg
                    )
                elif self.matrix_config.turn_reward_mode == "linear":
                    reward_cfg = TurnRewardConfig(
                        speak_grace=self.matrix_config.speak_grace,
                        speak_tau=self.matrix_config.speak_tau,
                        silence_grace=self.matrix_config.silence_grace,
                        silence_tau=self.matrix_config.silence_tau,
                        overlap_base_weight=self.matrix_config.overlap_base_weight,
                        overlap_max_weight=self.matrix_config.overlap_max_weight,
                        overlap_grace=self.matrix_config.overlap_grace,
                        overlap_tau=self.matrix_config.overlap_tau,
                        handoff_bonus_weight=self.matrix_config.handoff_bonus_weight,
                    )
                    turn_reward_mean, turn_reward_stats = turn_taking_reward(
                        activity_logits, input_activity_mask, cfg=reward_cfg
                    )
                else:
                    raise ValueError(
                        f"Unknown turn_reward_mode {self.matrix_config.turn_reward_mode!r}"
                    )
                same_handoff_loss, same_handoff_stats = same_handoff_cross_entropy(
                    activity_logits,
                    input_activity_mask,
                    activity_labels,
                )

        loss = None
        if content_loss is not None or activity_loss is not None or turn_reward_mean is not None:
            loss = flat_logits.new_zeros(())
            if content_loss is not None:
                loss = loss + self.matrix_config.lambda_content * content_loss
            if activity_loss is not None:
                loss = loss + self.matrix_config.lambda_activity * activity_loss
            if turn_reward_mean is not None:
                loss = loss - self.matrix_config.lambda_reward * turn_reward_mean
            if same_handoff_loss is not None:
                loss = loss + self.matrix_config.lambda_same_handoff * same_handoff_loss

        return MatrixCausalLMOutput(
            loss=loss,
            logits=logits,
            flat_logits=flat_logits,
            position_ids=position_ids,
            flat_input_ids=flat_input_ids,
            activity_logits=activity_logits,
            content_loss=content_loss,
            activity_loss=activity_loss,
            turn_reward=turn_reward_mean,
            turn_reward_stats=turn_reward_stats,
            same_handoff_loss=same_handoff_loss,
            same_handoff_stats=same_handoff_stats,
        )

    # ------------------------------------------------------------------
    # Inference-only incremental generation (KV cache)
    # ------------------------------------------------------------------
    def build_column_position_ids(
        self,
        batch_size: int,
        num_agents: int,
        column_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Flat position ids for one matrix column ``[B, A]``."""
        mode = self.matrix_config.position_mode
        if mode == "flat":
            start = column_index * num_agents
            pos = torch.arange(start, start + num_agents, device=device)
        elif mode == "column":
            pos = torch.full((num_agents,), column_index, device=device)
        else:
            raise ValueError(f"Unknown position_mode: {mode!r}")
        return pos.unsqueeze(0).expand(batch_size, num_agents)

    def _column_token_embeds(
        self,
        column_ids: torch.LongTensor,
        column_activity: torch.Tensor,
        agent_ids: torch.LongTensor,
        channel_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Build flat ``[B, A, H]`` embeddings for one matrix column."""
        flat_ids = column_ids.reshape(column_ids.shape[0], -1)
        flat_activity = column_activity.reshape(column_activity.shape[0], -1)
        flat_agent_ids = agent_ids.reshape(agent_ids.shape[0], -1)
        flat_channel_ids = (
            channel_ids.reshape(channel_ids.shape[0], -1) if channel_ids is not None else None
        )
        flat_embeds = self.get_input_embeddings_with_agents(
            flat_ids, flat_agent_ids, flat_channel_ids, flat_activity
        )
        return flat_embeds

    def _append_dynamic_state_column(
        self,
        column_ids: torch.LongTensor,
        column_activity: torch.Tensor,
        gru_hidden: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return dynamic-state contribution ``[B, A, H]`` for one column."""
        if self.agent_dynamic_state is None:
            return column_ids.new_zeros(
                column_ids.shape[0], column_ids.shape[1], self.hidden_size
            ), gru_hidden
        token_embeds = self.base_model.get_input_embeddings()(column_ids)
        token_embeds = token_embeds * column_activity.unsqueeze(-1).to(token_embeds.dtype)
        dynamic, gru_hidden = self.agent_dynamic_state.forward_step(
            token_embeds.reshape(column_ids.shape[0], column_ids.shape[1], -1),
            gru_hidden,
        )
        return dynamic.to(token_embeds.dtype), gru_hidden

    def _run_generation_forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        flat_agent_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        if self.agent_attention is not None:
            self.agent_attention.set_agent_ids(flat_agent_ids)
        if self.agent_same_attention_bias is not None:
            self.agent_same_attention_bias.set_agent_ids(flat_agent_ids)
        if self.agent_relation_bias is not None:
            self.agent_relation_bias.set_agent_ids(flat_agent_ids)

        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )
        flat_logits = outputs.logits
        flat_hidden = outputs.hidden_states[-1]
        flat_activity_logits = self.activity_head(
            flat_hidden.to(self.activity_head.weight.dtype)
        ).squeeze(-1)
        return flat_logits, flat_activity_logits, outputs.past_key_values

    @torch.no_grad()
    def encode_prefix_for_generation(
        self,
        input_ids: torch.LongTensor,
        input_activity_mask: Optional[torch.Tensor] = None,
        input_private_mask: Optional[torch.Tensor] = None,
        agent_visibility: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.LongTensor] = None,
        channel_ids: Optional[torch.LongTensor] = None,
    ) -> tuple[MatrixGenerationCache, torch.Tensor, torch.Tensor]:
        """Encode a teacher-forced prefix and return cache plus last-column logits."""
        if input_ids.dim() != 3:
            raise ValueError(f"input_ids must be [B, A, T], got shape {tuple(input_ids.shape)}")

        b, a, t = input_ids.shape
        device = input_ids.device
        if agent_ids is None:
            agent_ids = self._default_agent_ids(b, a, t, device)

        flat_input_ids = self.flatten_matrix(input_ids)
        flat_agent_ids = self.flatten_matrix(agent_ids)
        flat_channel_ids = self.flatten_matrix(channel_ids) if channel_ids is not None else None
        flat_activity_mask = (
            self.flatten_matrix(input_activity_mask) if input_activity_mask is not None else None
        )
        flat_private_mask = (
            self.flatten_matrix(input_private_mask) if input_private_mask is not None else None
        )

        inputs_embeds = self.get_input_embeddings_with_agents(
            flat_input_ids, flat_agent_ids, flat_channel_ids, flat_activity_mask
        )
        gru_hidden = None
        if self.agent_dynamic_state is not None:
            token_embeds_matrix = self.base_model.get_input_embeddings()(input_ids)
            if input_activity_mask is not None:
                token_embeds_matrix = token_embeds_matrix * input_activity_mask.unsqueeze(-1).to(
                    token_embeds_matrix.dtype
                )
            dynamic_state, gru_hidden = self.agent_dynamic_state.forward_with_state(
                token_embeds_matrix
            )
            dynamic_state_flat = dynamic_state.permute(0, 2, 1, 3).reshape(b, t * a, self.hidden_size)
            inputs_embeds = inputs_embeds + dynamic_state_flat.to(inputs_embeds.dtype)

        position_ids = self.build_position_ids(b, a, t, device)
        attention_mask = None
        flat_inactive_mask = None
        if self.matrix_config.attn_mask_mode == "matrix_causal":
            flat_inactive_mask = (
                (~flat_activity_mask.bool()) if flat_activity_mask is not None else None
            )
            attention_mask = build_matrix_causal_mask(
                batch_size=b,
                num_agents=a,
                seq_len=t,
                device=device,
                dtype=inputs_embeds.dtype,
                allow_same_column=self.matrix_config.allow_same_column,
                inactive_mask=flat_inactive_mask,
                private_mask=flat_private_mask,
                agent_visibility=agent_visibility,
            )

        flat_logits, flat_activity_logits, past_key_values = self._run_generation_forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            flat_agent_ids=flat_agent_ids,
            past_key_values=None,
            use_cache=True,
        )
        content_logits = self.unflatten_logits(flat_logits, num_agents=a, seq_len=t)[:, :, -1, :]
        activity_logits = self.unflatten_scalar(flat_activity_logits, num_agents=a, seq_len=t)[:, :, -1]

        cache = MatrixGenerationCache(
            past_key_values=past_key_values,
            batch_size=b,
            num_agents=a,
            seq_len=t,
            agent_visibility=agent_visibility,
            flat_inactive_mask=flat_inactive_mask,
            flat_private_mask=flat_private_mask,
            gru_hidden=gru_hidden,
            embed_dtype=inputs_embeds.dtype,
            mask_dtype=inputs_embeds.dtype,
            device=device,
        )
        return cache, content_logits, activity_logits

    @torch.no_grad()
    def forward_generation_column(
        self,
        cache: MatrixGenerationCache,
        column_ids: torch.LongTensor,
        column_activity: torch.Tensor,
        column_private: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.LongTensor] = None,
        channel_ids: Optional[torch.LongTensor] = None,
    ) -> tuple[MatrixGenerationCache, torch.Tensor, torch.Tensor]:
        """Append one generated column and return logits at that column."""
        if column_ids.dim() != 3 or column_ids.shape[2] != 1:
            raise ValueError("column_ids must be [B, A, 1]")
        b, a, _ = column_ids.shape
        if b != cache.batch_size or a != cache.num_agents:
            raise ValueError("column_ids shape does not match cache topology")

        device = cache.device
        if agent_ids is None:
            agent_ids = self._default_agent_ids(b, a, 1, device)

        column_ids = column_ids.to(device)
        column_activity = column_activity.to(device).bool()
        column_private = (
            column_private.to(device).bool()
            if column_private is not None
            else torch.zeros_like(column_activity, dtype=torch.bool)
        )

        flat_ids = column_ids.squeeze(-1)
        flat_activity = column_activity.squeeze(-1)
        flat_agent_ids = agent_ids.squeeze(-1)
        flat_channel_ids = channel_ids.squeeze(-1) if channel_ids is not None else None

        inputs_embeds = self._column_token_embeds(
            flat_ids.unsqueeze(-1),
            flat_activity.unsqueeze(-1),
            flat_agent_ids.unsqueeze(-1),
            flat_channel_ids.unsqueeze(-1) if flat_channel_ids is not None else None,
        )
        dynamic, gru_hidden = self._append_dynamic_state_column(
            flat_ids.unsqueeze(-1), flat_activity.unsqueeze(-1), cache.gru_hidden
        )
        inputs_embeds = inputs_embeds + dynamic.to(inputs_embeds.dtype)

        position_ids = self.build_column_position_ids(b, a, cache.seq_len, device)
        flat_inactive = ~flat_activity
        if cache.flat_inactive_mask is None:
            cache.flat_inactive_mask = flat_inactive
        else:
            cache.flat_inactive_mask = torch.cat([cache.flat_inactive_mask, flat_inactive], dim=1)
        flat_private = column_private.squeeze(-1)
        if cache.flat_private_mask is None:
            cache.flat_private_mask = flat_private
        else:
            cache.flat_private_mask = torch.cat([cache.flat_private_mask, flat_private], dim=1)

        new_seq_len = cache.seq_len + 1
        attention_mask = None
        if self.matrix_config.attn_mask_mode == "matrix_causal":
            attention_mask = build_matrix_causal_mask_new_column_queries(
                batch_size=b,
                num_agents=a,
                seq_len=new_seq_len,
                device=device,
                dtype=cache.mask_dtype,
                allow_same_column=self.matrix_config.allow_same_column,
                inactive_mask=cache.flat_inactive_mask,
                private_mask=cache.flat_private_mask,
                agent_visibility=cache.agent_visibility,
            )

        flat_logits, flat_activity_logits, past_key_values = self._run_generation_forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            flat_agent_ids=flat_agent_ids,
            past_key_values=cache.past_key_values,
            use_cache=True,
        )
        cache.past_key_values = past_key_values
        cache.gru_hidden = gru_hidden
        cache.seq_len = new_seq_len
        return cache, flat_logits, flat_activity_logits
