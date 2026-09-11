"""Attention-level agent conditioning for MatrixChat.

The base Qwen projections stay untouched (important for PEFT/LoRA
compatibility).  Forward hooks add a learned agent-dependent residual to the
outputs of q_proj, k_proj, and v_proj immediately before Qwen's Q/K
normalization and RoPE application.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder layer list through plain and PEFT-wrapped layouts."""
    candidates = (
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.base_model.model.base_model.model.model.layers,
        lambda m: m.layers,
    )
    for getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if isinstance(layers, nn.ModuleList) and layers:
            return layers
    current = model
    for _ in range(8):
        if hasattr(current, "layers"):
            layers = getattr(current, "layers")
            if isinstance(layers, nn.ModuleList) and layers:
                return layers
        if hasattr(current, "model"):
            current = getattr(current, "model")
            continue
        if hasattr(current, "base_model"):
            current = getattr(current, "base_model")
            continue
        break
    raise RuntimeError(
        "Could not locate decoder layers for agent attention conditioning. "
        "Checked common Qwen/Hugging Face/PEFT module layouts."
    )


class AgentQKVLayerAdapter(nn.Module):
    """Project one shared agent representation into one layer's Q/K/V spaces."""

    def __init__(
        self,
        representation_dim: int,
        hidden_size: int,
        q_size: int,
        k_size: int,
        v_size: int,
        *,
        gated: bool,
        init_std: float,
    ):
        super().__init__()
        self.q = nn.Linear(representation_dim, q_size, bias=False)
        self.k = nn.Linear(representation_dim, k_size, bias=False)
        self.v = nn.Linear(representation_dim, v_size, bias=False)
        self.gate = nn.Linear(hidden_size, 1) if gated else None

        # Start as a very small perturbation of the pretrained model while
        # retaining gradients for both the representation and projections.
        for projection in (self.q, self.k, self.v):
            nn.init.normal_(projection.weight, mean=0.0, std=init_std)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)


class AgentQKVConditioner(nn.Module):
    """Inject persistent row identity directly into every attention Q/K/V.

    ``mode="qkv"`` adds the projected identity with a fixed strength.
    ``mode="qkv_gated"`` additionally learns a content-dependent scalar gate
    at every layer and token, allowing the model to decide when identity is
    useful.  Agent representations can either be learned vectors or fixed
    centered one-hot (regular-simplex) codes.
    """

    VALID_MODES = {"qkv", "qkv_gated"}
    VALID_REPRESENTATIONS = {"learned", "simplex"}

    def __init__(
        self,
        base_model: nn.Module,
        *,
        max_agents: int,
        mode: str,
        representation: str,
        representation_dim: int,
        init_std: float,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown agent attention mode {mode!r}")
        if representation not in self.VALID_REPRESENTATIONS:
            raise ValueError(f"Unknown agent attention representation {representation!r}")
        if max_agents < 1:
            raise ValueError("max_agents must be positive")
        if representation == "learned" and representation_dim < 1:
            raise ValueError("representation_dim must be positive for learned representations")
        if init_std < 0:
            raise ValueError("agent attention init_std must be non-negative")

        self.mode = mode
        self.representation_kind = representation
        self.max_agents = max_agents
        self._flat_agent_ids: Optional[torch.Tensor] = None
        self._hook_handles: list = []

        if representation == "learned":
            self.agent_representations = nn.Embedding(max_agents, representation_dim)
            nn.init.normal_(self.agent_representations.weight, mean=0.0, std=0.02)
            rep_dim = representation_dim
            self.register_buffer("_simplex_codes", None, persistent=False)
        else:
            self.agent_representations = None
            codes = torch.eye(max_agents) - torch.full((max_agents, max_agents), 1.0 / max_agents)
            if max_agents > 1:
                codes = F.normalize(codes, dim=-1)
            self.register_buffer("_simplex_codes", codes, persistent=True)
            rep_dim = max_agents

        layers = find_decoder_layers(base_model)
        adapters = []
        for layer in layers:
            attention = layer.self_attn
            adapters.append(
                AgentQKVLayerAdapter(
                    rep_dim,
                    attention.q_proj.in_features,
                    attention.q_proj.out_features,
                    attention.k_proj.out_features,
                    attention.v_proj.out_features,
                    gated=(mode == "qkv_gated"),
                    init_std=init_std,
                )
            )
        self.layers = nn.ModuleList(adapters)

    def set_agent_ids(self, flat_agent_ids: torch.Tensor) -> None:
        """Set the ``[B, sequence]`` identities used by subsequent hooks."""
        self._flat_agent_ids = flat_agent_ids

    def _representations(self, agent_ids: torch.Tensor) -> torch.Tensor:
        if self.agent_representations is not None:
            return self.agent_representations(agent_ids)
        return self._simplex_codes[agent_ids]

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def install_hooks(self, base_model: nn.Module) -> None:
        """(Re)attach hooks, e.g. after PEFT replaces Q/K/V projection modules."""
        self.remove_hooks()
        layers = find_decoder_layers(base_model)
        if len(layers) != len(self.layers):
            raise RuntimeError(
                f"Agent conditioner has {len(self.layers)} layer adapters but "
                f"the base model exposes {len(layers)} decoder layers"
            )

        for layer_index, layer in enumerate(layers):
            attention = layer.self_attn
            adapter = self.layers[layer_index]
            for kind in ("q", "k", "v"):
                projection = getattr(attention, f"{kind}_proj")
                handle = projection.register_forward_hook(
                    self._make_projection_hook(adapter, kind)
                )
                self._hook_handles.append(handle)

    def _make_projection_hook(self, adapter: AgentQKVLayerAdapter, kind: str):
        def hook(_module, inputs, output):
            agent_ids = self._flat_agent_ids
            if agent_ids is None:
                return output
            if output.shape[:2] != agent_ids.shape:
                raise RuntimeError(
                    "Agent-conditioning ids/output shape mismatch: "
                    f"ids={tuple(agent_ids.shape)}, projection={tuple(output.shape)}"
                )

            representation = self._representations(agent_ids)
            delta = getattr(adapter, kind)(representation)
            if adapter.gate is not None:
                hidden_states = inputs[0]
                gate = torch.sigmoid(adapter.gate(hidden_states.float()))
                delta = delta * gate
            return output + delta.to(dtype=output.dtype)

        return hook


class AgentSameAttentionBias(nn.Module):
    """Learn a same-agent attention-logit bias independently at each layer.

    Bias construction happens inside the attention module's checkpointed
    forward call.  This is required for re-entrant gradient checkpointing:
    constructing one parameter-dependent mask outside the decoder and reusing
    it across every layer would make autograd traverse the same graph multiple
    times during backward.
    """

    def __init__(self, base_model: nn.Module, *, init: float = 0.0):
        super().__init__()
        layers = find_decoder_layers(base_model)
        self.layer_bias = nn.Parameter(torch.full((len(layers),), float(init)))
        self._flat_agent_ids: Optional[torch.Tensor] = None
        self._hook_handles: list = []

    def set_agent_ids(self, flat_agent_ids: torch.Tensor) -> None:
        self._flat_agent_ids = flat_agent_ids

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def install_hooks(self, base_model: nn.Module) -> None:
        self.remove_hooks()
        layers = find_decoder_layers(base_model)
        if len(layers) != len(self.layer_bias):
            raise RuntimeError(
                f"Relation bias has {len(self.layer_bias)} values but "
                f"the base model exposes {len(layers)} decoder layers"
            )
        for layer_index, layer in enumerate(layers):
            handle = layer.self_attn.register_forward_pre_hook(
                self._make_attention_hook(layer_index),
                with_kwargs=True,
            )
            self._hook_handles.append(handle)

    def _make_attention_hook(self, layer_index: int):
        def hook(_module, args, kwargs):
            agent_ids = self._flat_agent_ids
            attention_mask = kwargs.get("attention_mask")
            if agent_ids is None or attention_mask is None:
                return args, kwargs
            if attention_mask.ndim != 4:
                raise RuntimeError(
                    "Same-agent relation bias requires a 4D additive attention mask, "
                    f"got shape {tuple(attention_mask.shape)}"
                )
            if agent_ids.shape[-1] != attention_mask.shape[-1]:
                return args, kwargs
            same_agent = agent_ids.unsqueeze(2) == agent_ids.unsqueeze(1)
            relation = same_agent.unsqueeze(1).to(dtype=attention_mask.dtype)
            kwargs["attention_mask"] = (
                attention_mask
                + relation * self.layer_bias[layer_index].to(attention_mask.dtype)
            )
            return args, kwargs

        return hook


class AgentRelationBias(nn.Module):
    """Generalized pairwise agent-relation bias in the attention logits.

    Generalizes :class:`AgentSameAttentionBias`'s single same-agent scalar
    into either:

    - ``mode="same_diff"``: independent learned same-agent and
      different-agent scalars per layer (still permutation-equivariant,
      just no longer forcing the different-agent case to a neutral 0).
    - ``mode="bilinear"``: a learned bilinear attention-logit bias computed
      from per-agent representations, loosely inspired by AgentFormer's
      (Yuan et al., ICCV 2021) same-/cross-agent query-key branching. This
      is implemented as an ADDITIVE logit correction through the same
      checkpoint-safe ``attention_mask`` pre-hook used by
      :class:`AgentSameAttentionBias`, rather than by replacing the base
      Q/K projections outright, so it stays compatible with LoRA and
      re-entrant gradient checkpointing without touching q_proj/k_proj.

    Kept as a separate class (not a repurposing of ``AgentSameAttentionBias``)
    so existing ``agent_same_attention_bias`` checkpoints/configs are
    completely unaffected.
    """

    VALID_MODES = {"same_diff", "bilinear"}
    VALID_REPRESENTATIONS = {"learned", "simplex"}

    def __init__(
        self,
        base_model: nn.Module,
        *,
        mode: str,
        max_agents: int = 8,
        representation: str = "learned",
        representation_dim: int = 8,
        init_std: float = 1e-3,
        same_init: float = 0.0,
        diff_init: float = 0.0,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown agent relation bias mode {mode!r}")
        if representation not in self.VALID_REPRESENTATIONS:
            raise ValueError(f"Unknown agent relation bias representation {representation!r}")
        if max_agents < 1:
            raise ValueError("max_agents must be positive")

        layers = find_decoder_layers(base_model)
        num_layers = len(layers)
        self.mode = mode
        self.max_agents = max_agents
        self._flat_agent_ids: Optional[torch.Tensor] = None
        self._hook_handles: list = []

        self.same_bias = None
        self.diff_bias = None
        self.agent_representations = None
        self.query_projections = None
        self.key_projections = None
        self.register_buffer("_simplex_codes", None, persistent=False)

        if mode == "same_diff":
            self.same_bias = nn.Parameter(torch.full((num_layers,), float(same_init)))
            self.diff_bias = nn.Parameter(torch.full((num_layers,), float(diff_init)))
            return

        if representation_dim < 1:
            raise ValueError("representation_dim must be positive for bilinear relation bias")
        if representation == "learned":
            self.agent_representations = nn.Embedding(max_agents, representation_dim)
            nn.init.normal_(self.agent_representations.weight, mean=0.0, std=0.02)
            rep_dim = representation_dim
        else:
            codes = torch.eye(max_agents) - torch.full((max_agents, max_agents), 1.0 / max_agents)
            if max_agents > 1:
                codes = F.normalize(codes, dim=-1)
            self.register_buffer("_simplex_codes", codes, persistent=True)
            rep_dim = max_agents
        self._rep_dim = rep_dim

        self.query_projections = nn.ModuleList(
            [nn.Linear(rep_dim, rep_dim, bias=False) for _ in range(num_layers)]
        )
        self.key_projections = nn.ModuleList(
            [nn.Linear(rep_dim, rep_dim, bias=False) for _ in range(num_layers)]
        )
        for projection in list(self.query_projections) + list(self.key_projections):
            nn.init.normal_(projection.weight, mean=0.0, std=init_std)

    def set_agent_ids(self, flat_agent_ids: torch.Tensor) -> None:
        self._flat_agent_ids = flat_agent_ids

    def _representations(self, agent_ids: torch.Tensor) -> torch.Tensor:
        if self.agent_representations is not None:
            return self.agent_representations(agent_ids)
        return self._simplex_codes[agent_ids]

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def install_hooks(self, base_model: nn.Module) -> None:
        self.remove_hooks()
        layers = find_decoder_layers(base_model)
        expected = len(self.same_bias) if self.mode == "same_diff" else len(self.query_projections)
        if len(layers) != expected:
            raise RuntimeError(
                f"Agent relation bias has {expected} layer parameters but "
                f"the base model exposes {len(layers)} decoder layers"
            )
        for layer_index, layer in enumerate(layers):
            handle = layer.self_attn.register_forward_pre_hook(
                self._make_attention_hook(layer_index),
                with_kwargs=True,
            )
            self._hook_handles.append(handle)

    def _make_attention_hook(self, layer_index: int):
        def hook(_module, args, kwargs):
            agent_ids = self._flat_agent_ids
            attention_mask = kwargs.get("attention_mask")
            if agent_ids is None or attention_mask is None:
                return args, kwargs
            if attention_mask.ndim != 4:
                raise RuntimeError(
                    "Agent relation bias requires a 4D additive attention mask, "
                    f"got shape {tuple(attention_mask.shape)}"
                )
            if agent_ids.shape[-1] != attention_mask.shape[-1]:
                return args, kwargs

            if self.mode == "same_diff":
                same_agent = agent_ids.unsqueeze(2) == agent_ids.unsqueeze(1)
                same_agent = same_agent.unsqueeze(1).to(dtype=attention_mask.dtype)
                same = self.same_bias[layer_index].to(attention_mask.dtype)
                diff = self.diff_bias[layer_index].to(attention_mask.dtype)
                bias = same_agent * same + (1.0 - same_agent) * diff
            else:
                representation = self._representations(agent_ids)
                q_id = self.query_projections[layer_index](representation)
                k_id = self.key_projections[layer_index](representation)
                scale = self._rep_dim**0.5
                bias = torch.matmul(q_id, k_id.transpose(-1, -2)) / scale
                bias = bias.unsqueeze(1).to(dtype=attention_mask.dtype)

            kwargs["attention_mask"] = attention_mask + bias
            return args, kwargs

        return hook
