"""Two-head residual critic and per-role advantage normalizer (plan sections 5-6)."""

from __future__ import annotations

import torch
import torch.nn as nn


def default_tap_layers(num_layers):
    """Quartile layers of the base model, e.g. (9, 18, 27, 36) for Qwen3-4B's 36 layers.

    Index i means ``hidden_states[i]`` (0 = embeddings, num_layers = last layer).
    """
    return tuple(sorted({max(1, round(num_layers * k / 4)) for k in (1, 2, 3, 4)}))


def game_feature_dim(max_agents):
    """round_norm, alive_ratio, phase one-hot (4), alive mask (max_agents)."""
    return 2 + 4 + max_agents


class MultiLayerResidualCritic(nn.Module):
    """Lean residual critic with two value heads, ``V_round`` and ``V_game``.

    Takes detached Qwen hidden states from ``tap_layers`` at each turn's
    pre-turn cell (already gathered, ``[U, L, H]``) plus public game-state
    features ``[U, game_feature_dim]``; nothing flows back into the policy.
    """

    def __init__(self, hidden_size, num_taps, max_agents=8, residual_dim=224, game_feat_dim=32,
                 trunk_dim=256, dropout=0.1):
        super().__init__()
        self.layer_norms = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(num_taps)])
        self.layer_projs = nn.ModuleList([nn.Linear(hidden_size, residual_dim, bias=False)
                                          for _ in range(num_taps)])
        self.layer_gates = nn.Parameter(torch.zeros(num_taps))
        self.game_encoder = nn.Sequential(
            nn.Linear(game_feature_dim(max_agents), game_feat_dim), nn.SiLU(),
            nn.Linear(game_feat_dim, game_feat_dim), nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(residual_dim + game_feat_dim, trunk_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(trunk_dim, trunk_dim), nn.SiLU(),
        )
        self.v_round_head = nn.Linear(trunk_dim, 1)
        self.v_game_head = nn.Linear(trunk_dim, 1)

    def forward(self, hidden, game_features):
        """``hidden`` ``[U, L, H]``, ``game_features`` ``[U, G]`` -> ``(v_round, v_game)``, each ``[U]``."""
        dtype = self.layer_projs[0].weight.dtype
        weights = torch.softmax(self.layer_gates, dim=0)
        fused = 0.0
        for i, (norm, proj) in enumerate(zip(self.layer_norms, self.layer_projs)):
            fused = fused + weights[i] * proj(norm(hidden[:, i].detach().to(dtype)))
        z = self.trunk(torch.cat([fused, self.game_encoder(game_features.to(dtype))], dim=-1))
        return self.v_round_head(z).squeeze(-1), self.v_game_head(z).squeeze(-1)


class RoleConditionedEMA(nn.Module):
    """Sample-weighted, bias-corrected per-role EMA normalizer for (A_round, A_game).

    Update once per rollout step, on the whole step's batch (all ranks), before
    the PPO epochs. Each role's advantages are standardized with that role's
    running mean and variance, so every role contributes unit-variance signal.
    """

    def __init__(self, num_roles=6, num_streams=2, ema_decay=0.95, ref_count=16.0, eps=1e-5):
        super().__init__()
        self.num_roles, self.num_streams = num_roles, num_streams
        self.ema_decay, self.ref_count, self.eps = ema_decay, ref_count, eps
        self.register_buffer("raw_mean", torch.zeros(num_roles, num_streams))
        self.register_buffer("raw_var", torch.zeros(num_roles, num_streams))
        self.register_buffer("decay_prod", torch.ones(num_roles, num_streams))

    @torch.no_grad()
    def normalize_and_update(self, advantages, role_ids):
        """``advantages`` ``[N, S]``, ``role_ids`` ``[N]`` -> normalized ``[N, S]``."""
        out = torch.zeros_like(advantages)
        for r in range(self.num_roles):
            mask = role_ids == r
            n_r = int(mask.sum())
            if n_r == 0:
                continue
            beta = 1.0 - self.ema_decay ** (n_r / self.ref_count)
            for s in range(self.num_streams):
                vals = advantages[mask, s]
                batch_mean = vals.mean()
                batch_var = vals.var(unbiased=False) if n_r > 1 else vals.new_ones(())
                if float(self.decay_prod[r, s]) < 1.0 - 1e-7:
                    denom = 1.0 - self.decay_prod[r, s]
                    mean_r = self.raw_mean[r, s] / denom
                    var_r = (self.raw_var[r, s] / denom).clamp(min=self.eps)
                else:  # first batch for this role: use the batch's own statistics
                    mean_r, var_r = batch_mean, batch_var.clamp(min=self.eps)
                out[mask, s] = (vals - mean_r) / torch.sqrt(var_r + self.eps)
                self.raw_mean[r, s] = (1 - beta) * self.raw_mean[r, s] + beta * batch_mean
                self.raw_var[r, s] = (1 - beta) * self.raw_var[r, s] + beta * batch_var
                self.decay_prod[r, s] = self.decay_prod[r, s] * (1 - beta)
        return out
