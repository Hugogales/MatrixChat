"""Recorded Werewolf games -> PPO / KL / Pillar-1 losses for MatrixQwen.

Pipeline for one game (see ``scripts/training/train_werewolf_rl.py``):

1. :func:`prepare_game` turns the recorded game into a :class:`GameBatch`
   (matrix tensors, every action, floor states, text-penalty targets, turn
   rewards and critic game-state features).
2. :func:`evaluate_game` runs one teacher-forced ``forward_selected`` pass and
   returns the log-prob of every recorded action under the *current*
   weights, the policy's speak probabilities, the text-penalty terms and
   (optionally) critic features at each turn's pre-turn cell.
3. :func:`game_losses` combines those with advantages into the actor loss
   ``lambda_conv * L_conv + lambda_ppo * L_ppo_clip + beta_kl * KL``.

Action log-probs are factored (plan section 2.1): each speak decision, each
word token and each name choice is its own PPO action sharing its turn's
advantage, except day-vote choices, which use the plan's exact baseline
``S_vote(sampled) - sum_v p_old(v) S_vote(v)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .critic import game_feature_dim
from .game import PHASE_IDS, ROLE_IDS, TURN_PHASE
from .rewards import RewardConfig, floor_states, text_penalty_targets, turn_step_rewards, vote_score


@dataclass
class GameBatch:
    """One recorded game as tensors (CPU until :meth:`to`)."""

    input_ids: torch.Tensor          # [1, A, T]
    activity: torch.Tensor           # [1, A, T] bool
    private: torch.Tensor            # [1, A, T] bool
    visibility: torch.Tensor | None  # [1, A, A] bool
    logit_positions: torch.Tensor    # [K] flat positions whose vocab logits are needed
    # Word-token actions.
    tok_row: torch.Tensor            # [N] row into the selected logits
    tok_id: torch.Tensor             # [N]
    tok_logp_rollout: torch.Tensor   # [N]
    tok_turn: torch.Tensor           # [N] global turn index
    tok_pen: torch.Tensor            # [2, P] (action, token id) repetition-penalized pairs
    tok_ban: torch.Tensor            # [2, Q] (action, token id) banned pairs
    tok_rep: torch.Tensor            # [2, R] (action, token id) self-repetition targets
    tok_echo: torch.Tensor           # [2, E] (action, token id) echo targets
    # Speak decisions.
    spk_agent: torch.Tensor          # [M]
    spk_src: torch.Tensor            # [M]
    spk_act: torch.Tensor            # [M] 0/1
    spk_bias: torch.Tensor           # [M] silence-ramp bias added to the activity logit
    spk_logp_rollout: torch.Tensor   # [M]
    spk_turn: torch.Tensor           # [M]
    spk_group: torch.Tensor          # [M] id of the (src) column group, for p_other
    spk_state: dict                  # floor-state tensors, each [M]
    # Name choices.
    ch_row: torch.Tensor             # [C]
    ch_cands: torch.Tensor           # [C, Kmax] candidate token ids (0-padded)
    ch_mask: torch.Tensor            # [C, Kmax] bool
    ch_chosen: torch.Tensor          # [C]
    ch_logp_rollout: torch.Tensor    # [C]
    ch_turn: torch.Tensor            # [C]
    ch_is_vote: torch.Tensor         # [C] bool: day vote (exact-baseline advantage)
    ch_svote: torch.Tensor           # [C, Kmax] S_vote of each candidate (day votes)
    # Turns (the MDP steps).
    turn_agent: torch.Tensor         # [U]
    turn_role: torch.Tensor          # [U]
    turn_hidden_pos: torch.Tensor    # [U] flat position of the pre-turn cell
    turn_features: torch.Tensor      # [U, G]
    turn_r_round: torch.Tensor       # [U]
    turn_r_game: torch.Tensor        # [U]
    agent_turns: list = field(default_factory=list)  # per agent: its turn indices in order
    info: dict = field(default_factory=dict)

    def to(self, device):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        self.spk_state = {k: v.to(device) for k, v in self.spk_state.items()}
        return self


def _pairs(lists):
    rows = [i for i, ids in enumerate(lists) for _ in ids]
    cols = [t for ids in lists for t in ids]
    return torch.tensor([rows, cols], dtype=torch.long).view(2, -1)


def prepare_game(result, tokenizer_exempt, reward_cfg: RewardConfig, max_agents, max_rounds):
    """Turn ``play_game(..., record=True)`` output into a :class:`GameBatch`."""
    ro = result["rollout"]
    events = result["events"]
    roles = events["roles"]
    input_ids, activity, private = ro["input_ids"], ro["activity"], ro["private"]
    num_agents = input_ids.shape[1]
    turns, tokens, speaks, choices = ro["turns"], ro["tokens"], ro["speaks"], ro["choices"]
    tokens = [t for t in tokens if t["turn"] >= 0]
    speaks = [s for s in speaks if s["turn"] >= 0]

    def flat(agent, col):
        return col * num_agents + agent

    positions = sorted({flat(t["agent"], t["src"]) for t in tokens} | {flat(c["agent"], c["src"]) for c in choices})
    if not positions:
        positions = [0]
    row_of = {p: i for i, p in enumerate(positions)}

    self_rep, echo = text_penalty_targets(input_ids[0], activity[0], tokens, tokenizer_exempt, reward_cfg)
    states = floor_states(activity[0], events["discussion_spans"], speaks, reward_cfg)
    groups = {}
    spk_group = [groups.setdefault(s["src"], len(groups)) for s in speaks]

    # Day-vote S_vote for every candidate (the exact zero-variance baseline).
    kmax = max([len(c["cand_tokens"]) for c in choices] + [1])
    ch_cands = torch.zeros(len(choices), kmax, dtype=torch.long)
    ch_mask = torch.zeros(len(choices), kmax, dtype=torch.bool)
    ch_svote = torch.zeros(len(choices), kmax)
    for i, c in enumerate(choices):
        k = len(c["cand_tokens"])
        ch_cands[i, :k] = torch.tensor(c["cand_tokens"])
        ch_mask[i, :k] = True
        if c["kind"] == "ballot":
            ch_svote[i, :k] = torch.tensor([vote_score(roles, c["agent"], s) for s in c["cand_seats"]])

    r_round, r_game = turn_step_rewards(turns, events, reward_cfg)
    feat = torch.zeros(len(turns), game_feature_dim(max_agents))
    for i, turn in enumerate(turns):
        feat[i, 0] = turn["day"] / max_rounds
        feat[i, 1] = len(turn["alive"]) / num_agents
        feat[i, 2 + PHASE_IDS[TURN_PHASE[turn["kind"]]]] = 1.0
        for seat in turn["alive"]:
            feat[i, 6 + seat] = 1.0
    agent_turns = [[] for _ in range(num_agents)]
    for i in sorted(range(len(turns)), key=lambda j: turns[j]["start"]):
        agent_turns[turns[i]["agent"]].append(i)

    lt = lambda xs, dtype=torch.long: torch.tensor(xs, dtype=dtype)  # noqa: E731
    vis = ro["visibility"]
    return GameBatch(
        input_ids=input_ids, activity=activity.bool(), private=private.bool(),
        visibility=vis.bool() if vis is not None else None,
        logit_positions=lt(positions),
        tok_row=lt([row_of[flat(t["agent"], t["src"])] for t in tokens]),
        tok_id=lt([t["token"] for t in tokens]),
        tok_logp_rollout=lt([t["logp"] for t in tokens], torch.float32),
        tok_turn=lt([t["turn"] for t in tokens]),
        tok_pen=_pairs([t["penalized"] for t in tokens]),
        tok_ban=_pairs([t["banned"] for t in tokens]),
        tok_rep=_pairs(self_rep),
        tok_echo=_pairs(echo),
        spk_agent=lt([s["agent"] for s in speaks]),
        spk_src=lt([s["src"] for s in speaks]),
        spk_act=lt([s["speak"] for s in speaks], torch.float32),
        spk_bias=lt([s.get("bias", 0.0) for s in speaks], torch.float32),
        spk_logp_rollout=lt([s["logp"] for s in speaks], torch.float32),
        spk_turn=lt([s["turn"] for s in speaks]),
        spk_group=lt(spk_group),
        spk_state={k: lt(v, torch.float32) for k, v in states.items()},
        ch_row=lt([row_of[flat(c["agent"], c["src"])] for c in choices]),
        ch_cands=ch_cands, ch_mask=ch_mask,
        ch_chosen=lt([c["chosen"] for c in choices]),
        ch_logp_rollout=lt([c["logp"] for c in choices], torch.float32),
        ch_turn=lt([c["turn"] for c in choices]),
        ch_is_vote=lt([c["kind"] == "ballot" for c in choices], torch.bool),
        ch_svote=ch_svote,
        turn_agent=lt([t["agent"] for t in turns]),
        turn_role=lt([ROLE_IDS[roles[t["agent"]]] for t in turns]),
        turn_hidden_pos=lt([flat(t["agent"], max(t["start"] - 1, 0)) for t in turns]),
        turn_features=feat,
        turn_r_round=lt(r_round, torch.float32),
        turn_r_game=lt(r_game, torch.float32),
        agent_turns=agent_turns,
        info={"columns": int(input_ids.shape[2]), "winner": events["winner"], "roles": roles,
              "days": result["days_played"]},
    )


@dataclass
class SamplingConfig:
    """How rollouts sampled; the recomputation must use the same values."""

    temperature: float
    repetition_penalty: float
    vocab_limit: int | None
    activity_temperature: float
    vote_temperature: float


def _token_chunk(rows, ids, pen_mask, ban_mask, rep_mask, echo_mask, scfg):
    """Filtered log-prob of the sampled token, plus raw P(rep)/P(echo) maxima."""
    z = rows.float()
    raw_logp = torch.log_softmax(z, dim=-1)
    rp = scfg.repetition_penalty
    if rp != 1.0:
        z = torch.where(pen_mask, torch.where(z > 0, z / rp, z * rp), z)
    if scfg.vocab_limit is not None and scfg.vocab_limit < z.shape[-1]:
        z = z.clone()
        z[:, scfg.vocab_limit:] = float("-inf")
    z = z.masked_fill(ban_mask, float("-inf"))
    logp = torch.log_softmax(z / scfg.temperature, dim=-1).gather(1, ids.unsqueeze(1)).squeeze(1)
    neg_inf = torch.finfo(raw_logp.dtype).min
    rep_max = raw_logp.masked_fill(~rep_mask, neg_inf).max(dim=-1).values
    echo_max = raw_logp.masked_fill(~echo_mask, neg_inf).max(dim=-1).values
    return logp, rep_max, echo_max


def _pair_mask(pairs, lo, hi, vocab, device):
    mask = torch.zeros(hi - lo, vocab, dtype=torch.bool, device=device)
    sel = (pairs[0] >= lo) & (pairs[0] < hi)
    if sel.any():
        mask[pairs[0, sel] - lo, pairs[1, sel]] = True
    return mask


def evaluate_game(model, g: GameBatch, scfg: SamplingConfig, *, tap_layers=(), chunk=128,
                  use_checkpoint=True):
    """One teacher-forced pass: log-probs of every recorded action under the current weights."""
    out = model.forward_selected(
        g.input_ids, g.logit_positions,
        input_activity_mask=g.activity, input_private_mask=g.private, agent_visibility=g.visibility,
        hidden_layers=tuple(tap_layers), hidden_positions=g.turn_hidden_pos if tap_layers else None,
    )
    logits = out["logits"][0]               # [K, V]
    act = out["activity_logits"][0].float()  # [A, T]
    vocab = logits.shape[-1]
    device = logits.device

    # Word tokens, chunked (each chunk materializes [chunk, V] float tensors).
    n = g.tok_id.numel()
    tok_logp, rep_max, echo_max = [], [], []
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        args = (logits[g.tok_row[lo:hi]], g.tok_id[lo:hi],
                _pair_mask(g.tok_pen, lo, hi, vocab, device), _pair_mask(g.tok_ban, lo, hi, vocab, device),
                _pair_mask(g.tok_rep, lo, hi, vocab, device), _pair_mask(g.tok_echo, lo, hi, vocab, device))
        if use_checkpoint and torch.is_grad_enabled():
            res = checkpoint(_token_chunk, *args, scfg, use_reentrant=False)
        else:
            res = _token_chunk(*args, scfg)
        tok_logp.append(res[0])
        rep_max.append(res[1])
        echo_max.append(res[2])
    empty = logits.new_zeros(0, dtype=torch.float32)
    tok_logp = torch.cat(tok_logp) if tok_logp else empty
    rep_max = torch.cat(rep_max) if rep_max else empty
    echo_max = torch.cat(echo_max) if echo_max else empty

    # Speak decisions.
    spk_logit = (act[g.spk_agent, g.spk_src] + g.spk_bias) / scfg.activity_temperature
    spk_logp = torch.where(g.spk_act > 0, F.logsigmoid(spk_logit), F.logsigmoid(-spk_logit))
    q = torch.sigmoid(spk_logit)

    # Name choices: softmax over each candidate's single name token.
    if g.ch_row.numel():
        cand_logits = logits[g.ch_row].float().gather(1, g.ch_cands) / scfg.vote_temperature
        cand_logp = torch.log_softmax(cand_logits.masked_fill(~g.ch_mask, float("-inf")), dim=-1)
        ch_logp = cand_logp.gather(1, g.ch_chosen.unsqueeze(1)).squeeze(1)
    else:
        cand_logp = empty.view(0, 1)
        ch_logp = empty
    return {
        "tok_logp": tok_logp, "rep_max": rep_max, "echo_max": echo_max,
        "spk_logp": spk_logp, "q": q, "ch_logp": ch_logp, "cand_logp": cand_logp,
        "hidden": out.get("hidden"),
    }


def p_other(q, groups):
    """P(some other agent in the same column speaks), per decision (detached)."""
    q = q.detach().clamp(0.0, 1.0 - 1e-6)
    log_quiet = torch.log1p(-q)
    total = torch.zeros(int(groups.max()) + 1 if groups.numel() else 0, device=q.device).index_add_(
        0, groups, log_quiet)
    return 1.0 - torch.exp(total[groups] - log_quiet)


def unlikelihood(max_logp, has_target):
    """``-log(1 - P)`` at positions with a target, 0 elsewhere."""
    p = max_logp.exp().clamp(max=1.0 - 1e-6)
    return torch.where(has_target, -torch.log1p(-p), torch.zeros_like(p))


@dataclass
class LossConfig:
    lambda_conv: float = 1.00
    lambda_ppo: float = 0.25
    beta_kl: float = 0.02
    clip_eps: float = 0.2
    kl_on_speak: bool = True  # False for a plain-Qwen reference (it has no activity head)


def game_losses(g: GameBatch, cur, old, ref, adv, reward_cfg: RewardConfig, loss_cfg: LossConfig,
                norm):
    """Actor loss of one game, scaled so summing over all games gives batch means.

    ``cur``/``old``/``ref``: :func:`evaluate_game` outputs (old/ref detached).
    ``adv``: dict with ``turn`` (strategic advantage per turn) and ``vote``
    (exact-baseline advantage per choice). ``norm``: global action counts
    across every game on every rank: ``actions``, ``speaks``, ``tokens``.
    """
    stats = {}
    # PPO clip over every action type.
    new = torch.cat([cur["tok_logp"], cur["spk_logp"], cur["ch_logp"]])
    prev = torch.cat([old["tok_logp"], old["spk_logp"], old["ch_logp"]]).detach()
    a_turn = adv["turn"]
    a_choice = torch.where(g.ch_is_vote, adv["vote"], a_turn[g.ch_turn])
    advantage = torch.cat([a_turn[g.tok_turn], a_turn[g.spk_turn], a_choice]).detach()
    ratio = torch.exp(new - prev)
    clipped = torch.clamp(ratio, 1 - loss_cfg.clip_eps, 1 + loss_cfg.clip_eps)
    ppo = -torch.min(ratio * advantage, clipped * advantage).sum() / norm["actions"]
    stats["clip_frac"] = float(((ratio - 1).abs() > loss_cfg.clip_eps).float().sum())
    stats["approx_kl_old"] = float((prev - new).detach().sum())

    # KL to the reference (k3 estimator on the sampled actions).
    if loss_cfg.kl_on_speak:
        ref_lp = torch.cat([ref["tok_logp"], ref["spk_logp"], ref["ch_logp"]]).detach()
        cur_lp, kl_count = new, norm["actions"]
    else:
        ref_lp = torch.cat([ref["tok_logp"], ref["ch_logp"]]).detach()
        cur_lp, kl_count = torch.cat([cur["tok_logp"], cur["ch_logp"]]), norm["actions"] - norm["speaks"]
    log_ratio = ref_lp - cur_lp
    kl = (torch.exp(log_ratio) - log_ratio - 1).sum() / max(kl_count, 1)
    stats["kl_ref"] = float(kl.detach() * max(kl_count, 1))

    # Pillar 1: floor penalties on speak decisions + text unlikelihood on tokens.
    if g.spk_act.numel():
        floor = floor_penalty_sum(cur["q"], g, reward_cfg) / max(norm["speaks"], 1)
    else:
        floor = new.new_zeros(())
    n_tok = g.tok_id.numel()
    has_rep = torch.zeros(n_tok, dtype=torch.bool, device=new.device)
    has_echo = torch.zeros(n_tok, dtype=torch.bool, device=new.device)
    if g.tok_rep.numel():
        has_rep[g.tok_rep[0]] = True
    if g.tok_echo.numel():
        has_echo[g.tok_echo[0]] = True
    rep = unlikelihood(cur["rep_max"], has_rep).sum() / max(norm["tokens"], 1)
    echo = unlikelihood(cur["echo_max"], has_echo).sum() / max(norm["tokens"], 1)
    conv = floor + reward_cfg.lambda_rep * rep + reward_cfg.lambda_echo * echo
    stats.update(floor=float(floor.detach()) * max(norm["speaks"], 1),
                 self_rep=float(rep.detach()) * max(norm["tokens"], 1),
                 echo=float(echo.detach()) * max(norm["tokens"], 1))

    loss = loss_cfg.lambda_conv * conv + loss_cfg.lambda_ppo * ppo + loss_cfg.beta_kl * kl
    return loss, stats


def floor_penalty_sum(q, g: GameBatch, reward_cfg: RewardConfig):
    from .rewards import floor_penalty

    return floor_penalty(q, p_other(q, g.spk_group), g.spk_state, reward_cfg).sum()


def vote_advantages(g: GameBatch, old):
    """Exact zero-variance baseline for day votes: S_vote(sampled) - E_old[S_vote]."""
    if not g.ch_row.numel():
        return torch.zeros(0, device=g.ch_row.device)
    p_old = old["cand_logp"].detach().exp().masked_fill(~g.ch_mask, 0.0)
    expected = (p_old * g.ch_svote).sum(dim=1)
    sampled = g.ch_svote.gather(1, g.ch_chosen.unsqueeze(1)).squeeze(1)
    return sampled - expected
