"""The three reward/loss pillars of WEREWOLF_RL_PLAN.md, computed from one game.

Pillar 1 (conversation hygiene) is a penalty-only *differentiable* loss on the
activity head and the vocab head, so this module provides the per-action
realized floor state (non-differentiable counters from the game that was
actually played) plus :func:`floor_penalty`, which turns the policy's current
speak probabilities into the penalty.

Pillar 2 (good round) and Pillar 3 (good game) are scalar rewards from the
game's events: :func:`round_rewards`, :func:`game_rewards`, and
:func:`turn_step_rewards`, which places them on each agent's turns.

Where the plan leaves a detail open, the choice made here is documented on the
config field that controls it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch

ROLE_WORDS = ("Werewolf", "Werewolves", "Seer", "Doctor", "Villager", "Villagers", "Minion", "vote",
              "votes", "voted", "Wolf", "wolf")


@dataclass
class RewardConfig:
    # Pillar 1: per-agent penalty-only floor thresholds (plan section 4.1).
    overlap_grace: float = 5.0      # 1..5 tokens of overlap = 0 penalty
    overlap_tau: float = 4.0
    overlap_max_weight: float = 3.5
    speak_grace: float = 30.0       # 1..30 tokens per turn = 0 penalty
    speak_tau: float = 40.0
    monopoly_len_weight: float = 1.5
    share_cap: float = 0.45         # <=45% of the day's discussion tokens = 0 penalty
    share_weight: float = 2.0
    # Not in the plan: the round-share term only counts once the day's discussion
    # has this many spoken tokens. Without it the first speaker of every day holds
    # 100% share on token 1 and is penalized just for opening the floor.
    share_min_tokens: int = 30
    silence_grace: float = 3.0      # 1..3 silent columns = 0 penalty
    silence_tau: float = 8.0
    silence_max_weight: float = 1.5
    # Text penalties (unlikelihood on the vocab head). The plan names no weights;
    # 0.03 matches the SFT run's lambda_repetition_penalty.
    lambda_rep: float = 0.03
    lambda_echo: float = 0.03
    # Self-repetition = continuing a 4-gram the agent already said this game
    # (the plan's "within-agent repeated 4-grams").
    self_rep_ngram: int = 4
    # Cross-agent echo = continuing a 2-token span another agent said in the
    # last 32 columns; player names and role words are exempt.
    echo_window: int = 32
    # Pillar 2 internal split (plan section 4.2).
    w_suspicion: float = 0.50
    w_vote: float = 0.50


# ----------------------------------------------------------------------------
# Pillars 2 and 3
# ----------------------------------------------------------------------------
def vote_score(roles, voter, target):
    """S_vote of one ballot: +1 for voting the other team, -1 for your own."""
    wolf_voter = roles[voter] == "Werewolf"
    wolf_target = roles[target] == "Werewolf"
    return 1.0 if wolf_voter != wolf_target else -1.0


def round_rewards(events, cfg: RewardConfig):
    """``{(seat, day): {"s_suspicion", "s_vote", "r_round"}}`` for every day ballot.

    Round d = Day d + Night d. S_suspicion is computed from the votes cast
    against the player in Day d's vote. The Seer's S_vote averages its day
    ballot with its Night-d look (+1 for a not-yet-checked player, -1 for a
    re-check).
    """
    roles = events["roles"]
    out = {}
    for day_ev in events["days"]:
        d = day_ev["day"]
        n_alive = len(day_ev["alive"])
        against = Counter(target for _, target in day_ev["votes"])
        for voter, target in day_ev["votes"]:
            s_susp = 1.0 - 2.0 * against[voter] / max(1, n_alive - 1)
            out[(voter, d)] = {"s_suspicion": s_susp, "s_vote": vote_score(roles, voter, target)}
    for night_ev in events["nights"]:
        if night_ev.get("seer"):
            seer, _target, new_info = night_ev["seer"]
            key = (seer, night_ev["night"])
            look = 1.0 if new_info else -1.0
            if key in out:
                out[key]["s_vote"] = 0.5 * (out[key]["s_vote"] + look)
    for vals in out.values():
        vals["r_round"] = cfg.w_suspicion * vals["s_suspicion"] + cfg.w_vote * vals["s_vote"]
    return out


def game_rewards(events):
    """S_win per seat: +1 if the seat's team won, -1 if it lost, 0 if nobody won."""
    winner = events["winner"]
    out = []
    for role in events["roles"]:
        team = "Werewolves" if role == "Werewolf" else "Village"
        out.append(0.0 if winner is None else (1.0 if team == winner else -1.0))
    return out


def turn_step_rewards(turns, events, cfg: RewardConfig):
    """Per-turn step rewards ``(r_round_step, r_game_step)``, both lists over ``turns``.

    ``r_round`` for round d lands on the agent's last turn of round d; S_win
    lands on the agent's last turn of the game (plan section 5.2).
    """
    rounds = round_rewards(events, cfg)
    wins = game_rewards(events)
    r_round = [0.0] * len(turns)
    r_game = [0.0] * len(turns)
    last_in_round, last_overall = {}, {}
    for i, turn in enumerate(turns):
        a = turn["agent"]
        key = (a, turn["day"])
        if key not in last_in_round or turns[last_in_round[key]]["start"] < turn["start"]:
            last_in_round[key] = i
        if a not in last_overall or turns[last_overall[a]]["start"] < turn["start"]:
            last_overall[a] = i
    for key, i in last_in_round.items():
        if key in rounds:
            r_round[i] = rounds[key]["r_round"]
    for a, i in last_overall.items():
        r_game[i] = wins[a]
    return r_round, r_game


def discounted_gae(rewards, values, gamma, lam):
    """GAE over one agent's turn sequence; the value after the last turn is 0."""
    advantages = [0.0] * len(rewards)
    running = 0.0
    for u in reversed(range(len(rewards))):
        next_value = values[u + 1] if u + 1 < len(rewards) else 0.0
        delta = rewards[u] + gamma * next_value - values[u]
        running = delta + gamma * lam * running
        advantages[u] = running
    return advantages


# ----------------------------------------------------------------------------
# Pillar 1: floor state and penalty
# ----------------------------------------------------------------------------
def floor_states(activity, spans, speaks, cfg: RewardConfig):
    """Realized floor counters before each free-discussion speak decision.

    ``activity``: ``[A, T]`` bool matrix of the played game. ``spans``: the
    free-discussion column ranges. ``speaks``: recorded speak actions; one
    decided at ``src`` chooses column ``src + 1``, so it sees the state after
    column ``src``. Returns a dict of lists aligned with ``speaks``:

    * ``omega`` -- consecutive columns the agent has been overlapping without
      floor protection. The floor owner (the sole speaker before an overlap
      began) is protected while its own run is within ``speak_grace``.
    * ``run`` -- the agent's current consecutive speaking run (its turn length).
    * ``share`` -- its fraction of the day's discussion tokens so far (0 until
      ``share_min_tokens`` tokens have been spoken).
    * ``silence`` -- consecutive all-silent columns at the table.
    * ``not_last`` -- 1 unless the agent spoke in the most recent spoken column.
    """
    activity = activity.bool().cpu()
    num_agents = activity.shape[0]
    wanted = {}
    for i, s in enumerate(speaks):
        wanted.setdefault(s["src"], []).append(i)
    out = {k: [0.0] * len(speaks) for k in ("omega", "run", "share", "silence", "not_last")}

    for span in spans:
        if not span.get("free", True):
            continue
        run = [0] * num_agents
        omega = [0] * num_agents
        tokens = [0] * num_agents
        total = 0
        silence = 0
        owner = None
        last = {span.get("opener")}

        def emit(src):
            for i in wanted.get(src, []):
                a = speaks[i]["agent"]
                out["omega"][i] = float(omega[a])
                out["run"][i] = float(run[a])
                out["share"][i] = tokens[a] / total if total >= cfg.share_min_tokens else 0.0
                out["silence"][i] = float(silence)
                out["not_last"][i] = 0.0 if a in last else 1.0

        emit(span["start"] - 1)
        for c in range(span["start"], span["end"]):
            speakers = [a for a in range(num_agents) if activity[a, c]]
            silence = 0 if speakers else silence + 1
            for a in range(num_agents):
                run[a] = run[a] + 1 if a in speakers else 0
            if len(speakers) == 1:
                owner = speakers[0]
            elif not speakers or owner not in speakers:
                owner = None  # silence, or a collision with no prior owner
            for a in range(num_agents):
                if len(speakers) > 1 and a in speakers:
                    protected = a == owner and run[a] <= cfg.speak_grace
                    omega[a] = 0 if protected else omega[a] + 1
                else:
                    omega[a] = 0
            for a in speakers:
                tokens[a] += 1
            total += len(speakers)
            if speakers:
                last = set(speakers)
            emit(c)
    return out


def floor_penalty(q, p_other, state, cfg: RewardConfig):
    """Per-decision Pillar 1 floor penalty (>= 0), differentiable in ``q``.

    ``q``: the policy's P(speak) for each decision; ``p_other``: probability
    that some other living agent speaks in the same column (pass it detached,
    so the penalty is attributed only to the intruding agent); ``state``:
    tensors from :func:`floor_states`.
    """
    omega, run, share, silence, not_last = (state[k] for k in ("omega", "run", "share", "silence", "not_last"))
    relu = torch.relu
    overlap = cfg.overlap_max_weight * (1 - torch.exp(-relu(omega - cfg.overlap_grace) / cfg.overlap_tau))
    overlap = overlap * q * p_other
    monopoly = (cfg.monopoly_len_weight * (1 - torch.exp(-relu(run - cfg.speak_grace) / cfg.speak_tau))
                + cfg.share_weight * relu((share - cfg.share_cap) / 0.25)) * q
    silent = cfg.silence_max_weight * (1 - torch.exp(-relu(silence - cfg.silence_grace) / cfg.silence_tau))
    silent = silent * not_last * (1 - share) * (1 - q)
    return overlap + monopoly + silent


def exempt_token_ids(tokenizer, names):
    """Token ids of player names and role/game words, never counted as an echo."""
    ids = set()
    for word in list(names) + list(ROLE_WORDS):
        for variant in {word, word.lower(), word.capitalize()}:
            for text in (variant, " " + variant):
                ids.update(tokenizer(text, add_special_tokens=False)["input_ids"])
    return ids


def text_penalty_targets(input_ids, activity, tokens, exempt, cfg: RewardConfig):
    """Per token action, the token ids whose probability the text penalties push down.

    Returns ``(self_rep, echo)``, two lists of id lists aligned with ``tokens``:

    * self-repetition: tokens that would continue a ``self_rep_ngram``-gram the
      agent already said earlier in the game;
    * echo: tokens that would continue a 2-token span another agent said in
      the last ``echo_window`` columns (both tokens non-exempt).
    """
    input_ids = input_ids.cpu()
    activity = activity.bool().cpu()
    num_agents, num_cols = activity.shape
    # Each agent's spoken tokens with their columns, in order.
    spoken = []
    for a in range(num_agents):
        cols = activity[a].nonzero().flatten().tolist()
        spoken.append((cols, input_ids[a, cols].tolist()))
    n = cfg.self_rep_ngram
    self_rep, echo = [], []
    for action in tokens:
        a, src = action["agent"], action["src"]
        cols, toks = spoken[a]
        k = _count_upto(cols, src)
        hist = toks[:k]
        reps = set()
        if n > 1 and len(hist) >= n - 1:
            tail = tuple(hist[-(n - 1):])
            for i in range(len(hist) - n + 1):
                if tuple(hist[i:i + n - 1]) == tail:
                    reps.add(hist[i + n - 1])
        self_rep.append(sorted(reps))
        echoes = set()
        # Mid-utterance only: the agent's previous token must be in column src.
        if k and cols[k - 1] == src and hist[-1] not in exempt:
            prev = hist[-1]
            lo = src - cfg.echo_window
            for b in range(num_agents):
                if b == a:
                    continue
                bcols, btoks = spoken[b]
                first = max(1, _count_upto(bcols, lo) + 1)  # first j with bcols[j - 1] > lo
                for j in range(first, _count_upto(bcols, src)):
                    if (bcols[j] == bcols[j - 1] + 1 and btoks[j - 1] == prev
                            and btoks[j] not in exempt):
                        echoes.add(btoks[j])
        echo.append(sorted(echoes))
    return self_rep, echo


def _count_upto(sorted_cols, src):
    """Number of entries <= src in an ascending list."""
    lo, hi = 0, len(sorted_cols)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_cols[mid] <= src:
            lo = mid + 1
        else:
            hi = mid
    return lo
