"""One Werewolf game between MatrixChat agents, played in one shared matrix.

Shared by ``scripts/eval/werewolf_game.py`` (play and log a game) and
``scripts/training/train_werewolf_rl.py`` (collect RL rollouts). There is no
separate game engine: the game is prompt tokens, forced speaking turns and
per-turn visibility in one KV-cached matrix.

Rules (see the eval script's docstring for the full description):

* Day d, then Night d, until a side wins (or ``--max_days``).
* Village wins when every Werewolf is dead; Werewolves win at parity
  (living Werewolves >= living non-Werewolves).
* Day vote: every living player must name another living player (no skip).
  A unique plurality is eliminated; a tie eliminates one tied player at random.
* Night: the Seer (``--seer``) privately looks at one player's card, then the
  living Werewolves privately vote on a kill (plurality, ties random).

With ``record=True`` every sampled action is logged with the log-probability
it was sampled with, so the trainer can recompute exactly the same
distribution in a teacher-forced pass:

* ``speaks``  -- free-discussion speak/yield decisions,
  ``speak ~ Bernoulli(sigmoid(activity_logit / activity_temperature))``.
* ``tokens``  -- every sampled word token (including a sampled stop token that
  ends a turn), with the repetition-penalized and banned token ids that the
  sampling filters applied. These filters depend only on the history, so the
  trainer can redo them exactly. Top-p is not allowed when recording (its
  kept set depends on the weights).
* ``choices`` -- votes, kill votes and Seer looks: a softmax over the
  candidates' single name tokens at ``vote_temperature``.
* ``turns``   -- each agent's turns (utterances and ballots), the steps of
  the turn-level MDP the advantages are computed on.

Every action names ``src``: the matrix column whose outputs it was sampled
from (the newest column at that moment), so the trainer reads logits and
activity at ``(agent, src)``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F

from data.adapters._names import RANDOM_NAME_POOL
from data.convert import ConvertConfig, _pause_columns

ROLES = ("Villager", "Seer", "Werewolf")
ROLE_IDS = {role: i for i, role in enumerate(ROLES)}
# Phase ids for the critic's game-state features (WEREWOLF_RL_PLAN.md section 2.2).
PHASE_IDS = {"night": 0, "discussion": 1, "ballot": 2, "vote": 3}
TURN_PHASE = {"kill": "night", "seer": "night", "discussion": "discussion", "ballot": "ballot"}

GOALS = {
    "Werewolf": "I must not let anyone find out.",
    "Villager": "I need to work out who the Werewolf is.",
    "Seer": "Each night I can secretly look at one player's card. "
            "I need to use what I learn to find the Werewolf.",
}
# The name slot must read as an accusation. "I vote for X" reads to the model as
# support ("...because I don't think she's a werewolf"), so voters picked people
# they trusted and the eliminated seat was rarely the Werewolf.
VOTE_PREFIX = "I think the Werewolf is"


def add_game_args(p):
    """Game-rule and sampling flags, shared by the eval script and the trainer."""
    p.add_argument("--num_players", type=int, default=6)
    p.add_argument("--num_werewolves", type=int, default=1)
    p.add_argument("--seer", action="store_true",
                   help="Make one non-Werewolf the Seer: each night it privately picks a living player, "
                        "explains why, and learns whether that player is a Werewolf.")
    p.add_argument("--plain_cards", action="store_true",
                   help="Use the exact training card format ('You are X. Your role is Y.'), without the "
                        "goal. Default cards are first person: 'I am X. I am a Villager. ...'.")
    p.add_argument("--name_pool", choices=["single_token", "all"], default="single_token",
                   help="single_token (default): only names that are one token after every ballot prefix, "
                        "so a vote is an exact softmax over one token per candidate (required for RL "
                        "rollouts). all: the full name pool (multi-token names are scored by branching).")
    p.add_argument("--max_days", type=int, default=0,
                   help="0 (default) = play until a side wins. N > 0 stops after N days.")
    p.add_argument("--opener", default="Okay, so who do we think the werewolf is?",
                   help="Question that opens every day. From day 2 on, a different living player "
                        "opens and first announces who died in the night.")
    p.add_argument("--death_announcement",
                   default="Last night, {dead} was killed by the Werewolves. {dead} is dead and out of the game.")
    # Night.
    p.add_argument("--night_prompt",
                   default="It's night {night}. Werewolves only: choose one Villager to kill tonight.",
                   help="Private to the Werewolves; said in the first living Werewolf's row.")
    p.add_argument("--kill_prefix", default="Tonight I want to kill",
                   help="Prefilled start of each Werewolf's private kill ballot; the scored name follows.")
    p.add_argument("--seer_prompt",
                   default="It's night {night}. As the Seer, I can secretly look at one player's card now.",
                   help="Private to the Seer; said in the Seer's row at the start of the night.")
    p.add_argument("--seer_prefix", default="I want to look at",
                   help="Prefilled start of the Seer's private pick; the scored name, then 'because ...'.")
    p.add_argument("--seer_result", default="I looked at {target}'s card: {target} is {seen}.",
                   help="Private to the Seer; {seen} is 'a Werewolf' or 'a Villager'.")
    # Discussion.
    p.add_argument("--discussion_mode", choices=["free", "turns"], default="free",
                   help="free (default): the activity head decides who speaks each column "
                        "(interruptions/overlap allowed) until everyone is quiet for --quiet_seconds. "
                        "turns: forced seat order for --discussion_rounds rounds.")
    p.add_argument("--speak_sampling", choices=["bernoulli", "threshold"], default="bernoulli",
                   help="free mode. bernoulli (default): speak ~ Bernoulli(sigmoid(logit / "
                        "--activity_temperature)), the policy RL trains. threshold: the old rule, speak "
                        "when P(speak) > --activity_threshold (not usable for RL rollouts).")
    p.add_argument("--activity_temperature", type=float, default=1.0,
                   help="bernoulli mode: temperature on the activity logit. 1.0 = the head's own "
                        "probability; lower values approach the 0.5-threshold rule.")
    p.add_argument("--activity_threshold", type=float, default=0.5, help="threshold mode only.")
    p.add_argument("--quiet_seconds", type=float, default=3.0,
                   help="free mode with --rescue_after 0 only: end the discussion after this long with "
                        "nobody speaking. Converted to all-silent columns exactly as training data encodes "
                        "pauses (data/convert.py).")
    p.add_argument("--silence_ramp", type=float, default=0.75,
                   help="free mode: during a table-wide silence, add this much to every living player's "
                        "speak logit per silent column past --silence_ramp_start (0 = off). The bias is "
                        "part of the sampled policy and is recorded, so RL stays exact.")
    p.add_argument("--silence_ramp_start", type=int, default=2,
                   help="Silent columns before the ramp starts (natural pauses are left alone).")
    p.add_argument("--rescue_after", type=int, default=5,
                   help="free mode: after this many all-silent columns, the living player with the "
                        "highest P(speak) is made to speak (its first token cannot be a stop). "
                        "0 = no rescue; the discussion then ends after --quiet_seconds of quiet.")
    p.add_argument("--rescue_end_after", type=int, default=3,
                   help="The Nth time the silence reaches --rescue_after columns ends the day's "
                        "discussion instead of rescuing it (same convention as model/generation.py).")
    p.add_argument("--max_discussion_columns", type=int, default=600,
                   help="free mode: safety cap on columns per day's discussion (0 = no cap).")
    p.add_argument("--discussion_rounds", type=int, default=2, help="turns mode: times around the table.")
    p.add_argument("--turn_tokens", type=int, default=40, help="turns mode: max generated tokens per turn.")
    p.add_argument("--max_sentences", type=int, default=2, help="turns mode: end a turn after this many.")
    p.add_argument("--starters", nargs="*", default=[],
                   help="turns mode: optional prefilled openings, one chosen at random per turn.")
    # Vote.
    p.add_argument("--vote_prompt",
                   default="Alright, time to vote. Say who you think the Werewolf is and why. "
                           "Whoever gets the most votes is out.")
    p.add_argument("--vote_prefix", default=VOTE_PREFIX,
                   help="Prefilled start of every ballot; the scored name follows it.")
    p.add_argument("--vote_temperature", type=float, default=1.0,
                   help="Sample the vote from the name probabilities at this temperature (0 = argmax).")
    p.add_argument("--ballot_order", choices=["reason_first", "name_first"], default="reason_first",
                   help="reason_first: the voter reasons freely (the forced justification), THEN names a "
                        "suspect. name_first: name, then a post-hoc 'because ...'.")
    p.add_argument("--reason_tokens", type=int, default=40, help="Max generated tokens of ballot reasoning.")
    p.add_argument("--reason_sentences", type=int, default=2, help="reason_first: max sentences of reasoning.")
    p.add_argument("--blind_votes", action="store_true",
                   help="Each player votes from the discussion alone, without hearing earlier votes "
                        "(not usable for RL rollouts: ballots then live outside the shared matrix).")
    p.add_argument("--vote_result",
                   default="{out} got the most votes, so {out} is out of the game.",
                   help="Said publicly by the day's opener after the vote (roles are not revealed).")
    p.add_argument("--vote_tie",
                   default="It's a tie between {tied}, so we drew lots: {out} is out of the game.",
                   help="Said when the vote ties; one tied player is eliminated at random.")
    # Sampling for spoken tokens.
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=1.0,
                   help="Nucleus sampling (1.0 = off, required for RL rollouts).")
    p.add_argument("--repetition_penalty", type=float, default=1.3)
    p.add_argument("--no_repeat_ngram", type=int, default=3, help="Ban repeating n-grams within a turn.")
    p.add_argument("--no_copy_ngram", type=int, default=5,
                   help="Ban repeating any n-gram from an earlier model-written turn by ANY player; 0 disables.")
    return p


def check_record_args(args):
    """Reject settings whose actions the trainer could not recompute exactly."""
    problems = []
    if args.discussion_mode == "free" and args.speak_sampling != "bernoulli":
        problems.append("--speak_sampling bernoulli")
    if args.temperature <= 0:
        problems.append("--temperature > 0")
    if args.vote_temperature <= 0:
        problems.append("--vote_temperature > 0")
    if args.top_p < 1.0:
        problems.append("--top_p 1.0")
    if args.blind_votes:
        problems.append("no --blind_votes")
    if args.name_pool != "single_token":
        problems.append("--name_pool single_token")
    if problems:
        raise ValueError("RL rollouts need: " + ", ".join(problems))


# ----------------------------------------------------------------------------
# Sampling filters (shared with the trainer so both compute one distribution)
# ----------------------------------------------------------------------------
def apply_filters(logits, penalized, banned, repetition_penalty, vocab_limit):
    """The sampling filters on one row of logits; returns float32 logits.

    ``penalized``: ids divided (positive logit) / multiplied (negative) by
    ``repetition_penalty``. ``banned``: ids set to -inf. Ids at or beyond
    ``vocab_limit`` (padding rows of the embedding matrix) are -inf.
    """
    z = logits.float().clone()
    if vocab_limit is not None and vocab_limit < z.shape[-1]:
        z[vocab_limit:] = float("-inf")
    if repetition_penalty != 1.0 and penalized:
        idx = torch.tensor(penalized, device=z.device)
        seen = z[idx]
        z[idx] = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
    if banned:
        z[torch.tensor(banned, device=z.device)] = float("-inf")
    return z


_YIELD_IDS = {}


class Speaker:
    """Sampling for spoken tokens: repetition penalty, n-gram bans, temperature, top-p."""

    def __init__(self, tokenizer, args):
        self.tokenizer, self.args = tokenizer, args
        stop = {tokenizer.eos_token_id} if getattr(tokenizer, "eos_token_id", None) is not None else set()
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        for special in ("<|im_end|>", "<|endoftext|>"):
            token_id = convert(special) if convert else None
            if isinstance(token_id, int) and token_id != getattr(tokenizer, "unk_token_id", None):
                stop.add(token_id)
        self.stop_ids = stop
        try:
            self.vocab_limit = len(tokenizer)
        except TypeError:
            self.vocab_limit = None
        # (n-1)-gram -> next tokens already said in an earlier model-written turn
        # (any player), so nobody can parrot an earlier line verbatim.
        self.copy_index = {}

    def remember(self, token_ids):
        """Record a finished model-written turn for the cross-turn copy ban."""
        m = self.args.no_copy_ngram
        if m < 2:
            return
        for i in range(len(token_ids) - m + 1):
            self.copy_index.setdefault(tuple(token_ids[i:i + m - 1]), set()).add(token_ids[i + m - 1])

    def yield_ids(self):
        """Stop ids plus whitespace-only newline tokens: sampling one of these means "stop talking"."""
        cache = _YIELD_IDS.get(id(self.tokenizer))
        if cache is None:
            limit = self.vocab_limit or 0
            cache = sorted(self.stop_ids | {
                i for i in range(limit)
                if "\n" in (piece := self.tokenizer.decode([i])) and not piece.strip()})
            _YIELD_IDS[id(self.tokenizer)] = cache
        return cache

    def constraints(self, history, extra_banned=()):
        """``(penalized, banned)`` token ids the filters apply after ``history``."""
        args = self.args
        banned = set(extra_banned)
        m = args.no_copy_ngram
        if m >= 2 and len(history) >= m - 1:
            banned |= self.copy_index.get(tuple(history[-(m - 1):]), set())
        n = args.no_repeat_ngram
        if n > 1 and len(history) >= n - 1:
            tail = tuple(history[-(n - 1):])
            banned |= {history[i + n - 1] for i in range(len(history) - n + 1)
                       if tuple(history[i:i + n - 1]) == tail}
        penalized = sorted(set(history)) if args.repetition_penalty != 1.0 else []
        return penalized, sorted(banned)

    def sample(self, logits, history, extra_banned=()):
        """Return ``(token_id, logp, penalized, banned)``; ``logp`` is under the sampled distribution."""
        args = self.args
        penalized, banned = self.constraints(history, extra_banned)
        z = apply_filters(logits, penalized, banned, args.repetition_penalty, self.vocab_limit)
        if args.temperature <= 0:
            return int(z.argmax()), 0.0, penalized, banned
        logz = torch.log_softmax(z / args.temperature, dim=-1)
        if args.top_p < 1.0:
            probs = logz.exp()
            sorted_probs, order = probs.sort(descending=True)
            outside = sorted_probs.cumsum(-1) - sorted_probs > args.top_p
            sorted_probs[outside] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, order, sorted_probs)
            probs = probs / probs.sum()
            token_id = int(torch.multinomial(probs, 1))
            return token_id, float(torch.log(probs[token_id])), penalized, banned
        token_id = int(torch.multinomial(logz.exp(), 1))
        return token_id, float(logz[token_id]), penalized, banned


# ----------------------------------------------------------------------------
# The shared matrix
# ----------------------------------------------------------------------------
class Table:
    """The shared game matrix, extended one column at a time through the KV cache.

    ``logits[0, a]`` holds agent ``a``'s next-content logits and
    ``activity[0, a]`` its next-column speak logit, given everything so far.
    Every pushed column is also kept (``matrix()``) so the trainer can replay
    the whole game teacher-forced.
    """

    def __init__(self, model, tokenizer, input_ids, activity, private, visibility):
        self.model, self.tokenizer = model, tokenizer
        self.num_agents = input_ids.shape[1]
        self.device = input_ids.device
        self.prefix = (input_ids, activity, private)
        self.visibility = visibility
        self.columns = []  # (ids, active, private) lists of length num_agents
        self.cache, self.logits, self.activity = model.encode_prefix_for_generation(
            input_ids=input_ids, input_activity_mask=activity,
            input_private_mask=private, agent_visibility=visibility,
        )

    @property
    def seq_len(self):
        return self.cache.seq_len

    abort_check = None  # optional callable; raising from it abandons the game

    def push_column(self, tokens, private=False):
        """Append one column: ``tokens`` maps seat -> token id for every agent
        speaking in it (empty = an all-silent column, i.e. a pause)."""
        if self.abort_check is not None:
            self.abort_check()
        ids = [0] * self.num_agents
        active = [False] * self.num_agents
        for seat, token_id in tokens.items():
            ids[seat] = int(token_id)
            active[seat] = True
        priv = [a and bool(private) for a in active]
        self.columns.append((ids, active, priv))
        dev = self.device
        self.cache, self.logits, self.activity = self.model.forward_generation_column(
            self.cache,
            torch.tensor(ids, dtype=torch.long, device=dev).view(1, -1, 1),
            torch.tensor(active, dtype=torch.bool, device=dev).view(1, -1, 1),
            column_private=torch.tensor(priv, dtype=torch.bool, device=dev).view(1, -1, 1),
        )

    def push(self, speaker, token_id, private=False):
        """Append one token in ``speaker``'s row (everyone else silent)."""
        self.push_column({speaker: token_id}, private=private)

    def say(self, speaker, text, private=False):
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        for token_id in ids:
            self.push(speaker, token_id, private=private)
        return list(ids)

    def fork(self):
        twin = copy.copy(self)
        twin.cache = copy.deepcopy(self.cache)
        twin.columns = list(self.columns)
        return twin

    def matrix(self):
        """``(input_ids, activity, private)`` ``[1, A, T]`` CPU tensors of the whole game."""
        ids0, act0, priv0 = (x.detach().cpu() for x in self.prefix)
        if not self.columns:
            return ids0, act0.bool(), priv0.bool() if priv0 is not None else None
        ids = torch.tensor([c[0] for c in self.columns], dtype=torch.long).t().unsqueeze(0)
        act = torch.tensor([c[1] for c in self.columns], dtype=torch.bool).t().unsqueeze(0)
        priv = torch.tensor([c[2] for c in self.columns], dtype=torch.bool).t().unsqueeze(0)
        priv0 = priv0.bool() if priv0 is not None else torch.zeros_like(act0, dtype=torch.bool)
        return (torch.cat([ids0, ids], dim=2), torch.cat([act0.bool(), act], dim=2),
                torch.cat([priv0, priv], dim=2))


# ----------------------------------------------------------------------------
# Action recording
# ----------------------------------------------------------------------------
class Recorder:
    """Every sampled action of one game, with the log-prob it was sampled at."""

    def __init__(self):
        self.turns, self.tokens, self.speaks, self.choices = [], [], [], []
        self.alive = []  # kept current by the game loop

    def begin_turn(self, agent, start, kind, day):
        self.turns.append({"agent": agent, "start": start, "kind": kind, "day": day,
                           "alive": list(self.alive)})
        return len(self.turns) - 1

    def token(self, agent, src, token, logp, turn, penalized, banned):
        self.tokens.append({"agent": agent, "src": src, "token": token, "logp": logp, "turn": turn,
                            "penalized": penalized, "banned": banned})

    def speak(self, agent, src, speak, logp, day, turn=-1, bias=0.0):
        self.speaks.append({"agent": agent, "src": src, "speak": speak, "logp": logp, "day": day,
                            "turn": turn, "bias": bias})
        return len(self.speaks) - 1

    def choice(self, agent, src, cand_tokens, cand_seats, chosen, logp, kind, turn, day):
        self.choices.append({"agent": agent, "src": src, "cand_tokens": cand_tokens,
                             "cand_seats": cand_seats, "chosen": chosen, "logp": logp,
                             "kind": kind, "turn": turn, "day": day})

    def finalize(self):
        """Give pending actions (silence between turns) to the agent's next turn.

        A speak/yield decision taken while an agent is not mid-utterance is
        part of the run-up to its next turn; if it never speaks again, the
        decision joins its last turn. Returns the plain-dict record.
        """
        by_agent = {}
        for i, turn in enumerate(self.turns):
            by_agent.setdefault(turn["agent"], []).append((turn["start"], i))
        for action in self.tokens + self.speaks:
            if action["turn"] >= 0:
                continue
            starts = by_agent.get(action["agent"], [])
            later = [i for start, i in starts if start > action["src"]]
            action["turn"] = later[0] if later else (starts[-1][1] if starts else -1)
        return {"turns": self.turns, "tokens": self.tokens, "speaks": self.speaks,
                "choices": self.choices}


# ----------------------------------------------------------------------------
# Game setup
# ----------------------------------------------------------------------------
def name_token_ids(tokenizer, prefix, name):
    """Token ids of `` name`` as it would follow ``prefix`` in the same turn."""
    pre = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full = tokenizer(f"{prefix} {name}", add_special_tokens=False)["input_ids"]
    if full[:len(pre)] != pre:
        raise ValueError(f"tokenizer merges {prefix!r} with {name!r}; cannot score the name cleanly")
    return full[len(pre):]


def single_token_names(tokenizer, args, pool=RANDOM_NAME_POOL):
    """Names that are one clean token after every prefix a ballot can use."""
    prefixes = [lead + p for p in (args.vote_prefix, args.kill_prefix, args.seer_prefix) for lead in ("", " ")]
    keep = []
    for name in pool:
        try:
            if all(len(name_token_ids(tokenizer, pre, name)) == 1 for pre in prefixes):
                keep.append(name)
        except ValueError:
            continue
    return keep


def deal_roles(num_players, game_seed, opener_text, plain_cards=False, num_werewolves=1, seer=False,
               name_pool=RANDOM_NAME_POOL):
    """Return ``(rng, names, roles, wolves, opener, cards, turns)`` for one game.

    Werewolf cards are visible to every Werewolf (and so is all later private
    night talk); Villager and Seer cards are visible to their owner only.
    """
    if not 1 <= num_werewolves < num_players / 2:
        raise ValueError(f"need 1 <= num_werewolves < num_players/2, got {num_werewolves} of {num_players}")
    if len(name_pool) < num_players:
        raise ValueError(f"name pool has {len(name_pool)} names, need {num_players}")
    rng = random.Random(game_seed)
    names = rng.sample(list(name_pool), num_players)
    # One werewolf keeps the original RNG draw so old seeds deal the same game.
    wolves = [rng.randrange(num_players)] if num_werewolves == 1 else sorted(rng.sample(range(num_players), num_werewolves))
    opener = rng.randrange(num_players)
    roles = ["Werewolf" if i in wolves else "Villager" for i in range(num_players)]
    if seer:  # drawn after everything else, so games without --seer deal exactly as before
        roles[rng.choice([i for i in range(num_players) if i not in wolves])] = "Seer"
    cards = []
    for i in range(num_players):
        mates = [names[w] for w in wolves if w != i] if roles[i] == "Werewolf" else []
        mate_list = mates[0] if len(mates) == 1 else f"{', '.join(mates[:-1])} and {mates[-1]}" if mates else ""
        if plain_cards:  # exact training format (data/adapters/werewolf.py), second person
            card = f"You are {names[i]}. Your role is {roles[i]}."
            if mates:
                card += f" The other Werewolf{'s are' if len(mates) > 1 else ' is'} {mate_list}."
        else:  # first person, from the card owner's own perspective
            card = f"I am {names[i]}. I am a {roles[i]}."
            if mates:
                card += f" My fellow Werewol{'ves are' if len(mates) > 1 else 'f is'} {mate_list}."
            goal = GOALS[roles[i]]
            if num_werewolves > 1:
                goal = goal.replace("who the Werewolf is", "who the Werewolves are").replace(
                    "find the Werewolf.", "find the Werewolves.")
            card += " " + goal
        cards.append(card)
    turns = []
    for i in range(num_players):
        turns.append({"speaker": i, "text": f"Hi, I'm {names[i]}.", "visible_to": None})
        visible_to = [w for w in wolves if w != i] if i in wolves else []  # [] = owner only
        turns.append({"speaker": i, "text": cards[i], "visible_to": visible_to})
    turns.append({"speaker": opener, "text": opener_text, "visible_to": None})
    return rng, names, roles, wolves, opener, cards, turns


def _plain_name(peft_name):
    """``base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight`` -> the un-wrapped name."""
    name = peft_name[len("base_model.model."):] if peft_name.startswith("base_model.model.") else peft_name
    return name.replace(".base_layer.", ".")


def load_game_model(model_path, checkpoint_dir, num_players, dynamic_state, device, dtype,
                    hyperparameters=None, log=print, keep_pretrained=False):
    """Load base Qwen + checkpoint, rebuilding every saved matrix-config field.

    If ``hyperparameters`` names a JSON file, its fields override the saved
    matrix config (differences are printed so a mismatch is visible).
    ``keep_pretrained`` keeps (on CPU) the original Qwen values of the base
    layers the checkpoint overwrites, as ``model.pretrained_unfrozen``.
    Returns ``(model, tokenizer, cfg, saved_state)``; the model is in eval mode.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM
    from training.checkpoint import apply_checkpoint_to_model, load_checkpoint_state

    saved = load_checkpoint_state(checkpoint_dir)  # fail before loading the 8 GB base model
    if hyperparameters:
        with open(hyperparameters) as f:
            hp = json.load(f)
        saved_cfg = saved.get("matrix_config", {}) or {}
        diffs = {k: (saved_cfg.get(k), v) for k, v in hp.items() if k in saved_cfg and saved_cfg[k] != v}
        if diffs:
            log(f"[warn] {hyperparameters} differs from saved config (saved -> used): {diffs}")
        saved["matrix_config"] = {**saved_cfg, **hp}
        log(f"[info] hyperparameters: {hyperparameters}")
    probe_path = os.path.join(checkpoint_dir, "best_probe.json")
    if os.path.isfile(probe_path):
        with open(probe_path) as f:
            probe = json.load(f)
        log(f"[info] best_probe: score={probe.get('best_probe_score')} "
            f"epoch={probe.get('epoch')} step={probe.get('step')}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        base = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa")
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation="sdpa")
    fields = {f.name for f in dataclasses.fields(MatrixQwenConfig)} - {"base_model_name"}
    cfg = {k: v for k, v in saved.get("matrix_config", {}).items() if k in fields}
    if dynamic_state == "off":
        cfg["agent_dynamic_state_mode"] = "none"
    max_agents = cfg.get("max_agents", 8)
    if num_players > max_agents:
        raise ValueError(f"{num_players} players > checkpoint max_agents={max_agents}")
    model = MatrixQwenForCausalLM(base, MatrixQwenConfig(base_model_name=model_path, **cfg))
    # Original Qwen values of the base layers the checkpoint overwrites (keyed by
    # the checkpoint's names), for the trainer's "plain Qwen" KL reference.
    unfrozen = saved.get("unfrozen_base_parameters") or {}
    plain = dict(base.named_parameters())
    model.pretrained_unfrozen = {
        key: plain[_plain_name(key)].detach().cpu().clone()
        for key in unfrozen if _plain_name(key) in plain
    } if keep_pretrained else {}
    apply_checkpoint_to_model(model, checkpoint_dir, saved)
    # Return the full config (training fields such as max_flat_len included) for reporting.
    return model.to(device).eval(), tokenizer, {**saved.get("matrix_config", {}), **cfg}, saved


# ----------------------------------------------------------------------------
# Game phases
# ----------------------------------------------------------------------------
def score_names(table, voter, candidate_ids):
    """Total log-probability of each candidate's name continuing ``voter``'s turn."""
    first = torch.log_softmax(table.logits[0, voter].float(), dim=-1)
    scores = []
    for ids in candidate_ids:
        total = float(first[ids[0]])
        if len(ids) > 1:  # only multi-token names need a branch (a full KV-cache copy)
            branch = table.fork()
            for k, token_id in enumerate(ids[1:]):
                branch.push(voter, ids[k])
                total += float(torch.log_softmax(branch.logits[0, voter].float(), dim=-1)[token_id])
            del branch
        scores.append(total)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def continue_turn(table, speaker, seat, prefix_ids, max_tokens, max_sentences, private=False,
                  rec=None, turn=-1):
    """Let ``seat`` keep talking after already-pushed ``prefix_ids``."""
    out = []
    sentences = 0
    for _ in range(max_tokens):
        src = table.seq_len - 1
        token_id, logp, penalized, banned = speaker.sample(table.logits[0, seat], prefix_ids + out)
        if rec is not None:  # a sampled stop is an action too: it ends the turn
            rec.token(seat, src, token_id, logp, turn, penalized, banned)
        if token_id in speaker.stop_ids:
            break
        piece = speaker.tokenizer.decode([token_id])
        if "\n" in piece and not piece.strip():
            break
        table.push(seat, token_id, private=private)
        out.append(token_id)
        if "\n" in piece:
            break
        if len(out) >= 2 and any(c in piece for c in ".!?"):
            sentences += 1
            if sentences >= max_sentences:
                break
    return out


def cast_ballot(booth, voter, prefix_text, candidates, names, tokenizer, speaker, args, private=False,
                order=None, rec=None, kind="ballot", day=0):
    """One ballot: ``prefix_text`` + a candidate's name, sampled from the name probabilities.

    ``reason_first`` (default): the voter first speaks freely, then says
    ``prefix_text`` and the name -- so the name is conditioned on the reasoning.
    ``name_first``: name first, then ``because ...`` (a post-hoc explanation).
    """
    turn = rec.begin_turn(voter, booth.seq_len, kind, day) if rec is not None else -1

    def pick_name(lead):
        prefix = booth.say(voter, lead + prefix_text, private=private)
        candidate_ids = [name_token_ids(tokenizer, lead + prefix_text, names[c]) for c in candidates]
        if all(len(ids) == 1 for ids in candidate_ids):
            first = torch.log_softmax(booth.logits[0, voter].float(), dim=-1)
            scores = first[torch.tensor([ids[0] for ids in candidate_ids], device=first.device)].cpu()
        elif rec is not None:
            raise ValueError("multi-token candidate names cannot be recorded; use --name_pool single_token")
        else:
            scores = torch.tensor(score_names(booth, voter, candidate_ids))
        vt = args.vote_temperature
        if vt > 0:
            log_probs = torch.log_softmax(scores / vt, dim=0)
            pick = int(torch.multinomial(log_probs.exp(), 1))
        else:
            log_probs = torch.log_softmax(scores, dim=0)
            pick = int(scores.argmax())
        if rec is not None:
            rec.choice(voter, booth.seq_len - 1, [ids[0] for ids in candidate_ids], list(candidates),
                       pick, float(log_probs[pick]), kind, turn, day)
        for token_id in candidate_ids[pick]:
            booth.push(voter, token_id, private=private)
        probs = torch.softmax(scores, dim=0)
        return prefix + candidate_ids[pick], candidates[pick], {
            names[c]: round(float(p), 3) for c, p in zip(candidates, probs)}

    if (order or args.ballot_order) == "reason_first":
        reason = continue_turn(booth, speaker, voter, [], args.reason_tokens, args.reason_sentences,
                               private=private, rec=rec, turn=turn)
        lead = " " if reason else ""
        named, target, probs = pick_name(lead)
        tokens = reason + named + booth.say(voter, ".", private=private)
    else:
        named, target, probs = pick_name("")
        because = booth.say(voter, " because", private=private)
        reason = continue_turn(booth, speaker, voter, named + because, args.reason_tokens, 1,
                               private=private, rec=rec, turn=turn)
        tokens = named + because + reason
    speaker.remember(tokens)
    return {"voter": voter, "target": target, "probs": probs,
            "text": tokenizer.decode(tokens, skip_special_tokens=True)}


def plurality(targets):
    """Return ``(tally, leaders)``; a unique leader is ``leaders[0]``."""
    tally = Counter(t for t in targets if t is not None)
    if not tally:
        return tally, []
    top = max(tally.values())
    return tally, sorted(c for c, count in tally.items() if count == top)


def free_discussion(table, alive, speaker, tokenizer, args, quiet_columns, on_utterance, rec=None, day=0):
    """Let the model run the conversation itself.

    Every column, each living agent samples whether to speak (``bernoulli``:
    ``sigmoid((activity_logit + silence_bias) / activity_temperature)``;
    ``threshold``: P(speak) > threshold) and, if so, samples one token;
    several may overlap, and sampling a stop or blank-newline token counts as
    yielding. Dead agents are always silent.

    Silence handling: during a table-wide silence every living agent's speak
    logit gets ``silence_bias = silence_ramp * max(0, silent - silence_ramp_start)``.
    After ``rescue_after`` silent columns the agent with the highest P(speak)
    is made to speak (an environment move; its token is still sampled, with
    yield tokens banned). The ``rescue_end_after``-th such silence ends the
    discussion. With ``rescue_after == 0`` the discussion instead ends after
    ``quiet_columns`` silent columns. ``max_discussion_columns`` caps it.
    Returns ``(utterances, columns, hit_cap, rescues)``.
    """
    utterances, open_utts, open_turn = [], {}, {}
    silent = columns = rescues = 0
    hit_cap = False

    def close(seat):
        ids = open_utts.pop(seat)
        open_turn.pop(seat, None)
        speaker.remember(ids)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        utterances.append((seat, text))
        on_utterance(seat, text)

    while True:
        if args.max_discussion_columns and columns >= args.max_discussion_columns:
            hit_cap = True
            break
        src = table.seq_len - 1
        activity = table.activity[0].float()
        column = {}
        if args.rescue_after > 0 and silent >= args.rescue_after:
            rescues += 1
            if rescues >= args.rescue_end_after:
                break
            # Rescue: the most willing living agent must say something.
            seat = max(alive, key=lambda s: float(activity[s]))
            token_id, logp, penalized, banned = speaker.sample(
                table.logits[0, seat], [], extra_banned=speaker.yield_ids())
            if rec is not None:
                turn = rec.begin_turn(seat, table.seq_len, "discussion", day)
                open_turn[seat] = turn
                rec.token(seat, src, token_id, logp, turn, penalized, banned)
            column[seat] = token_id
        elif args.rescue_after <= 0 and silent >= quiet_columns:
            break
        else:
            bias = args.silence_ramp * max(0, silent - args.silence_ramp_start) if silent else 0.0
            for seat in alive:
                logit = activity[seat] + bias
                if args.speak_sampling == "bernoulli":
                    scaled = logit / args.activity_temperature
                    speak = bool(torch.rand((), device=scaled.device) < torch.sigmoid(scaled))
                    logp_speak = float(F.logsigmoid(scaled if speak else -scaled))
                else:
                    speak, logp_speak = float(torch.sigmoid(logit)) > args.activity_threshold, 0.0
                mid_turn = open_turn.get(seat, -1)
                speak_idx = (rec.speak(seat, src, int(speak), logp_speak, day, turn=mid_turn, bias=bias)
                             if rec is not None else None)
                if not speak:
                    continue
                token_id, logp, penalized, banned = speaker.sample(table.logits[0, seat], open_utts.get(seat, []))
                piece = tokenizer.decode([token_id])
                yielded = token_id in speaker.stop_ids or ("\n" in piece and not piece.strip())
                if rec is not None:
                    turn = mid_turn
                    if turn < 0 and not yielded:  # first token of a new utterance
                        turn = rec.begin_turn(seat, table.seq_len, "discussion", day)
                        open_turn[seat] = turn
                    rec.speaks[speak_idx]["turn"] = turn
                    rec.token(seat, src, token_id, logp, turn, penalized, banned)
                if not yielded:
                    column[seat] = token_id
        for seat in [s for s in open_utts if s not in column]:
            close(seat)
        for seat, token_id in column.items():
            open_utts.setdefault(seat, []).append(token_id)
        table.push_column(column)
        columns += 1
        silent = 0 if column else silent + 1
    for seat in list(open_utts):
        close(seat)
    return utterances, columns, hit_cap, rescues


def check_winner(alive, roles):
    """Village wins when every Werewolf is dead; Werewolves win at parity."""
    wolves = sum(roles[s] == "Werewolf" for s in alive)
    if wolves == 0:
        return "Village"
    if wolves >= len(alive) - wolves:
        return "Werewolves"
    return None


def play_game(model, tokenizer, args, game_seed, *, record=False, log=print, name_pool=None,
              abort_check=None):
    """Play one game. Returns a JSON-able result dict.

    The result holds the transcript (``phases``), the outcome, and ``events``
    (per-day votes and per-night actions, what the rewards are computed from).
    With ``record=True`` it also holds ``rollout``: the full game matrix and
    every sampled action (see the module docstring). ``abort_check`` is
    called before every new column; an exception it raises abandons the game.
    """
    from scripts.eval.evaluate_checkpoint_suite import _build_multiturn_inputs

    if record:
        check_record_args(args)
    n = args.num_players
    if name_pool is None:
        name_pool = single_token_names(tokenizer, args) if args.name_pool == "single_token" else RANDOM_NAME_POOL
    rng, names, roles, wolves, opener, cards, prelude = deal_roles(
        n, game_seed, args.opener, plain_cards=args.plain_cards, num_werewolves=args.num_werewolves,
        seer=args.seer, name_pool=name_pool,
    )
    # Same pause encoding as training: a pause of d seconds -> ceil(d - 1) silent columns (max 8).
    quiet_columns = max(1, _pause_columns(args.quiet_seconds, ConvertConfig()))
    device = next(model.parameters()).device
    torch.manual_seed(game_seed)
    speaker = Speaker(tokenizer, args)
    rec = Recorder() if record else None

    def label(seat):
        tag = {"Werewolf": " [WOLF]", "Seer": " [SEER]"}.get(roles[seat], "")
        return f"{names[seat]}{tag}"

    def say_line(seat, text):
        log(f"{label(seat)}: {text}")

    log("=" * 70)
    log(f"game_seed={game_seed} players={n} werewolves={len(wolves)} seer={args.seer} "
        f"discussion={args.discussion_mode} speak_sampling={args.speak_sampling}"
        f"{f' (activity_temperature={args.activity_temperature:g})' if args.speak_sampling == 'bernoulli' else ''}")
    if args.discussion_mode == "free" and args.rescue_after > 0:
        log(f"free discussion: silence ramp +{args.silence_ramp:g} logit/column after "
            f"{args.silence_ramp_start} silent column(s); rescue after {args.rescue_after}; the "
            f"{args.rescue_end_after}. such silence ends it; cap {args.max_discussion_columns or 'none'} columns/day")
    elif args.discussion_mode == "free":
        log(f"free discussion ends after {args.quiet_seconds:g}s of quiet = {quiet_columns} all-silent "
            f"column(s); cap {args.max_discussion_columns or 'none'} columns/day")
    for i in range(n):
        log(f"  agent {i}: {names[i]:<10} role card (private): {cards[i]!r}")

    inp, mask, private_mask, visibility = _build_multiturn_inputs(tokenizer, prelude, n, device)
    table = Table(model, tokenizer, inp, mask, private_mask, visibility)
    table.abort_check = abort_check  # checked before every column (e.g. yield the GPU)

    alive = list(range(n))
    seer_checked = []
    openers_used = []
    phases, day_events, night_events, discussion_spans = [], [], [], []
    deaths = {}  # seat -> {"day": d, "cause": "vote"|"kill"}
    villager_vote_probs = []  # (P(vote for a living Werewolf), chance) per Villager-team ballot
    killed = None
    winner = None
    day = 0

    def set_alive():
        if rec is not None:
            rec.alive = list(alive)

    while not args.max_days or day < args.max_days:
        day += 1
        set_alive()
        # ---- Day ----
        if day == 1:
            day_opener = opener  # its opener line is already in the prelude
            opening = args.opener
        else:
            fresh = [s for s in alive if s not in openers_used]
            day_opener = rng.choice(fresh or [s for s in alive if s != openers_used[-1]] or alive)
            opening = args.death_announcement.format(dead=names[killed]) + " " + args.opener
            table.say(day_opener, opening)
        openers_used.append(day_opener)
        log(f"\n--- day {day} (alive: {', '.join(names[s] for s in alive)}) ---")
        say_line(day_opener, opening)

        seat_order = [s for s in ((day_opener + 1 + k) % n for k in range(n)) if s in alive]  # opener last
        discussion, discussion_columns, hit_cap, rescues = [], 0, False, 0
        span_start = table.seq_len
        if args.discussion_mode == "free":
            discussion, discussion_columns, hit_cap, rescues = free_discussion(
                table, alive, speaker, tokenizer, args, quiet_columns, say_line, rec=rec, day=day)
            ended = ("hit the column cap" if hit_cap else
                     f"ended at the {args.rescue_end_after}. long silence" if args.rescue_after > 0 else
                     f"ended after {args.quiet_seconds:g}s of quiet")
            log(f"[discussion: {discussion_columns} columns, {rescues if args.rescue_after > 0 else 0} "
                f"silence rescue(s), {ended}]")
        else:
            for _ in range(args.discussion_rounds):
                for seat in seat_order:
                    turn = rec.begin_turn(seat, table.seq_len, "discussion", day) if rec is not None else -1
                    prefix = table.say(seat, rng.choice(args.starters)) if args.starters else []
                    out = continue_turn(table, speaker, seat, prefix, args.turn_tokens, args.max_sentences,
                                        rec=rec, turn=turn)
                    speaker.remember(prefix + out)
                    text = tokenizer.decode(prefix + out, skip_special_tokens=True)
                    discussion.append((seat, text))
                    say_line(seat, text)
        discussion_spans.append({"day": day, "start": span_start, "end": table.seq_len,
                                 "opener": day_opener, "free": args.discussion_mode == "free"})

        # Day vote: every living player must name another living player.
        table.say(day_opener, args.vote_prompt)
        say_line(day_opener, args.vote_prompt)
        ballot_box = table.fork() if args.blind_votes else table
        votes = []
        alive_at_vote = list(alive)
        for voter in seat_order:
            booth = ballot_box.fork() if args.blind_votes else table
            candidates = [c for c in alive if c != voter]
            v = cast_ballot(booth, voter, args.vote_prefix, candidates, names, tokenizer, speaker, args,
                            rec=rec, kind="ballot", day=day)
            votes.append(v)
            say_line(voter, v["text"])
            log(f"    P(vote): {v['probs']}")
            if roles[voter] != "Werewolf":  # Villager team (Villagers and the Seer)
                living_wolves = [c for c in candidates if roles[c] == "Werewolf"]
                villager_vote_probs.append((sum(v["probs"][names[c]] for c in living_wolves),
                                            len(living_wolves) / len(candidates)))
        tally, leaders = plurality(v["target"] for v in votes)
        voted_out = leaders[0] if len(leaders) == 1 else rng.choice(leaders)
        if len(leaders) == 1:
            result = args.vote_result.format(out=names[voted_out])
        else:
            tied = " and ".join(names[c] for c in leaders)
            result = args.vote_tie.format(tied=tied, out=names[voted_out])
        table.say(day_opener, result)
        log("tally: " + ", ".join(f"{names[c]}={tally[c]}" for c in sorted(tally, key=lambda c: -tally[c])))
        say_line(day_opener, result)
        alive.remove(voted_out)
        deaths[voted_out] = {"day": day, "cause": "vote"}
        set_alive()
        log(f"[god] {names[voted_out]} was a {roles[voted_out]}")
        day_events.append({"day": day, "alive": alive_at_vote,
                           "votes": [[v["voter"], v["target"]] for v in votes],
                           "voted_out": voted_out, "tie": len(leaders) > 1})
        phases.append({
            "phase": "day", "day": day, "opener": names[day_opener], "opening": opening,
            "discussion": [{"speaker": label(s), "text": t} for s, t in discussion],
            "discussion_columns": discussion_columns, "discussion_hit_cap": hit_cap,
            "discussion_rescues": max(0, rescues - (0 if hit_cap else 1)) if args.rescue_after > 0 else 0,
            "vote_prompt": args.vote_prompt,
            "votes": [{"voter": names[v["voter"]], "voter_role": roles[v["voter"]],
                       "target": names[v["target"]], "text": v["text"], "probs": v["probs"]} for v in votes],
            "tally": {names[c]: count for c, count in tally.items()},
            "tie": [names[c] for c in leaders] if len(leaders) > 1 else None,
            "voted_out": names[voted_out], "voted_out_role": roles[voted_out],
            "result_text": result,
        })
        winner = check_winner(alive, roles)
        if winner or day == args.max_days:
            break

        # ---- Night: the Seer looks, then the living Werewolves pick a kill ----
        night_alive = list(alive)
        look = None
        seer = next((s for s in alive if roles[s] == "Seer"), None)
        if seer is not None:
            options = [s for s in alive if s != seer and s not in seer_checked] or [s for s in alive if s != seer]
            prompt = args.seer_prompt.format(night=day)
            table.say(seer, prompt, private=True)
            log(f"\n--- night {day}: seer (private to {names[seer]}) ---")
            say_line(seer, prompt)
            v = cast_ballot(table, seer, args.seer_prefix, options, names, tokenizer, speaker, args,
                            private=True, order="name_first", rec=rec, kind="seer", day=day)
            target = v["target"]
            new_info = target not in seer_checked
            seen = "a Werewolf" if roles[target] == "Werewolf" else "a Villager"
            info = args.seer_result.format(target=names[target], seen=seen)
            table.say(seer, " " + info, private=True)
            seer_checked.append(target)
            say_line(seer, v["text"])
            log(f"    P(look): {v['probs']}")
            say_line(seer, info)
            look = {"seer": names[seer], "prompt": prompt, "target": names[target], "text": v["text"],
                    "probs": v["probs"], "info": info, "new_info": new_info}

        night_wolves = [s for s in alive if roles[s] == "Werewolf"]
        targets = [s for s in alive if roles[s] != "Werewolf"]
        night_text = args.night_prompt.format(night=day)
        table.say(night_wolves[0], night_text, private=True)
        log(f"\n--- night {day} (private to Werewolves) ---")
        say_line(night_wolves[0], night_text)
        kill_votes = []
        for wolf in night_wolves:
            v = cast_ballot(table, wolf, args.kill_prefix, targets, names, tokenizer, speaker, args,
                            private=True, rec=rec, kind="kill", day=day)
            kill_votes.append(v)
            say_line(wolf, v["text"])
            log(f"    P(kill): {v['probs']}")
        kill_tally, kill_leaders = plurality(v["target"] for v in kill_votes)
        killed = kill_leaders[0] if len(kill_leaders) == 1 else rng.choice(kill_leaders)
        alive.remove(killed)
        deaths[killed] = {"day": day, "cause": "kill"}
        set_alive()
        log(f"[god] the Werewolves killed {names[killed]}"
            f"{' (tie broken at random)' if len(kill_leaders) > 1 else ''}")
        night_events.append({
            "night": day, "alive": night_alive,
            "seer": ([seer, look and seer_checked[-1], look["new_info"]] if look else None),
            "kill_votes": [[v["voter"], v["target"]] for v in kill_votes], "killed": killed,
        })
        phases.append({
            "phase": "night", "night": day, "seer": look, "prompt": night_text,
            "kill_votes": [{"voter": names[v["voter"]], "target": names[v["target"]],
                            "text": v["text"], "probs": v["probs"]} for v in kill_votes],
            "tally": {names[c]: count for c, count in kill_tally.items()},
            "killed": names[killed], "tie_broken_at_random": len(kill_leaders) > 1,
        })
        winner = check_winner(alive, roles)
        if winner:
            break

    p_wolf = sum(p for p, _ in villager_vote_probs) / max(len(villager_vote_probs), 1)
    chance = sum(c for _, c in villager_vote_probs) / max(len(villager_vote_probs), 1)
    log("\n--- result ---")
    log(f"days played: {day} | alive: {', '.join(label(s) for s in alive)}")
    log(f"WINNER: {winner.upper() + ' WIN' if winner else f'nobody (stopped at --max_days {args.max_days})'}")
    log(f"Villagers' mean P(vote for a living Werewolf) = {p_wolf:.3f} (chance = {chance:.3f}) "
        f"over {len(villager_vote_probs)} ballots")
    log(f"matrix length: {table.seq_len} columns")

    result = {
        "game_seed": game_seed, "names": names, "roles": roles, "werewolves": [names[w] for w in wolves],
        "role_cards": cards, "phases": phases, "winner": winner, "days_played": day,
        "alive_at_end": [names[s] for s in alive],
        "villager_mean_p_vote_wolf": p_wolf, "villager_vote_chance": chance,
        "matrix_columns": table.seq_len,
        "events": {"days": day_events, "nights": night_events, "deaths": deaths,
                   "discussion_spans": discussion_spans, "num_players": n,
                   "roles": roles, "winner": winner, "names": names},
    }
    if record:
        input_ids, activity, private = table.matrix()
        result["rollout"] = {
            "input_ids": input_ids, "activity": activity, "private": private,
            "visibility": visibility.detach().cpu() if visibility is not None else None,
            **rec.finalize(),
        }
    return result
