"""Play one Werewolf game between MatrixChat agents using only role tokens.

Every player gets the prelude the Werewolf training source uses
(``data/adapters/werewolf.py``): a PUBLIC ``Hi, I'm {name}.`` followed by a
PRIVATE role card that only that player can attend to. ``--num_werewolves``
random seats are Werewolves (they can see each other's cards) and everyone
else is a Villager.

Game loop: Day 1 -> Night 1 -> Day 2 -> ... until a side wins (or ``--max_days``
if set). Village wins when every Werewolf is dead; Werewolves win when every
Villager is dead. Role cards are first person ("I am X. I am a Villager.")
unless ``--plain_cards``. Dead players (voted out or killed) never speak,
vote, or appear as candidates again. Each day is opened by a different living
player; from Day 2 on the opener first names who was killed in the night.

Night
    The living Werewolves privately (cells only Werewolves can attend to) hear
    ``--night_prompt`` and each casts a kill ballot ``Tonight I want to kill
    <name> because ...`` over the living Villagers, scored like the day vote.
    Later Werewolves see earlier ballots. Plurality wins; ties are random.

Day discussion
    ``free`` (default): the model runs the conversation itself. Every column,
    each living player whose activity head says P(speak) > ``--activity_threshold``
    speaks one token (overlap and interruptions allowed; dead players are always
    silent). Talk continues as long as anyone wants to speak and ends once
    nobody has spoken for ``--quiet_seconds`` (default 3 s), converted to
    all-silent columns exactly as the training data encodes pauses
    (``data/convert.py``: d seconds -> ceil(d - 1) columns, so 3 s = 2 columns).
    ``--max_discussion_columns`` is a safety cap.
    ``turns``: players speak in seat order for ``--discussion_rounds`` rounds,
    each turn up to ``--max_sentences`` sentences.

Seer (``--seer``; off by default)
    One non-Werewolf is the Seer. Each day before the vote (``--seer_timing
    before_vote``; or ``before_discussion``), the Seer privately picks a living
    player it hasn't looked at (``I want to look at <name> because ...``, the
    name scored like a vote), then privately learns ``I looked at <name>'s
    card: <name> is a Werewolf / a Villager.`` Only the Seer can see any of it.

Vote
    Each ballot (``--ballot_order reason_first``, default): the player first
    speaks freely (up to ``--reason_sentences``), then says ``I think the
    Werewolf is`` (``--vote_prefix``) and the model scores each other living
    player's name as the continuation -- so the accusation follows from the
    player's own reasoning. The vote is sampled from that distribution
    (``--vote_temperature 0`` = most likely name). ``name_first`` restores the
    old ``<name> because ...`` order, where the reason is written after the
    pick. ``--no_copy_ngram`` stops any player repeating a span from an
    earlier turn by anyone, so players cannot parrot each other's lines. Votes are public and
    sequential unless ``--blind_votes``. A unique plurality eliminates that
    player (announced publicly, role not revealed); a tie eliminates nobody.

Everything happens in one shared matrix through the model's KV cache. There is
no game engine -- only prompt tokens, forced speaking turns, and per-turn
visibility. The checkpoint is rebuilt from EVERY saved matrix-config field
(unlike ``turn_taking_probe.load_model``), so GRU checkpoints load whole.

Usage:
    python scripts/eval/werewolf_game.py --checkpoint_dir checkpoints/checkpoint \
        --hyperparameters hyperparameters/adaptcleanv3_cand_00003.json \
        --game_seed 7 --output logs/werewolf_games/game_7.json
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import random
import sys
from collections import Counter

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.adapters._names import RANDOM_NAME_POOL
from data.convert import ConvertConfig, _pause_columns
from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM
from scripts.eval.evaluate_checkpoint_suite import _build_multiturn_inputs
from scripts.eval.turn_taking_probe import resolve_device
from training.checkpoint import apply_checkpoint_to_model, load_checkpoint_state

GOALS = {
    "Werewolf": "I must not let anyone find out.",
    "Villager": "I need to work out who the Werewolf is.",
    "Seer": "Each day before the vote I can secretly look at one player's card. "
            "I need to use what I learn to find the Werewolf.",
}
# The name slot must read as an accusation. "I vote for X" reads to the model as
# support ("...because I don't think she's a werewolf"), so voters picked people
# they trusted and the eliminated seat was rarely the Werewolf.
VOTE_PREFIX = "I think the Werewolf is"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", default="checkpoints/checkpoint",
                   help="Directory holding model_state.pt, lora_adapters/ and (optionally) best_probe.json.")
    p.add_argument("--hyperparameters", default="hyperparameters/adaptcleanv3_cand_00003.json",
                   help="Training hyperparameter JSON; its matrix-config fields override the ones saved "
                        "in model_state.pt ('' = use only the saved config).")
    p.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--num_players", type=int, default=6)
    p.add_argument("--game_seed", type=int, default=0,
                   help="Decides names, roles, the opener, starters, and sampling.")
    p.add_argument("--plain_cards", action="store_true",
                   help="Use the exact training card format ('You are X. Your role is Y.'), without the "
                        "goal. Default cards are first person: 'I am X. I am a Villager. ...'.")
    p.add_argument("--num_werewolves", type=int, default=1)
    p.add_argument("--max_days", type=int, default=0,
                   help="0 (default) = play until a side wins; every night kills a Villager, so the "
                        "game always ends. N > 0 stops after N days (1 = single day, no night).")
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
    # Seer (off by default: not part of the base game yet).
    p.add_argument("--seer", action="store_true",
                   help="Make one non-Werewolf the Seer: each day, before the vote, it privately picks a "
                        "living player, explains why, and learns whether that player is a Werewolf.")
    p.add_argument("--seer_timing", choices=["before_vote", "before_discussion"], default="before_vote",
                   help="When in the day the Seer looks: after the discussion, right before the vote "
                        "(default), or at the start of the day so it can use what it learns in discussion.")
    p.add_argument("--seer_prompt", default="It's day {day}. As the Seer, I can secretly look at one player's card now.",
                   help="Private to the Seer; said in the Seer's row.")
    p.add_argument("--seer_prefix", default="I want to look at",
                   help="Prefilled start of the Seer's private pick; the scored name, then 'because ...'.")
    p.add_argument("--seer_result", default="I looked at {target}'s card: {target} is {seen}.",
                   help="Private to the Seer; {seen} is 'a Werewolf' or 'a Villager'.")
    # Discussion.
    p.add_argument("--discussion_mode", choices=["free", "turns"], default="free",
                   help="free (default): the model's activity head decides who speaks each column "
                        "(interruptions/overlap allowed) until everyone is quiet for --quiet_seconds. "
                        "turns: forced seat order for --discussion_rounds rounds.")
    p.add_argument("--quiet_seconds", type=float, default=3.0,
                   help="free mode: end the discussion after this long with nobody speaking. Converted "
                        "to all-silent columns exactly as training data encodes pauses (data/convert.py).")
    p.add_argument("--max_discussion_columns", type=int, default=600,
                   help="free mode: safety cap on columns per day's discussion (0 = no cap).")
    p.add_argument("--discussion_rounds", type=int, default=2, help="turns mode: times around the table.")
    p.add_argument("--turn_tokens", type=int, default=40, help="turns mode: max generated tokens per turn.")
    p.add_argument("--max_sentences", type=int, default=2, help="turns mode: end a turn after this many.")
    p.add_argument("--starters", nargs="*", default=[],
                   help="turns mode: optional prefilled openings, one chosen at random per turn. "
                        "Default none: the model writes every turn from its first token.")
    p.add_argument("--activity_threshold", type=float, default=0.5,
                   help="free mode: an agent speaks in a column when P(speak) exceeds this.")
    # Vote.
    p.add_argument("--vote_prompt",
                   default="Alright, time to vote. Say who you think the Werewolf is and why. "
                           "Whoever gets the most votes is out.")
    p.add_argument("--vote_prefix", default=VOTE_PREFIX,
                   help="Prefilled start of every ballot; the scored name follows it, so it must "
                        "name the accused (e.g. 'I think the Werewolf is').")
    p.add_argument("--vote_temperature", type=float, default=1.0,
                   help="Sample the vote from the name probabilities at this temperature (0 = argmax).")
    p.add_argument("--ballot_order", choices=["reason_first", "name_first"], default="reason_first",
                   help="reason_first: the voter reasons freely, THEN names a suspect (the name is "
                        "conditioned on the reasoning). name_first: name, then a post-hoc 'because ...'.")
    p.add_argument("--reason_tokens", type=int, default=40, help="Max generated tokens of ballot reasoning.")
    p.add_argument("--reason_sentences", type=int, default=2, help="reason_first: max sentences of reasoning.")
    p.add_argument("--blind_votes", action="store_true",
                   help="Each player votes from the discussion alone, without hearing earlier votes.")
    p.add_argument("--vote_result",
                   default="{out} got the most votes, so {out} is out of the game.",
                   help="Said publicly by the day's opener after the vote (roles are not revealed).")
    p.add_argument("--vote_tie", default="It's a tie, so nobody is out today.")
    p.add_argument("--vote_skip", choices=["all", "villagers", "off"], default="all",
                   help="Who may skip a day vote. 'all' (default) so a skip doesn't reveal a Villager "
                        "(if only Villagers could skip, every skipper would be provably innocent).")
    p.add_argument("--skip_prefix", default="I'm not sure who the Werewolf is",
                   help="Compared against --vote_prefix by model probability to decide a skip.")
    p.add_argument("--skip_end", default=", so I'm skipping my vote.")
    p.add_argument("--skip_bias", type=float, default=0.0,
                   help="Added to the skip option's log-probability (>0 = skip more often).")
    p.add_argument("--vote_all_skipped", default="Everyone skipped, so nobody is out today.")
    # Sampling for forced turns.
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.3)
    p.add_argument("--no_repeat_ngram", type=int, default=3, help="Ban repeating n-grams within a turn.")
    p.add_argument("--no_copy_ngram", type=int, default=5,
                   help="Ban repeating any n-gram from an earlier model-written turn by ANY player "
                        "(stops players parroting each other); 0 disables.")
    p.add_argument("--dynamic_state", choices=["on", "off"], default="on",
                   help="off drops the GRU, matching the repo's load_model behaviour.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    return p.parse_args(argv)


def deal_roles(num_players, game_seed, opener_text, plain_cards=False, num_werewolves=1, seer=False):
    """Return ``(rng, names, roles, wolves, opener, cards, turns)`` for one game.

    Werewolf cards are visible to every Werewolf (and so is all later private
    night talk); Villager and Seer cards are visible to their owner only.
    """
    if not 1 <= num_werewolves < num_players / 2:
        raise ValueError(f"need 1 <= num_werewolves < num_players/2, got {num_werewolves} of {num_players}")
    rng = random.Random(game_seed)
    names = rng.sample(RANDOM_NAME_POOL, num_players)
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


def load_game_model(model_path, checkpoint_dir, num_players, dynamic_state, device, dtype,
                    hyperparameters=None):
    """Load base Qwen + checkpoint, rebuilding every saved matrix-config field.

    If ``hyperparameters`` names a JSON file, its fields override the saved
    matrix config (differences are printed so a mismatch is visible).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    saved = load_checkpoint_state(checkpoint_dir)  # fail before loading the 8 GB base model
    if hyperparameters:
        with open(hyperparameters) as f:
            hp = json.load(f)
        saved_cfg = saved.get("matrix_config", {}) or {}
        diffs = {k: (saved_cfg.get(k), v) for k, v in hp.items() if k in saved_cfg and saved_cfg[k] != v}
        if diffs:
            print(f"[warn] {hyperparameters} differs from saved config (saved -> used): {diffs}")
        saved["matrix_config"] = {**saved_cfg, **hp}
        print(f"[info] hyperparameters: {hyperparameters}")
    probe_path = os.path.join(checkpoint_dir, "best_probe.json")
    if os.path.isfile(probe_path):
        with open(probe_path) as f:
            probe = json.load(f)
        print(f"[info] best_probe: score={probe.get('best_probe_score')} "
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
    apply_checkpoint_to_model(model, checkpoint_dir, saved)
    # Return the full config (training fields such as max_flat_len included) for reporting.
    return model.to(device).eval(), tokenizer, {**saved.get("matrix_config", {}), **cfg}


class Table:
    """The shared game matrix, extended one column at a time through the KV cache.

    ``logits[0, a]`` holds agent ``a``'s next-content logits and
    ``activity[0, a]`` its next-column speak logit, given everything so far
    (exactly what ``model.generation.generate_matrix`` reads).
    """

    def __init__(self, model, tokenizer, input_ids, activity, private, visibility):
        self.model, self.tokenizer = model, tokenizer
        self.num_agents = input_ids.shape[1]
        self.device = input_ids.device
        self.cache, self.logits, self.activity = model.encode_prefix_for_generation(
            input_ids=input_ids, input_activity_mask=activity,
            input_private_mask=private, agent_visibility=visibility,
        )

    def push_column(self, tokens, private=False):
        """Append one column: ``tokens`` maps seat -> token id for every agent
        speaking in it (empty = an all-silent column, i.e. a pause). ``private``
        cells are seen only by agents the prelude let see the speaker's private cells."""
        ids = torch.zeros((1, self.num_agents, 1), dtype=torch.long, device=self.device)
        active = torch.zeros((1, self.num_agents, 1), dtype=torch.bool, device=self.device)
        for seat, token_id in tokens.items():
            ids[0, seat, 0] = int(token_id)
            active[0, seat, 0] = True
        priv = active & bool(private)
        self.cache, self.logits, self.activity = self.model.forward_generation_column(
            self.cache, ids, active, column_private=priv,
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
        return twin


class Speaker:
    """Sampling for forced speaking turns (repetition penalty, n-gram ban, top-p)."""

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

    def sample(self, logits, history):
        args = self.args
        logits = logits.float().clone()
        m = args.no_copy_ngram
        if m >= 2 and len(history) >= m - 1:
            copied = self.copy_index.get(tuple(history[-(m - 1):]))
            if copied:
                logits[list(copied)] = float("-inf")
        if self.vocab_limit is not None and self.vocab_limit < logits.shape[-1]:
            logits[self.vocab_limit:] = float("-inf")
        if args.repetition_penalty != 1.0 and history:
            idx = torch.tensor(sorted(set(history)), device=logits.device)
            seen = logits[idx]
            logits[idx] = torch.where(seen > 0, seen / args.repetition_penalty, seen * args.repetition_penalty)
        n = args.no_repeat_ngram
        if n > 1 and len(history) >= n - 1:
            tail = tuple(history[-(n - 1):])
            banned = [history[i + n - 1] for i in range(len(history) - n + 1)
                      if tuple(history[i:i + n - 1]) == tail]
            if banned:
                logits[banned] = float("-inf")
        if args.temperature <= 0:
            return int(logits.argmax())
        probs = torch.softmax(logits / args.temperature, dim=-1)
        if args.top_p < 1.0:
            sorted_probs, order = probs.sort(descending=True)
            outside = sorted_probs.cumsum(-1) - sorted_probs > args.top_p
            sorted_probs[outside] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, order, sorted_probs)
        return int(torch.multinomial(probs, 1))

    def continue_turn(self, table, speaker, prefix_ids, max_tokens, max_sentences, private=False):
        """Let ``speaker`` keep talking after already-pushed ``prefix_ids``."""
        out = []
        sentences = 0
        for _ in range(max_tokens):
            token_id = self.sample(table.logits[0, speaker], prefix_ids + out)
            if token_id in self.stop_ids:
                break
            piece = self.tokenizer.decode([token_id])
            if "\n" in piece and not piece.strip():
                break
            table.push(speaker, token_id, private=private)
            out.append(token_id)
            if "\n" in piece:
                break
            if len(out) >= 2 and any(c in piece for c in ".!?"):
                sentences += 1
                if sentences >= max_sentences:
                    break
        return out


def name_token_ids(tokenizer, prefix, name):
    """Token ids of `` name`` as it would follow ``prefix`` in the same turn."""
    pre = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full = tokenizer(f"{prefix} {name}", add_special_tokens=False)["input_ids"]
    if full[:len(pre)] != pre:
        raise ValueError(f"tokenizer merges {prefix!r} with {name!r}; cannot score the name cleanly")
    return full[len(pre):]


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


def cast_ballot(booth, voter, prefix_text, candidates, names, tokenizer, speaker, args, private=False,
                allow_skip=False, order=None):
    """One ballot: ``prefix_text`` + a candidate's name, sampled from the name probabilities.

    ``reason_first`` (default): the voter first speaks freely, then says
    ``prefix_text`` and the name -- so the name is conditioned on the reasoning.
    ``name_first``: name first, then ``because ...`` (a post-hoc explanation).
    ``allow_skip``: before naming anyone, the model's own probability of
    ``prefix_text`` vs ``--skip_prefix`` ("I'm not sure who the Werewolf is")
    decides whether the voter accuses someone or skips (``target`` None).
    """
    def decide_skip(lead):
        options = [tokenizer(lead + text, add_special_tokens=False)["input_ids"]
                   for text in (prefix_text, args.skip_prefix)]
        scores = torch.tensor(score_names(booth, voter, options))
        scores[1] += args.skip_bias
        p_skip = float(torch.softmax(scores, dim=0)[1])
        if args.vote_temperature > 0:
            skip = int(torch.multinomial(torch.softmax(scores / args.vote_temperature, dim=0), 1)) == 1
        else:
            skip = bool(scores[1] > scores[0])
        return skip, round(p_skip, 3)

    def say_skip(lead):
        return booth.say(voter, lead + args.skip_prefix + args.skip_end, private=private)

    def pick_name(lead):
        prefix = booth.say(voter, lead + prefix_text, private=private)
        candidate_ids = [name_token_ids(tokenizer, lead + prefix_text, names[c]) for c in candidates]
        scores = torch.tensor(score_names(booth, voter, candidate_ids))
        if args.vote_temperature > 0:
            pick = int(torch.multinomial(torch.softmax(scores / args.vote_temperature, dim=0), 1))
        else:
            pick = int(scores.argmax())
        for token_id in candidate_ids[pick]:
            booth.push(voter, token_id, private=private)
        probs = torch.softmax(scores, dim=0)
        return prefix + candidate_ids[pick], candidates[pick], {
            names[c]: round(float(p), 3) for c, p in zip(candidates, probs)}

    skip, p_skip = False, None
    if (order or args.ballot_order) == "reason_first":
        reason = speaker.continue_turn(booth, voter, [], args.reason_tokens, args.reason_sentences,
                                       private=private)
        lead = " " if reason else ""
        if allow_skip:
            skip, p_skip = decide_skip(lead)
        if skip:
            target, probs = None, {}
            tokens = reason + say_skip(lead)
        else:
            named, target, probs = pick_name(lead)
            tokens = reason + named + booth.say(voter, ".", private=private)
    else:
        if allow_skip:
            skip, p_skip = decide_skip("")
        if skip:
            target, probs = None, {}
            tokens = say_skip("")
        else:
            named, target, probs = pick_name("")
            because = booth.say(voter, " because", private=private)
            reason = speaker.continue_turn(booth, voter, named + because, args.reason_tokens, 1,
                                           private=private)
            tokens = named + because + reason
    speaker.remember(tokens)
    return {"voter": voter, "target": target, "probs": probs, "p_skip": p_skip,
            "text": tokenizer.decode(tokens, skip_special_tokens=True)}


def plurality(targets):
    """Return ``(tally, leaders)``; a unique leader is ``leaders[0]``. Skips (None) don't count."""
    tally = Counter(t for t in targets if t is not None)
    if not tally:
        return tally, []
    top = max(tally.values())
    return tally, [c for c, count in tally.items() if count == top]


def free_discussion(table, alive, speaker, tokenizer, args, quiet_columns, on_utterance):
    """Let the model run the conversation itself until everyone goes quiet.

    Every column, each living agent whose P(speak) (the activity head) exceeds
    ``--activity_threshold`` speaks one sampled token; several may overlap, and
    sampling a stop/newline token counts as yielding. Dead agents are always
    silent. Silent columns are real pauses in the matrix. The discussion ends
    after ``quiet_columns`` consecutive all-silent columns (or the column cap).
    Returns ``(utterances, columns, hit_cap)`` with utterances in the order
    they finished.
    """
    utterances, open_utts = [], {}  # open_utts: seat -> token ids of the current utterance
    quiet = columns = 0

    def close(seat):
        ids = open_utts.pop(seat)
        speaker.remember(ids)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        utterances.append((seat, text))
        on_utterance(seat, text)

    while quiet < quiet_columns:
        if args.max_discussion_columns and columns >= args.max_discussion_columns:
            break
        p_speak = torch.sigmoid(table.activity[0].float())
        column = {}
        for seat in alive:
            if float(p_speak[seat]) <= args.activity_threshold:
                continue
            token_id = speaker.sample(table.logits[0, seat], open_utts.get(seat, []))
            piece = tokenizer.decode([token_id])
            if token_id in speaker.stop_ids or ("\n" in piece and not piece.strip()):
                continue  # chose to yield
            column[seat] = token_id
        for seat in [s for s in open_utts if s not in column]:
            close(seat)
        for seat, token_id in column.items():
            open_utts.setdefault(seat, []).append(token_id)
        table.push_column(column)
        columns += 1
        quiet = 0 if column else quiet + 1
    hit_cap = quiet < quiet_columns
    for seat in list(open_utts):
        close(seat)
    return utterances, columns, hit_cap


def seer_look(table, seer, alive, checked, names, roles, tokenizer, speaker, args, day):
    """The Seer privately picks a living player, explains why, then learns their role."""
    options = [s for s in alive if s != seer and s not in checked] or [s for s in alive if s != seer]
    prompt = args.seer_prompt.format(day=day)
    table.say(seer, prompt, private=True)
    v = cast_ballot(table, seer, args.seer_prefix, options, names, tokenizer, speaker, args,
                    private=True, order="name_first")
    target = v["target"]
    seen = "a Werewolf" if roles[target] == "Werewolf" else "a Villager"
    info = args.seer_result.format(target=names[target], seen=seen)
    table.say(seer, " " + info, private=True)
    checked.append(target)
    return {"seer": names[seer], "prompt": prompt, "target": names[target], "text": v["text"],
            "probs": v["probs"], "info": info}


def check_winner(alive, roles):
    """Village wins when every Werewolf is dead; Werewolves win when every Villager is."""
    wolves = sum(roles[s] == "Werewolf" for s in alive)
    if wolves == 0:
        return "Village"
    if wolves == len(alive):
        return "Werewolves"
    return None


def main(argv=None):
    args = parse_args(argv)
    n = args.num_players
    rng, names, roles, wolves, opener, cards, turns = deal_roles(
        n, args.game_seed, args.opener, plain_cards=args.plain_cards, num_werewolves=args.num_werewolves,
        seer=args.seer,
    )
    # Same pause encoding as training: a pause of d seconds -> ceil(d - 1) silent columns (max 8).
    quiet_columns = max(1, _pause_columns(args.quiet_seconds, ConvertConfig()))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"--device {args.device} requested but CUDA is unavailable (torch {torch.__version__} "
            f"from {torch.__file__}, built for CUDA {torch.version.cuda}). A 4B game on CPU will "
            "not finish; pass --device cpu to force it."
        )
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tokenizer, cfg = load_game_model(
        args.model_path, args.checkpoint_dir, n, args.dynamic_state, device, dtype,
        hyperparameters=args.hyperparameters,
    )
    torch.manual_seed(args.game_seed)
    speaker = Speaker(tokenizer, args)
    dynamic = cfg.get("agent_dynamic_state_mode", "none")

    def label(seat):
        if isinstance(seat, int):
            tag = {"Werewolf": " [WOLF]", "Seer": " [SEER]"}.get(roles[seat], "")
            return f"{names[seat]}{tag}"
        return "OVERLAP(" + ", ".join(names[int(a)] for a in seat.split(":")[1].split(",")) + ")"

    def say_line(seat, text):
        print(f"{label(seat)}: {text}", flush=True)

    # God-view header.
    print("=" * 70)
    print(f"game_seed={args.game_seed} checkpoint={args.checkpoint_dir} players={n} "
          f"werewolves={len(wolves)} seer={args.seer} discussion={args.discussion_mode} "
          f"dynamic_state={dynamic} blind_votes={args.blind_votes}")
    if args.discussion_mode == "free":
        print(f"free discussion ends after {args.quiet_seconds:g}s of quiet = {quiet_columns} all-silent "
              f"column(s); cap {args.max_discussion_columns or 'none'} columns/day")
    for i in range(n):
        print(f"  agent {i}: {names[i]:<10} role card (private): {cards[i]!r}")

    # Prelude: names, role cards, day-1 opener.
    inp, mask, private_mask, visibility = _build_multiturn_inputs(tokenizer, turns, n, device)
    table = Table(model, tokenizer, inp, mask, private_mask, visibility)

    def seer_turn(day):
        seer = next((s for s in alive if roles[s] == "Seer"), None)
        if seer is None:
            return None
        print(f"--- seer (private to {names[seer]}) ---")
        look = seer_look(table, seer, alive, seer_checked, names, roles, tokenizer, speaker, args, day)
        say_line(seer, look["prompt"])
        say_line(seer, look["text"])
        print(f"    P(look): {look['probs']}")
        say_line(seer, look["info"])
        return look

    alive = list(range(n))
    seer_checked = []
    openers_used = []
    phases = []
    villager_vote_probs = []  # (P(vote for a living Werewolf), chance) per non-skipped Villager ballot
    skips = []  # per Villager day ballot: True if skipped
    killed = None
    winner = None
    day = 0
    while not args.max_days or day < args.max_days:
        day += 1
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
        print(f"\n--- day {day} (alive: {', '.join(names[s] for s in alive)}) ---")
        say_line(day_opener, opening)

        seat_order = [s for s in ((day_opener + 1 + k) % n for k in range(n)) if s in alive]  # opener last
        look = seer_turn(day) if args.seer_timing == "before_discussion" else None
        if look:
            print("--- discussion ---")
        discussion, discussion_columns, hit_cap = [], 0, False
        if args.discussion_mode == "free":
            discussion, discussion_columns, hit_cap = free_discussion(
                table, alive, speaker, tokenizer, args, quiet_columns, say_line)
            print(f"[discussion: {discussion_columns} columns, "
                  f"{'hit the column cap' if hit_cap else f'ended after {args.quiet_seconds:g}s of quiet'}]")
        else:
            for _ in range(args.discussion_rounds):
                for seat in seat_order:
                    prefix = table.say(seat, rng.choice(args.starters)) if args.starters else []
                    out = speaker.continue_turn(table, seat, prefix, args.turn_tokens, args.max_sentences)
                    speaker.remember(prefix + out)
                    text = tokenizer.decode(prefix + out, skip_special_tokens=True)
                    discussion.append((seat, text))
                    say_line(seat, text)

        if args.seer_timing == "before_vote":
            look = seer_turn(day)

        # Day vote.
        table.say(day_opener, args.vote_prompt)
        say_line(day_opener, args.vote_prompt)
        ballot_box = table.fork() if args.blind_votes else table
        votes = []
        for voter in seat_order:
            booth = ballot_box.fork() if args.blind_votes else table
            candidates = [c for c in alive if c != voter]
            allow_skip = args.vote_skip == "all" or (args.vote_skip == "villagers" and roles[voter] != "Werewolf")
            v = cast_ballot(booth, voter, args.vote_prefix, candidates, names, tokenizer, speaker, args,
                            allow_skip=allow_skip)
            votes.append(v)
            say_line(voter, v["text"])
            print(f"    P(skip): {v['p_skip']}" if v["target"] is None
                  else f"    P(vote): {v['probs']}" + (f" | P(skip): {v['p_skip']}" if allow_skip else ""))
            if roles[voter] != "Werewolf":  # Villager team (Villagers and the Seer)
                skips.append(v["target"] is None)
                if v["target"] is not None:
                    living_wolves = [c for c in candidates if roles[c] == "Werewolf"]
                    villager_vote_probs.append((sum(v["probs"][names[c]] for c in living_wolves),
                                                len(living_wolves) / len(candidates)))
        tally, leaders = plurality(v["target"] for v in votes)
        voted_out = leaders[0] if len(leaders) == 1 else None
        if voted_out is not None:
            result = args.vote_result.format(out=names[voted_out])
        else:
            result = args.vote_all_skipped if not leaders else args.vote_tie
        table.say(day_opener, result)
        n_skips = sum(v["target"] is None for v in votes)
        print("tally: " + ", ".join(f"{names[c]}={tally[c]}" for c in sorted(tally, key=lambda c: -tally[c]))
              + (f", skipped={n_skips}" if n_skips else ""))
        say_line(day_opener, result)
        if voted_out is not None:
            alive.remove(voted_out)
            print(f"[god] {names[voted_out]} was a {roles[voted_out]}")
        phases.append({
            "phase": "day", "day": day, "opener": names[day_opener], "opening": opening,
            "discussion": [{"speaker": label(s), "text": t} for s, t in discussion],
            "discussion_columns": discussion_columns, "discussion_hit_cap": hit_cap,
            "seer": look,
            "vote_prompt": args.vote_prompt,
            "votes": [{"voter": names[v["voter"]], "voter_role": roles[v["voter"]],
                       "target": names[v["target"]] if v["target"] is not None else None,
                       "skipped": v["target"] is None, "p_skip": v["p_skip"],
                       "text": v["text"], "probs": v["probs"]} for v in votes],
            "tally": {names[c]: count for c, count in tally.items()}, "skipped": n_skips,
            "voted_out": names[voted_out] if voted_out is not None else None,
            "voted_out_role": roles[voted_out] if voted_out is not None else None,
            "result_text": result,
        })
        winner = check_winner(alive, roles)
        if winner or day == args.max_days:
            break

        # ---- Night: living Werewolves privately pick a Villager to kill ----
        night_wolves = [s for s in alive if roles[s] == "Werewolf"]
        targets = [s for s in alive if roles[s] != "Werewolf"]
        night_text = args.night_prompt.format(night=day)
        table.say(night_wolves[0], night_text, private=True)
        print(f"\n--- night {day} (private to Werewolves) ---")
        say_line(night_wolves[0], night_text)
        kill_votes = []
        for wolf in night_wolves:
            v = cast_ballot(table, wolf, args.kill_prefix, targets, names, tokenizer, speaker, args, private=True)
            kill_votes.append(v)
            say_line(wolf, v["text"])
            print(f"    P(kill): {v['probs']}")
        kill_tally, kill_leaders = plurality(v["target"] for v in kill_votes)
        killed = kill_leaders[0] if len(kill_leaders) == 1 else rng.choice(kill_leaders)
        alive.remove(killed)
        print(f"[god] the Werewolves killed {names[killed]}"
              f"{' (tie broken at random)' if len(kill_leaders) > 1 else ''}")
        phases.append({
            "phase": "night", "night": day, "prompt": night_text,
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
    print("\n--- result ---")
    print(f"days played: {day} | alive: {', '.join(label(s) for s in alive)}")
    print(f"WINNER: {winner.upper() + ' WIN' if winner else f'nobody (stopped at --max_days {args.max_days})'}")
    print(f"Villagers' mean P(vote for a living Werewolf) = {p_wolf:.3f} (chance = {chance:.3f}) "
          f"over {len(villager_vote_probs)} non-skipped ballots")
    print(f"Villager skip rate = {sum(skips)}/{len(skips)}")
    print(f"matrix length: {table.cache.seq_len} columns (training max_flat_len "
          f"{cfg.get('max_flat_len', '?')} flat tokens = ~{cfg.get('max_flat_len', 0) // max(cfg.get('num_agents', n), 1)} "
          f"columns at {cfg.get('num_agents', '?')} agents)")
    print("=" * 70)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "game_seed": args.game_seed, "checkpoint": args.checkpoint_dir,
                "args": vars(args), "dynamic_state": dynamic,
                "names": names, "roles": roles, "werewolves": [names[w] for w in wolves],
                "role_cards": cards, "phases": phases,
                "winner": winner, "days_played": day, "alive_at_end": [names[s] for s in alive],
                "villager_mean_p_vote_wolf": p_wolf, "villager_vote_chance": chance,
                "villager_skips": sum(skips), "villager_ballots": len(skips),
                "matrix_columns": table.cache.seq_len,
            }, f, indent=2)
        print(f"[info] wrote {args.output}")


if __name__ == "__main__":
    main()
