"""Play one Werewolf game between MatrixChat agents using only role tokens.

Every player gets the prelude the Werewolf training source uses
(``data/adapters/werewolf.py``): a PUBLIC ``Hi, I'm {name}.`` followed by a
PRIVATE role card that only that player can attend to. One random seat is the
Werewolf; with ``--seer`` another seat privately learns one player's role.

Day discussion
    ``turns`` (default): players speak in seat order for ``--discussion_rounds``
    rounds. Each turn starts with a short prefilled phrase (``--starters``) and
    the model finishes it, with a repetition penalty, until a sentence ends.
    ``free``: the model's own turn-taking via ``generate_matrix`` (the research
    setting; in practice one agent tends to hold the floor).

Vote
    Every player starts a turn with ``I vote for``. The model scores each other
    player's name as the continuation of that turn; the vote is sampled from
    that distribution (``--vote_temperature 0`` = most likely name). The
    player then continues ``because ...`` to explain it. Votes are public and
    sequential unless ``--blind_votes``. A unique plurality eliminates that
    player; the village wins if it is the Werewolf.

Everything happens in one shared matrix through the model's KV cache. There is
no game engine -- only prompt tokens, forced speaking turns, and per-turn
visibility. The checkpoint is rebuilt from EVERY saved matrix-config field
(unlike ``turn_taking_probe.load_model``), so GRU checkpoints load whole.

Usage:
    python scripts/eval/werewolf_game.py --checkpoint_dir checkpoints/research_run_1 \
        --game_seed 7 --seer --output logs/werewolf_games/game_7.json
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
from model.generation import generate_matrix
from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM
from scripts.eval.demo_freeform_conversation import render_transcript
from scripts.eval.evaluate_checkpoint_suite import _build_multiturn_inputs
from scripts.eval.turn_taking_probe import resolve_device
from training.checkpoint import apply_checkpoint_to_model, load_checkpoint_state

DEFAULT_STARTERS = [
    "I think", "Honestly, I", "I'm suspicious of", "I trust", "I don't think", "Something about",
]
GOALS = {
    "Werewolf": "Don't let anyone find out.",
    "Villager": "Work out who the Werewolf is.",
    "Seer": "Use what you saw to help find the Werewolf.",
}
VOTE_PREFIX = "I vote for"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", default="checkpoints/adaptcleanv3_cand_00003/best_probe",
                   help="Directory holding model_state.pt (e.g. a run's best_probe/, not the run root).")
    p.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--num_players", type=int, default=6)
    p.add_argument("--game_seed", type=int, default=0,
                   help="Decides names, roles, the opener, starters, and sampling.")
    p.add_argument("--seer", action="store_true",
                   help="One non-werewolf privately learns another player's role.")
    p.add_argument("--plain_cards", action="store_true",
                   help="Use the exact training card format, without the one-line goal.")
    p.add_argument("--opener", default="Okay, so who do we think the werewolf is?")
    # Discussion.
    p.add_argument("--discussion_mode", choices=["turns", "free"], default="turns")
    p.add_argument("--discussion_rounds", type=int, default=2, help="turns mode: times around the table.")
    p.add_argument("--turn_tokens", type=int, default=40, help="turns mode: max generated tokens per turn.")
    p.add_argument("--max_sentences", type=int, default=2, help="turns mode: end a turn after this many.")
    p.add_argument("--starters", nargs="+", default=DEFAULT_STARTERS,
                   help="turns mode: each turn opens with one of these, chosen at random.")
    p.add_argument("--discussion_tokens", type=int, default=100, help="free mode: columns of discussion.")
    p.add_argument("--activity_threshold", type=float, default=0.5, help="free mode.")
    p.add_argument("--rescue_after", type=int, default=5, help="free mode: silent columns before a forced speaker.")
    p.add_argument("--rescue_end_after", type=int, default=3, help="free mode: rescues before the discussion ends.")
    # Vote.
    p.add_argument("--vote_prompt", default="Alright, time to vote. Say who you're voting for and why.")
    p.add_argument("--vote_temperature", type=float, default=1.0,
                   help="Sample the vote from the name probabilities at this temperature (0 = argmax).")
    p.add_argument("--reason_tokens", type=int, default=40, help="Max generated tokens after 'because'.")
    p.add_argument("--blind_votes", action="store_true",
                   help="Each player votes from the discussion alone, without hearing earlier votes.")
    # Sampling for forced turns.
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.3)
    p.add_argument("--no_repeat_ngram", type=int, default=3)
    p.add_argument("--dynamic_state", choices=["on", "off"], default="on",
                   help="off drops the GRU, matching the repo's load_model behaviour.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    return p.parse_args(argv)


def deal_roles(num_players, game_seed, opener_text, seer=False, plain_cards=False):
    """Return ``(rng, names, roles, wolf, opener, cards, turns)`` for one game."""
    rng = random.Random(game_seed)
    names = rng.sample(RANDOM_NAME_POOL, num_players)
    wolf = rng.randrange(num_players)
    opener = rng.randrange(num_players)
    roles = ["Werewolf" if i == wolf else "Villager" for i in range(num_players)]
    seer_seat = seer_target = None
    if seer:
        seer_seat = rng.choice([i for i in range(num_players) if i != wolf])
        seer_target = rng.choice([i for i in range(num_players) if i != seer_seat])
        roles[seer_seat] = "Seer"
    cards = []
    for i in range(num_players):
        card = f"You are {names[i]}. Your role is {roles[i]}."
        if i == seer_seat:
            seen = "the Werewolf" if seer_target == wolf else "a Villager"
            card += f" Last night you looked at {names[seer_target]}'s card: {names[seer_target]} is {seen}."
        if not plain_cards:
            card += " " + GOALS[roles[i]]
        cards.append(card)
    turns = []
    for i in range(num_players):
        turns.append({"speaker": i, "text": f"Hi, I'm {names[i]}.", "visible_to": None})
        turns.append({"speaker": i, "text": cards[i], "visible_to": []})  # private: owner only
    turns.append({"speaker": opener, "text": opener_text, "visible_to": None})
    return rng, names, roles, wolf, opener, cards, turns


def load_game_model(model_path, checkpoint_dir, num_players, dynamic_state, device, dtype):
    """Load base Qwen + checkpoint, rebuilding every saved matrix-config field."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    saved = load_checkpoint_state(checkpoint_dir)  # fail before loading the 8 GB base model
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
    return model.to(device).eval(), tokenizer, cfg


def trim_trailing_silence(tokens, speak):
    """Drop all-silent columns after the last spoken one."""
    spoken = speak[0].any(dim=0).nonzero()
    keep = int(spoken.max()) + 1 if len(spoken) else 0
    return tokens[..., :keep], speak[..., :keep]


class Table:
    """The shared game matrix, extended one column at a time through the KV cache.

    ``logits[0, a]`` always holds agent ``a``'s next-content logits given
    everything so far, exactly as ``generate_matrix`` reads them.
    """

    def __init__(self, model, tokenizer, input_ids, activity, private, visibility):
        self.model, self.tokenizer = model, tokenizer
        self.num_agents = input_ids.shape[1]
        self.device = input_ids.device
        self.cache, self.logits, _ = model.encode_prefix_for_generation(
            input_ids=input_ids, input_activity_mask=activity,
            input_private_mask=private, agent_visibility=visibility,
        )

    def push(self, speaker, token_id):
        ids = torch.zeros((1, self.num_agents, 1), dtype=torch.long, device=self.device)
        active = torch.zeros((1, self.num_agents, 1), dtype=torch.bool, device=self.device)
        ids[0, speaker, 0] = int(token_id)
        active[0, speaker, 0] = True
        self.cache, self.logits, _ = self.model.forward_generation_column(self.cache, ids, active)

    def say(self, speaker, text):
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        for token_id in ids:
            self.push(speaker, token_id)
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

    def sample(self, logits, history):
        args = self.args
        logits = logits.float().clone()
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

    def continue_turn(self, table, speaker, prefix_ids, max_tokens, max_sentences):
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
            table.push(speaker, token_id)
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
    scores = []
    for ids in candidate_ids:
        branch = table.fork()
        total = 0.0
        for k, token_id in enumerate(ids):
            total += float(torch.log_softmax(branch.logits[0, voter].float(), dim=-1)[token_id])
            if k + 1 < len(ids):
                branch.push(voter, token_id)
        scores.append(total)
    return scores


def main(argv=None):
    args = parse_args(argv)
    n = args.num_players
    rng, names, roles, wolf, opener, cards, turns = deal_roles(
        n, args.game_seed, args.opener, seer=args.seer, plain_cards=args.plain_cards,
    )

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
    )
    torch.manual_seed(args.game_seed)
    speaker = Speaker(tokenizer, args)

    # Prelude + opener, then the day discussion.
    inp, mask, private_mask, visibility = _build_multiturn_inputs(tokenizer, turns, n, device)
    discussion = []
    if args.discussion_mode == "free":
        tokens, speak, _ = generate_matrix(
            model, inp, args.discussion_tokens, input_activity_mask=mask,
            input_private_mask=private_mask, agent_visibility=visibility,
            temperature=args.temperature, activity_threshold=args.activity_threshold,
            placeholder_token_id=0, silence_rescue_after_columns=args.rescue_after,
            silence_rescue_end_after_triggers=args.rescue_end_after, return_probs=True,
        )
        tokens, speak = trim_trailing_silence(tokens, speak)
        discussion = render_transcript(tokenizer, tokens, speak, n)
        inp = torch.cat([inp, tokens], dim=2)
        mask = torch.cat([mask, speak], dim=2)
        private_mask = torch.cat([private_mask, torch.zeros_like(speak)], dim=2)
    table = Table(model, tokenizer, inp, mask, private_mask, visibility)

    seat_order = [(opener + 1 + k) % n for k in range(n)]  # opener speaks last each round
    if args.discussion_mode == "turns":
        for _ in range(args.discussion_rounds):
            for seat in seat_order:
                starter = rng.choice(args.starters)
                prefix = table.say(seat, starter)
                out = speaker.continue_turn(table, seat, prefix, args.turn_tokens, args.max_sentences)
                discussion.append((seat, tokenizer.decode(prefix + out, skip_special_tokens=True)))

    # Vote.
    table.say(opener, args.vote_prompt)
    ballot_box = table.fork() if args.blind_votes else table
    votes = []
    for voter in seat_order:
        booth = ballot_box.fork() if args.blind_votes else table
        candidates = [c for c in range(n) if c != voter]
        prefix = booth.say(voter, VOTE_PREFIX)
        candidate_ids = [name_token_ids(tokenizer, VOTE_PREFIX, names[c]) for c in candidates]
        scores = torch.tensor(score_names(booth, voter, candidate_ids))
        probs = torch.softmax(scores, dim=0)
        if args.vote_temperature > 0:
            pick = int(torch.multinomial(torch.softmax(scores / args.vote_temperature, dim=0), 1))
        else:
            pick = int(scores.argmax())
        target = candidates[pick]
        for token_id in candidate_ids[pick]:
            booth.push(voter, token_id)
        because = booth.say(voter, " because")
        reason = speaker.continue_turn(
            booth, voter, prefix + candidate_ids[pick] + because, args.reason_tokens, 1,
        )
        votes.append({
            "voter": voter, "target": target,
            "text": tokenizer.decode(prefix + candidate_ids[pick] + because + reason, skip_special_tokens=True),
            "probs": {names[c]: round(float(p), 3) for c, p in zip(candidates, probs)},
        })

    tally = Counter(v["target"] for v in votes)
    top = max(tally.values())
    leaders = [c for c, count in tally.items() if count == top]
    eliminated = leaders[0] if len(leaders) == 1 else None
    village_wins = eliminated == wolf
    non_wolves = [v for v in votes if v["voter"] != wolf]
    p_wolf = sum(v["probs"][names[wolf]] for v in non_wolves) / max(len(non_wolves), 1)
    dynamic = cfg.get("agent_dynamic_state_mode", "none")

    def label(seat):
        if isinstance(seat, int):
            return f"{names[seat]}{' [WOLF]' if seat == wolf else ' [SEER]' if roles[seat] == 'Seer' else ''}"
        return "OVERLAP(" + ", ".join(names[int(a)] for a in seat.split(":")[1].split(",")) + ")"

    # God-view report.
    print("=" * 70)
    print(f"game_seed={args.game_seed} checkpoint={args.checkpoint_dir} players={n} "
          f"discussion={args.discussion_mode} dynamic_state={dynamic} blind_votes={args.blind_votes}")
    for i in range(n):
        print(f"  agent {i}: {names[i]:<10} role card (private): {cards[i]!r}")
    print(f"{label(opener)}: {args.opener}")
    print("--- discussion ---")
    for seat, text in discussion:
        print(f"{label(seat)}: {text}")
    print(f"{label(opener)}: {args.vote_prompt}")
    print("--- vote ---")
    for v in votes:
        print(f"{label(v['voter'])}: {v['text']}")
        print(f"    P(vote): {v['probs']}")
    print("--- result ---")
    print("tally: " + ", ".join(f"{names[c]}={tally[c]}" for c in sorted(tally, key=lambda c: -tally[c])))
    if eliminated is None:
        print(f"tie between {[names[c] for c in leaders]} -> nobody eliminated -> WEREWOLF WINS")
    else:
        print(f"eliminated: {names[eliminated]} ({roles[eliminated]}) -> "
              f"{'VILLAGE WINS' if village_wins else 'WEREWOLF WINS'}")
    print(f"non-werewolves' mean P(vote for {names[wolf]}) = {p_wolf:.3f} (chance = {1 / (n - 1):.3f})")
    print("=" * 70)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "game_seed": args.game_seed, "checkpoint": args.checkpoint_dir,
                "args": vars(args), "dynamic_state": dynamic,
                "names": names, "roles": roles, "werewolf": names[wolf],
                "role_cards": cards, "opener": names[opener], "opener_text": args.opener,
                "discussion": [{"speaker": label(s), "text": t} for s, t in discussion],
                "vote_prompt": args.vote_prompt,
                "votes": [{"voter": names[v["voter"]], "voter_role": roles[v["voter"]],
                           "target": names[v["target"]], "text": v["text"], "probs": v["probs"]}
                          for v in votes],
                "tally": {names[c]: count for c, count in tally.items()},
                "eliminated": names[eliminated] if eliminated is not None else None,
                "village_wins": village_wins,
                "non_wolf_mean_p_vote_wolf": p_wolf,
            }, f, indent=2)
        print(f"[info] wrote {args.output}")


if __name__ == "__main__":
    main()
