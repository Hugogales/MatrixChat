"""Play one Werewolf game between MatrixChat agents using only role tokens.

Every player gets the same prelude the Werewolf training source uses
(``data/adapters/werewolf.py``): a PUBLIC ``Hi, I'm {name}.`` followed by a
PRIVATE ``You are {name}. Your role is {role}.`` that only that player can
attend to. One randomly chosen seat is the Werewolf; the rest are Villagers.
A random player opens the day discussion, the model generates it, then a
public vote prompt is appended and the model generates the vote. There is no
game engine -- only prompt tokens and per-turn visibility.

Unlike ``turn_taking_probe.load_model``, the checkpoint is rebuilt from EVERY
saved matrix-config field, so GRU dynamic-state checkpoints (c3) load whole.

Usage:
    python scripts/eval/werewolf_game.py \
        --checkpoint_dir checkpoints/adaptcleanv3_cand_00003/best_probe \
        --game_seed 7 --output logs/werewolf_games/game_7.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys

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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", default="checkpoints/adaptcleanv3_cand_00003/best_probe",
                   help="Directory holding model_state.pt (e.g. a run's best_probe/, not the run root).")
    p.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--num_players", type=int, default=6)
    p.add_argument("--game_seed", type=int, default=0,
                   help="Decides names, the Werewolf seat, the opener, and sampling.")
    p.add_argument("--opener", default="okay so who do we think the werewolf is")
    p.add_argument("--vote_prompt", default="alright time to vote, everyone say who you are voting for")
    # c3 trained at <=1536 flat tokens = ~256 columns at 6 agents; the prelude is ~105.
    p.add_argument("--discussion_tokens", type=int, default=100, help="Columns of day discussion.")
    p.add_argument("--vote_tokens", type=int, default=40, help="Columns of voting (0 = skip the vote).")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--activity_threshold", type=float, default=0.5)
    p.add_argument("--rescue_after", type=int, default=5,
                   help="All-silent columns before the likeliest speaker is forced to talk "
                        "(generate_matrix default 5; lower = less silence).")
    p.add_argument("--rescue_end_after", type=int, default=3,
                   help="End generation on this many silence rescues (generate_matrix default 3).")
    p.add_argument("--dynamic_state", choices=["on", "off"], default="on",
                   help="off drops the GRU, matching the repo's load_model behaviour.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    return p.parse_args(argv)


def deal_roles(num_players, game_seed, opener_text):
    """Return ``(names, roles, wolf, opener, turns)`` for one game."""
    rng = random.Random(game_seed)
    names = rng.sample(RANDOM_NAME_POOL, num_players)
    wolf = rng.randrange(num_players)
    roles = ["Werewolf" if i == wolf else "Villager" for i in range(num_players)]
    opener = rng.randrange(num_players)
    turns = []
    for i in range(num_players):
        turns.append({"speaker": i, "text": f"Hi, I'm {names[i]}.", "visible_to": None})
        turns.append({"speaker": i, "text": f"You are {names[i]}. Your role is {roles[i]}.",
                      "visible_to": []})  # private: owner only
    turns.append({"speaker": opener, "text": opener_text, "visible_to": None})
    return names, roles, wolf, opener, turns


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


def main(argv=None):
    args = parse_args(argv)
    n = args.num_players
    names, roles, wolf, opener, turns = deal_roles(n, args.game_seed, args.opener)

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
    gen = dict(
        temperature=args.temperature, activity_threshold=args.activity_threshold,
        placeholder_token_id=0, silence_rescue_after_columns=args.rescue_after,
        silence_rescue_end_after_triggers=args.rescue_end_after,
        return_probs=True,
    )

    # Day discussion.
    inp, mask, private_mask, visibility = _build_multiturn_inputs(tokenizer, turns, n, device)
    torch.manual_seed(args.game_seed)
    tokens, speak, _ = generate_matrix(
        model, inp, args.discussion_tokens, input_activity_mask=mask,
        input_private_mask=private_mask, agent_visibility=visibility, **gen,
    )
    tokens, speak = trim_trailing_silence(tokens, speak)
    discussion = render_transcript(tokenizer, tokens, speak, n)

    # Vote: discussion + one more public prompt, same private role cards.
    vote = []
    if args.vote_tokens > 0:
        vote_inp, vote_mask, _, _ = _build_multiturn_inputs(
            tokenizer, [{"speaker": opener, "text": args.vote_prompt, "visible_to": None}], n, device,
        )
        inp2 = torch.cat([inp, tokens, vote_inp], dim=2)
        mask2 = torch.cat([mask, speak, vote_mask], dim=2)
        private2 = torch.cat([private_mask, torch.zeros_like(speak), torch.zeros_like(vote_mask)], dim=2)
        vote_tokens, vote_speak, _ = generate_matrix(
            model, inp2, args.vote_tokens, input_activity_mask=mask2,
            input_private_mask=private2, agent_visibility=visibility, **gen,
        )
        vote = render_transcript(tokenizer, vote_tokens, vote_speak, n)

    def label(speaker):
        if isinstance(speaker, int):
            return f"{names[speaker]}{' [WOLF]' if speaker == wolf else ''}"
        return "OVERLAP(" + ", ".join(names[int(a)] for a in speaker.split(":")[1].split(",")) + ")"

    mentions = {name: 0 for name in names}
    for speaker, text in vote:
        if isinstance(speaker, int):
            for i, name in enumerate(names):
                if i != speaker and name.lower() in text.lower():
                    mentions[name] += 1
    top = max(mentions.values())
    accused = [name for name, count in mentions.items() if count == top and count > 0]
    dynamic = cfg.get("agent_dynamic_state_mode", "none")

    # God-view report.
    print("=" * 70)
    print(f"game_seed={args.game_seed} checkpoint={args.checkpoint_dir} players={n} dynamic_state={dynamic}")
    for i in range(n):
        print(f"  agent {i}: {names[i]:<10} role card (private): {turns[2 * i + 1]['text']!r}")
    print(f"opener: {names[opener]}: {args.opener}")
    print("--- discussion ---")
    for speaker, text in discussion:
        print(f"{label(speaker)}: {text}")
    if args.vote_tokens > 0:
        print(f"{names[opener]}: {args.vote_prompt}")
        print("--- vote ---")
        for speaker, text in vote:
            print(f"{label(speaker)}: {text}")
        print(f"vote-phase name mentions: {mentions}")
        print(f"most named: {accused or 'nobody'} | werewolf was {names[wolf]} -> "
              f"{'VILLAGE WINS' if accused == [names[wolf]] else 'WEREWOLF WINS / no consensus'}")
    print("=" * 70)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "game_seed": args.game_seed, "checkpoint": args.checkpoint_dir,
                "names": names, "roles": roles, "werewolf": names[wolf],
                "opener": names[opener], "opener_text": args.opener,
                "role_cards": [turns[2 * i + 1]["text"] for i in range(n)],
                "generation": gen, "dynamic_state": dynamic,
                "discussion": [{"speaker": label(s), "text": t} for s, t in discussion],
                "vote_prompt": args.vote_prompt if args.vote_tokens > 0 else None,
                "vote": [{"speaker": label(s), "text": t} for s, t in vote],
                "vote_name_mentions": mentions,
            }, f, indent=2)
        print(f"[info] wrote {args.output}")


if __name__ == "__main__":
    main()
