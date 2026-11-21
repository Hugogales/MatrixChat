"""Play one Werewolf game between MatrixChat agents using only role tokens.

Every player gets the prelude the Werewolf training source uses
(``data/adapters/werewolf.py``): a PUBLIC ``Hi, I'm {name}.`` followed by a
PRIVATE role card that only that player can attend to. ``--num_werewolves``
random seats are Werewolves (they can see each other's cards) and everyone
else is a Villager (one of them the Seer with ``--seer``).

The game itself lives in ``training/werewolf_rl/game.py`` and is shared with
the RL trainer (``scripts/training/train_werewolf_rl.py``), so what this script
plays is exactly what training plays.

Game loop: Day 1 -> Night 1 -> Day 2 -> ... until a side wins (or ``--max_days``
if set). Village wins when every Werewolf is dead; Werewolves win when every
other player is dead (``--werewolves_win parity``: as soon as there are as many
living Werewolves as living non-Werewolves). Role cards are first
person ("I am X. I am a Villager.") unless ``--plain_cards``. Dead players
never speak, vote, or appear as candidates again. Each day is opened by a
different living player; from Day 2 on the opener first names who was killed.

Day discussion
    ``free`` (default): the model runs the conversation itself. Every column,
    each living player samples whether to speak from its activity head,
    ``Bernoulli(sigmoid(logit / --activity_temperature))`` (``--speak_sampling
    threshold`` restores the old ``P(speak) > --activity_threshold`` rule), and
    a speaker emits one token (overlap and interruptions allowed). During a
    table-wide silence every living player's speak logit rises by
    ``--silence_ramp`` (0.75) per silent column past ``--silence_ramp_start``
    (2); after ``--rescue_after`` (5) silent columns the player with the
    highest P(speak) is made to speak, and the ``--rescue_end_after``-th (3rd)
    such silence ends the discussion (``--rescue_after 0``: end after
    ``--quiet_seconds`` of quiet instead). ``--max_discussion_columns`` caps it.
    ``turns``: players speak in seat order for ``--discussion_rounds`` rounds.

Vote
    Every living player must vote (no skipping). Each ballot
    (``--ballot_order reason_first``, default) is a forced justification --
    the player speaks freely (up to ``--reason_sentences``) -- then ``I think
    the Werewolf is`` (``--vote_prefix``) and a name sampled from the model's
    softmax over the other living players' names (``--vote_temperature``).
    Votes are public and sequential unless ``--blind_votes``. A unique
    plurality eliminates that player; a tie eliminates one tied player at
    random. Roles are not revealed.

Night
    With ``--seer``, the Seer first privately picks a living player it hasn't
    looked at (``I want to look at <name> because ...``) and privately learns
    ``I looked at <name>'s card: <name> is a Werewolf / a Villager.`` Then the
    living Werewolves privately hear ``--night_prompt`` and each casts a kill
    ballot over the living non-Werewolves. Plurality wins; ties are random.

Rewards (WEREWOLF_RL_PLAN.md)
    After the game the script prints and saves each player's Pillar-2 round
    rewards (S_suspicion: votes against them; S_vote: accuracy of their
    ballot / Seer look) and Pillar-3 S_win. ``--rollout_output x.pt`` also
    saves the full matrix and every sampled action with its log-probability
    (the trainer's rollout format).

Spoken tokens are sampled with ``--temperature``, ``--repetition_penalty``,
``--no_repeat_ngram`` and ``--no_copy_ngram`` (no player may repeat a span
from an earlier turn by anyone); top-p is off by default so the policy is
exactly what RL trains. Names come from the single-token part of the name
pool (``--name_pool all`` for the full pool).

Usage:
    python scripts/eval/werewolf_game.py --checkpoint_dir checkpoints/checkpoint \
        --hyperparameters hyperparameters/adaptcleanv3_cand_00003.json \
        --game_seed 7 --output logs/werewolf_games/game_7.json

    # Slurm run example
    sbatch --partition=teaching --gpus=1 --cpus-per-task=4 --time=02:00:00 \
        --job-name=werewolf_game \
        --output=logs/slurm/%x_%j.out --error=logs/slurm/%x_%j.err \
        --wrap="python scripts/eval/werewolf_game.py \
            --checkpoint_dir checkpoints/checkpoint \
            --hyperparameters hyperparameters/adaptcleanv3_cand_00003.json \
            --game_seed 1105 \
            --output logs/werewolf_games/checkpoint_1105_free.json"

"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.eval.turn_taking_probe import resolve_device  # noqa: E402
from training.werewolf_rl.game import add_game_args, load_game_model, play_game  # noqa: E402
from training.werewolf_rl.rewards import RewardConfig, game_rewards, round_rewards  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", default="checkpoints/checkpoint",
                   help="Directory holding model_state.pt, lora_adapters/ and (optionally) best_probe.json.")
    p.add_argument("--hyperparameters", default="hyperparameters/adaptcleanv3_cand_00003.json",
                   help="Training hyperparameter JSON; its matrix-config fields override the ones saved "
                        "in model_state.pt ('' = use only the saved config).")
    p.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--game_seed", type=int, default=0,
                   help="Decides names, roles, the opener, and sampling.")
    add_game_args(p)
    p.add_argument("--dynamic_state", choices=["on", "off"], default="on",
                   help="off drops the GRU, matching the repo's load_model behaviour.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None, help="Write the game (transcript, events, rewards) as JSON.")
    p.add_argument("--rollout_output", default=None,
                   help="Also record every sampled action (RL rollout format) and torch.save it here.")
    return p.parse_args(argv)


def reward_report(result, cfg=None):
    """Per-player Pillar 2 / Pillar 3 rewards of a finished game."""
    cfg = cfg or RewardConfig()
    events = result["events"]
    rounds = round_rewards(events, cfg)
    wins = game_rewards(events)
    players = []
    for seat, name in enumerate(result["names"]):
        days = sorted(d for s, d in rounds if s == seat)
        players.append({
            "name": name, "role": result["roles"][seat], "s_win": wins[seat],
            "rounds": [{"day": d, **{k: round(v, 3) for k, v in rounds[(seat, d)].items()}} for d in days],
        })
    return players


def main(argv=None):
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"--device {args.device} requested but CUDA is unavailable (torch {torch.__version__} "
            f"from {torch.__file__}, built for CUDA {torch.version.cuda}). A 4B game on CPU will "
            "not finish; pass --device cpu to force it."
        )
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tokenizer, cfg, _ = load_game_model(
        args.model_path, args.checkpoint_dir, args.num_players, args.dynamic_state, device, dtype,
        hyperparameters=args.hyperparameters,
    )
    print(f"[info] checkpoint={args.checkpoint_dir} dynamic_state={cfg.get('agent_dynamic_state_mode', 'none')}")
    with torch.no_grad():
        result = play_game(model, tokenizer, args, args.game_seed, record=bool(args.rollout_output),
                           log=lambda *a: print(*a, flush=True))
    print(f"(training max_flat_len {cfg.get('max_flat_len', '?')} flat tokens = "
          f"~{cfg.get('max_flat_len', 0) // max(cfg.get('num_agents', args.num_players), 1)} columns at "
          f"{cfg.get('num_agents', '?')} agents)")

    players = reward_report(result)
    print("\n--- rewards (Pillar 2 per round: S_suspicion, S_vote -> r_round; Pillar 3: S_win) ---")
    for pl in players:
        rounds = "  ".join(f"d{r['day']}: {r['s_suspicion']:+.2f},{r['s_vote']:+.2f}->{r['r_round']:+.2f}"
                           for r in pl["rounds"])
        print(f"  {pl['name']:<10} {pl['role']:<9} S_win={pl['s_win']:+.0f}  {rounds}")
    print("=" * 70)

    if args.rollout_output:
        os.makedirs(os.path.dirname(args.rollout_output) or ".", exist_ok=True)
        torch.save(result["rollout"], args.rollout_output)
        print(f"[info] wrote {args.rollout_output}")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        payload = {k: v for k, v in result.items() if k != "rollout"}
        payload.update({"checkpoint": args.checkpoint_dir, "args": vars(args),
                        "dynamic_state": cfg.get("agent_dynamic_state_mode", "none"), "rewards": players})
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[info] wrote {args.output}")


if __name__ == "__main__":
    main()
