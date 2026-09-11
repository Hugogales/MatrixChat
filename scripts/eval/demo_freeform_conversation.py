"""Ad hoc demo: seed a topic and let the multi-agent model free-run for a
long window, then render a readable chronological transcript (not just
per-agent concatenated text) to see what the agents actually talk about.
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

from model.generation import generate_matrix
from scripts.eval.evaluate_checkpoint_suite import _build_multiturn_inputs
from scripts.eval.turn_taking_probe import load_model, resolve_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--seed_text", required=True)
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--activity_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    return p.parse_args()


def render_transcript(tokenizer, tokens, speak, num_agents):
    """Merge consecutive same-speaker active columns into readable lines,
    in chronological (column) order -- a real dialogue transcript, not
    just each agent's text concatenated separately."""
    steps = tokens.shape[-1]
    lines = []
    cur_speaker, cur_ids = None, []
    for step in range(steps):
        active = [a for a in range(num_agents) if bool(speak[0, a, step])]
        speaker = active[0] if len(active) == 1 else ("OVERLAP:" + ",".join(map(str, active))) if active else None
        if speaker != cur_speaker:
            if cur_speaker is not None and cur_ids:
                lines.append((cur_speaker, tokenizer.decode(cur_ids, skip_special_tokens=True)))
            cur_speaker, cur_ids = speaker, []
        if active:
            for a in active:
                cur_ids.append(int(tokens[0, a, step]))
    if cur_speaker is not None and cur_ids:
        lines.append((cur_speaker, tokenizer.decode(cur_ids, skip_special_tokens=True)))
    return lines


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_model(args, device, dtype)

    turns = [{"speaker": 0, "text": args.seed_text, "visible_to": None}]
    inp, mask, private_mask, visibility = _build_multiturn_inputs(
        tokenizer, turns, args.num_agents, device,
    )
    tokens, speak, probs = generate_matrix(
        model, inp, args.max_new_tokens,
        input_activity_mask=mask, temperature=args.temperature,
        activity_threshold=args.activity_threshold, placeholder_token_id=0,
        input_private_mask=private_mask, agent_visibility=visibility,
        return_probs=True,
    )
    lines = render_transcript(tokenizer, tokens, speak, args.num_agents)

    print("=" * 70)
    print(f"seed (Agent 0): {args.seed_text!r}")
    print(f"agents: {args.num_agents}   max_new_tokens: {args.max_new_tokens}   "
          f"temperature: {args.temperature}   seed: {args.seed}")
    print("-" * 70)
    print(f"Agent 0: {args.seed_text}")
    for speaker, text in lines:
        label = f"Agent {speaker}" if isinstance(speaker, int) else speaker
        print(f"{label}: {text}")
    print("=" * 70)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "seed_text": args.seed_text, "num_agents": args.num_agents,
                "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                "seed": args.seed,
                "transcript": [{"speaker": s, "text": t} for s, t in lines],
            }, f, indent=2)


if __name__ == "__main__":
    main()
