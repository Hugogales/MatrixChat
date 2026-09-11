"""Render the exact [Agent x Time] activity/content grid for specific
(prompt, num_agents, seed) trials -- used to visualize handoff/overlap
examples found by demo_handoff_variation.py."""

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
from scripts.eval.evaluate_checkpoint_suite import _build_multiturn_inputs, _score_handoff
from scripts.eval.turn_taking_probe import load_model, resolve_device

TRIALS = [
    ("i will stop there and let someone else take it from here", 2, 0),
    ("well that is my two cents on the matter what do you all think", 5, 2),
    ("so that is basically the whole story does that make sense to everyone", 5, 2),
    ("alright i am done talking now who else has something to add", 5, 2),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--activity_threshold", type=float, default=0.5)
    p.add_argument("--num_agents", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_model(args, device, dtype)

    all_grids = []
    for prompt, num_agents, seed in TRIALS:
        torch.manual_seed(seed)
        turns = [{"speaker": 0, "text": prompt, "visible_to": None}]
        inp, mask, private_mask, visibility = _build_multiturn_inputs(tokenizer, turns, num_agents, device)
        tokens, speak, probs = generate_matrix(
            model, inp, args.max_new_tokens,
            input_activity_mask=mask, temperature=args.temperature,
            activity_threshold=args.activity_threshold, placeholder_token_id=0,
            input_private_mask=private_mask, agent_visibility=visibility,
            return_probs=True,
        )
        other_agents = list(range(1, num_agents))
        overlap, overlap_tokens, listener_any, first_listener, first_ref, last_ref, clean_handoff, interruption_handoff, dirty_handoff = _score_handoff(
            speak[0], "handoff", 0, other_agents,
        )
        grid = []
        for a in range(num_agents):
            row = []
            for s in range(tokens.shape[-1]):
                if bool(speak[0, a, s]):
                    row.append(tokenizer.decode([int(tokens[0, a, s])]).replace("\n", "\\n").strip() or "_")
                else:
                    row.append(".")
            grid.append(row)
        tag = (
            "CLEAN" if clean_handoff else
            ("INTERRUPTION" if interruption_handoff else
             ("DIRTY" if dirty_handoff else
              ("OVERLAP" if overlap else
               ("SILENT" if not listener_any else "MIXED"))))
        )
        result = {
            "prompt": prompt, "num_agents": num_agents, "seed": seed, "tag": tag,
            "clean_handoff": clean_handoff,
            "interruption_handoff": interruption_handoff,
            "dirty_handoff": dirty_handoff,
            "overlap": overlap,
            "overlap_tokens": overlap_tokens,
            "first_listener_step": first_listener, "last_reference_step": last_ref,
            "grid": grid,
        }
        all_grids.append(result)

        print("=" * 70)
        print(f"[{tag}] prompt={prompt!r} n={num_agents} seed={seed}")
        header = "step | " + " | ".join(f"agent {a}".ljust(10) for a in range(num_agents))
        print(header)
        print("-" * len(header))
        for s in range(len(grid[0])):
            row_str = f"{s:4d} | " + " | ".join(grid[a][s][:9].ljust(10) for a in range(num_agents))
            print(row_str)
        print()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_grids, f, indent=2)


if __name__ == "__main__":
    main()
