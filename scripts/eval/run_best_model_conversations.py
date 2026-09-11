"""Driver: run demo_freeform_conversation's generation for a fixed set of
conversation starters against a fixed set of "best model" checkpoints, and
dump every transcript (plus a couple of extra seeds/agent-counts per prompt
for robustness) to JSON for later write-up into a markdown report.
"""

from __future__ import annotations

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
from scripts.eval.demo_freeform_conversation import render_transcript

MODELS = [
    {
        "label": "h100mm_cand_00012 (overall leader, low-overlap)",
        "candidate_id": "h100mm_cand_00012",
        "checkpoint_dir": "checkpoints/h100mm_cand_00012/rung1_step1500",
    },
    {
        "label": "h100mm_cand_00050 (chain leader)",
        "candidate_id": "h100mm_cand_00050",
        "checkpoint_dir": "checkpoints/h100mm_cand_00050/last",
    },
]

PROMPTS = [
    "what do you think of spain winning the world cup",
    "I think jeremy is the werewolf, he is acting suspicios, right",
    "We were on a break",
    "let talk about the political and economical state of the world right now. what do you think",
]

NUM_AGENTS = 3
MAX_NEW_TOKENS = 160
SEEDS = [0, 1]


class Args:
    model_path = "models/Qwen3-4B-Instruct-2507"
    device = "cuda"
    num_agents = NUM_AGENTS


def main():
    device = resolve_device(Args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out_dir = os.path.join(_ROOT, "logs", "best_model_conversations")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for model_info in MODELS:
        args = Args()
        args.checkpoint_dir = model_info["checkpoint_dir"]
        print(f"\n{'#'*70}\nLoading {model_info['label']} from {args.checkpoint_dir}\n{'#'*70}")
        model, tokenizer = load_model(args, device, dtype)

        for prompt in PROMPTS:
            for seed in SEEDS:
                torch.manual_seed(seed)
                turns = [{"speaker": 0, "text": prompt, "visible_to": None}]
                inp, mask, private_mask, visibility = _build_multiturn_inputs(
                    tokenizer, turns, NUM_AGENTS, device,
                )
                tokens, speak, probs = generate_matrix(
                    model, inp, MAX_NEW_TOKENS,
                    input_activity_mask=mask, temperature=0.8,
                    activity_threshold=0.5, placeholder_token_id=0,
                    input_private_mask=private_mask, agent_visibility=visibility,
                    return_probs=True,
                )
                lines = render_transcript(tokenizer, tokens, speak, NUM_AGENTS)
                row = {
                    "candidate_id": model_info["candidate_id"],
                    "label": model_info["label"],
                    "checkpoint_dir": model_info["checkpoint_dir"],
                    "prompt": prompt,
                    "num_agents": NUM_AGENTS,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "seed": seed,
                    "transcript": [{"speaker": s, "text": t} for s, t in lines],
                }
                results.append(row)
                print(f"\n--- {model_info['candidate_id']} | seed={seed} | {prompt!r} ---")
                print(f"Agent 0: {prompt}")
                for s, t in lines:
                    label = f"Agent {s}" if isinstance(s, int) else s
                    print(f"{label}: {t}")

        del model
        torch.cuda.empty_cache()

    out_path = os.path.join(out_dir, "conversations.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[info] wrote {len(results)} transcripts to {out_path}")


if __name__ == "__main__":
    main()
