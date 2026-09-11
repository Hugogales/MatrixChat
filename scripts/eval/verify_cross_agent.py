"""Prove (or disprove) that MatrixChat agents truly share context.

Generating tokens is not proof that agent A conditions on agent B. This probe
edits ONE agent's tokens and measures, at the logit level, whether the OTHER
agent's prediction changes. It runs three controlled checks that together pin
down the exact attention semantics:

  1. PAST influence:        change agent 1's *earlier* columns
                            -> agent 0's last-column logits SHOULD change
                            (agents read each other's history).
  2. No future leakage:     change agent 1's *last* column
                            -> agent 0's *earlier*-column logits SHOULD NOT change
                            (strict causality).
  3. No same-column leak:    change agent 1's *last* column
                            -> agent 0's *last*-column logits SHOULD NOT change
                            (allow_same_column=False blocks same-step cross-agent).

A correct wrapper yields: check 1 = CHANGED, checks 2 and 3 = IDENTICAL (0.0).

Run:
    python scripts/eval/verify_cross_agent.py --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", type=str, default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    p.add_argument("--position_mode", type=str, default="flat", choices=["flat", "column"])
    p.add_argument("--allow_same_column", action="store_true",
                   help="If set, check 3 is expected to CHANGE (same column visible).")
    return p.parse_args(argv)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def agent0_logits(model, input_ids, column):
    """Return agent 0's logits at a given time column: [V]."""
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits[0, 0, column, :].float()


def report(name, base, other, expect_change):
    max_abs = (base - other).abs().max().item()
    changed = max_abs > 1e-6
    verdict = "CHANGED " if changed else "IDENTICAL"
    ok = "PASS" if changed == expect_change else "FAIL"
    want = "should CHANGE" if expect_change else "should be IDENTICAL"
    print(f"  [{ok}] {name}: max|Δlogit| = {max_abs:.6f}  -> {verdict}  ({want})")
    return changed == expect_change


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch_dtype, attn_implementation="eager")
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch_dtype, attn_implementation="eager")

    cfg = MatrixQwenConfig(
        base_model_name=args.model_path,
        max_agents=4,
        use_agent_embeddings=False,   # isolate the ATTENTION pathway, not embeddings
        position_mode=args.position_mode,
        attn_mask_mode="matrix_causal",
        allow_same_column=args.allow_same_column,
    )
    model = MatrixQwenForCausalLM(base_model, cfg).to(device).eval()

    # Two agents, equal-length rows (T columns). Use distinct, deterministic tokens.
    agent0 = tok("The weather in the city today is really")["input_ids"]
    agent1 = tok("My favorite hobby is playing chess on")["input_ids"]
    T = min(len(agent0), len(agent1))
    agent0, agent1 = agent0[:T], agent1[:T]
    base = torch.tensor([[agent0, agent1]], dtype=torch.long, device=device)  # [1, 2, T]
    last, earlier = T - 1, T - 2
    alt_token = int(tok("banana")["input_ids"][0])

    print("=" * 70)
    print(f"model: {args.model_path}  device: {device}  T={T} columns/agent")
    print(f"position_mode={args.position_mode}  allow_same_column={args.allow_same_column}")
    print("=" * 70)

    base_a0_last = agent0_logits(model, base, last)
    base_a0_earlier = agent0_logits(model, base, earlier)

    # Check 1: change agent 1's EARLIER columns (its history) -> a0 last-col logits.
    mod_past = base.clone()
    mod_past[0, 1, :earlier] = alt_token
    a0_last_modpast = agent0_logits(model, mod_past, last)

    # Check 2 & 3: change agent 1's LAST column only.
    mod_last = base.clone()
    mod_last[0, 1, last] = alt_token
    a0_earlier_modlast = agent0_logits(model, mod_last, earlier)
    a0_last_modlast = agent0_logits(model, mod_last, last)

    print("Editing AGENT 1, measuring AGENT 0's logits:\n")
    ok = []
    ok.append(report("1. change agent1 PAST   -> agent0 last-col   ",
                     base_a0_last, a0_last_modpast, expect_change=True))
    ok.append(report("2. change agent1 FUTURE -> agent0 earlier-col",
                     base_a0_earlier, a0_earlier_modlast, expect_change=False))
    ok.append(report("3. change agent1 SAMECOL-> agent0 last-col   ",
                     base_a0_last, a0_last_modlast, expect_change=args.allow_same_column))

    print("\n" + ("ALL CHECKS PASSED — agents share past context, no future/"
                  "same-column leakage." if all(ok) else "SOME CHECKS FAILED — see above."))


if __name__ == "__main__":
    main()
