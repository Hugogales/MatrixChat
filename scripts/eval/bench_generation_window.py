"""One-off benchmark: how does generate_matrix's wall-clock time scale with
max_new_tokens? By default this uses the inference KV-cache path; pass
``--no-kv-cache`` to measure the legacy full-sequence recompute behavior.

Not part of the regular eval suite -- ad hoc diagnostic, safe to delete.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.generation import generate_matrix
from scripts.eval.evaluate_checkpoint_suite import _build_multiturn_inputs
from scripts.eval.turn_taking_probe import load_model, resolve_device

# Mirrors demo_handoff_variation.py's "lengthy" context kind: representative
# of a mid-weight prompt (not the shortest cold-open, not the longest
# real_prefix), 3 agents, matching production sweep conditions.
_FILLER = [
    "yeah that makes sense to me",
    "i was thinking about that earlier too",
    "there is a lot to consider here",
    "we should probably look at this from a different angle",
    "i agree but i also see some issues",
    "let us keep going and see where this leads",
]
_PROMPT = "well that is my two cents on the matter what do you all think"


def build_scenario(tokenizer, num_agents, device):
    turns = [{"speaker": (i % (num_agents - 1)) + 1, "text": line, "visible_to": None}
             for i, line in enumerate(_FILLER)]
    turns.append({"speaker": 0, "text": _PROMPT, "visible_to": None})
    inp, mask, private_mask, visibility = _build_multiturn_inputs(tokenizer, turns, num_agents, device)
    return inp, mask, private_mask, visibility


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--token_values", default="24,48,96,144,192")
    p.add_argument("--repeats", type=int, default=2, help="Repeats per value, after 1 warmup call.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="Older GPUs (e.g. T4/V100) lack native bf16 tensor cores -- use float16 "
                        "there for a realistic timing, since this script is about RELATIVE "
                        "scaling with max_new_tokens, not matching production H100 absolute times.")
    p.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable incremental KV-cache generation and use full recompute.",
    )
    args = p.parse_args()

    device = resolve_device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[info] device={device}, dtype={dtype}", flush=True)
    model, tokenizer = load_model(args, device, dtype)
    print("[info] model loaded", flush=True)

    inp, mask, private_mask, visibility = build_scenario(tokenizer, args.num_agents, device)
    print(f"[info] prompt columns (T) = {inp.shape[2]}, num_agents (A) = {args.num_agents}, "
          f"flattened prompt length A*T = {inp.shape[2] * args.num_agents}", flush=True)

    token_values = [int(v) for v in args.token_values.split(",") if v.strip()]

    use_kv_cache = not args.no_kv_cache
    print(f"[info] use_kv_cache={use_kv_cache}", flush=True)

    print("[info] running warmup call...", flush=True)
    # Warmup (CUDA kernel compilation / allocator warmup shouldn't pollute timings).
    if device.type == "cuda":
        torch.cuda.synchronize()
    generate_matrix(model, inp, 8, input_activity_mask=mask, temperature=0.0,
                     placeholder_token_id=0, input_private_mask=private_mask,
                     agent_visibility=visibility, use_kv_cache=use_kv_cache)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[info] warmup done", flush=True)

    results = []
    for g in token_values:
        times = []
        for _ in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            generate_matrix(model, inp, g, input_activity_mask=mask, temperature=0.0,
                             placeholder_token_id=0, input_private_mask=private_mask,
                             agent_visibility=visibility, use_kv_cache=use_kv_cache)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)
        best = min(times)
        results.append((g, best, times))
        print(f"[result] max_new_tokens={g:4d}  best={best:7.3f}s  all={['%.3f' % t for t in times]}")

    print("\n[summary] max_new_tokens -> seconds, and ratio vs. the smallest value")
    base_g, base_t, _ = results[0]
    for g, t, _ in results:
        print(f"  {g:4d} tokens: {t:7.3f}s   ({t / base_t:5.2f}x time for {g / base_g:5.2f}x tokens)")


if __name__ == "__main__":
    main()
