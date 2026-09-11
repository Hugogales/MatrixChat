"""Ad hoc demo: how far does the P_kitchen_sink handoff behavior extend?

Same checkpoint as EXAMPLE_CLEAN_HANDOFF.txt, but sweeps DIFFERENT prompts,
DIFFERENT agent counts, DIFFERENT random seeds, and DIFFERENT preceding
CONTEXT (cold open, an in-context example of an already-completed handoff
with a 0- or 3-column silent gap between its two turns, right after a round
of introductions, after a lengthy multi-turn SYNTHETIC conversation, or after
a real multi-turn excerpt pulled straight from AMI/MELD/Werewolf) to see how
often a clean handoff actually happens, not just on one fixed example/context.

The broad sweep found context richness matters a lot (e.g. overlap jumped
from 18% cold to 91% after the synthetic "lengthy" filler) -- `real_prefix`
tests whether that holds with genuine prior dialogue instead of hand-written
filler, and `max_new_tokens` defaults much higher than before so a longer
stretch of the resulting "conversation" actually plays out.

On top of the single handoff/no-handoff decision, whenever a handoff IS
detected, generation is extended (by feeding the model's own output back in
as context) to see whether the conversation can chain into a SECOND and even
THIRD handoff, rather than stopping after one.
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
from scripts.eval.evaluate_checkpoint_suite import (
    _build_multiturn_inputs,
    _score_handoff,
    repetition_stats,
)
from scripts.eval.turn_taking_probe import load_model, resolve_device

PROMPTS = [
    "well that is my two cents on the matter what do you all think",
    "i think we covered everything on my end anyone want to jump in",
    "so that is basically the whole story does that make sense to everyone",
    "alright i am done talking now who else has something to add",
    "that is my final answer so what is everyone else voting for",
    "i will stop there and let someone else take it from here",
]
AGENT_COUNTS = [2, 3, 4, 5]
SEEDS = [0, 1, 2, 3]

# A fixed, generic "first handoff" line used to open the primed-handoff
# contexts -- agent 0 speaks it, then agent 1 "takes over" and poses the
# swept PROMPT as its own handoff-inviting follow-up, simulating that one
# handoff has already happened before generation even starts.
_PRIMED_FIRST_LINE = "honestly i think we should go with the second option so what does everyone else think"

# Generic, topic-neutral filler used to build a "lengthy conversation"
# context: agents other than 0 talk for a while before agent 0 finally
# speaks the swept PROMPT, having been silent throughout.
_LENGTHY_FILLER = [
    "yeah that makes sense to me",
    "i was thinking about that earlier too",
    "there is a lot to consider here",
    "we should probably look at this from a different angle",
    "i agree but i also see some issues",
    "let us keep going and see where this leads",
]

CONTEXT_KINDS = [
    "cold",
    "primed_handoff_gap0",
    "primed_handoff_gap3",
    "post_intro",
    "lengthy",
    "real_prefix",
]

# Lazily populated by _load_real_excerpts(): a handful of genuine multi-turn
# excerpts (one per source where available), each a list of {"speaker": int,
# "text": str} dicts already using small, 0-based speaker indices. Loaded
# once per process since it involves parsing raw dataset files.
_REAL_EXCERPTS = None


def _load_real_excerpts(raw_dir=None, excerpt_len=10):
    """One real multi-turn excerpt per available source (AMI, MELD, Werewolf),
    taken from partway into a real conversation (not its very first turns) so
    it reads like "having been in the room for a while", not a cold open.

    Each excerpt's speakers are remapped to 0-based indices in order of first
    appearance within the excerpt (matching how the adapters already handle
    agent numbering), independent of the source's original speaker count.
    Returns ``[]`` if raw data isn't available (offline dev, missing files) --
    callers should fall back to a synthetic context kind in that case.
    """
    global _REAL_EXCERPTS
    if _REAL_EXCERPTS is not None:
        return _REAL_EXCERPTS

    raw_dir = raw_dir or os.path.join(os.path.expanduser("~"), "matrixchat", "raw")
    excerpts = []
    from data.adapters import ami, meld, werewolf

    for adapter in (ami, meld, werewolf):
        try:
            for conv in adapter.iter_conversations(raw_dir):
                if len(conv.turns) < excerpt_len + 4:
                    continue
                start = len(conv.turns) // 3  # partway in, not the very start
                window = conv.turns[start:start + excerpt_len]
                seen = {}
                remapped = []
                for turn in window:
                    if turn.speaker not in seen:
                        seen[turn.speaker] = len(seen)
                    text = (turn.text or "").strip()
                    if text:
                        remapped.append({"speaker": seen[turn.speaker], "text": text, "visible_to": None})
                if len(remapped) >= 4 and len({t["speaker"] for t in remapped}) >= 2:
                    excerpts.append(remapped)
                    break  # one excerpt per source is enough
        except Exception as exc:  # offline/missing-data safety net
            print(f"[warn] could not load a real excerpt from {adapter.__name__}: {exc}")
    _REAL_EXCERPTS = excerpts
    return excerpts


def _build_context(kind, prompt, num_agents):
    """Return ``(turns, reference_speaker, score_kind)`` for one context kind.

    ``turns`` is the prelude+prompt fed to ``_build_multiturn_inputs``;
    ``reference_speaker`` is whoever is about to yield (scored against);
    ``score_kind`` controls ``_score_handoff``'s clean-handoff eligibility.
    """
    if kind == "cold":
        return [{"speaker": 0, "text": prompt, "visible_to": None}], 0, "handoff"

    if kind in ("primed_handoff_gap0", "primed_handoff_gap3"):
        gap = 0 if kind == "primed_handoff_gap0" else 3
        turns = [
            {"speaker": 0, "text": _PRIMED_FIRST_LINE, "visible_to": None},
            {"silence_columns": gap},
            {"speaker": 1, "text": prompt, "visible_to": None},
        ]
        # Agent 1 already "took over" once; we're testing whether a SECOND
        # handoff (away from agent 1) happens.
        return turns, 1, "primed_handoff"

    if kind == "post_intro":
        turns = [
            {"speaker": a, "text": f"hi everyone i am speaker {a}", "visible_to": None}
            for a in range(num_agents)
        ]
        turns.append({"speaker": 0, "text": prompt, "visible_to": None})
        return turns, 0, "handoff"

    if kind == "lengthy":
        turns = []
        others = [a for a in range(num_agents) if a != 0] or [0]
        for i, line in enumerate(_LENGTHY_FILLER):
            turns.append({"speaker": others[i % len(others)], "text": line, "visible_to": None})
        turns.append({"speaker": 0, "text": prompt, "visible_to": None})
        return turns, 0, "handoff"

    if kind == "real_prefix":
        excerpts = _load_real_excerpts()
        if not excerpts:
            # No raw data available (e.g. offline dev) -- fall back to the
            # synthetic "lengthy" kind rather than erroring the whole sweep.
            return _build_context("lengthy", prompt, num_agents)
        excerpt = excerpts[hash(prompt) % len(excerpts)]
        turns = [
            {"speaker": t["speaker"] % num_agents, "text": t["text"], "visible_to": None}
            for t in excerpt
        ]
        reference_speaker = turns[-1]["speaker"]
        # The most recent real speaker "continues" with our swept handoff-inviting
        # line -- tests whether genuine prior dialogue changes handoff behavior
        # compared to the hand-written "lengthy" filler above.
        turns.append({"speaker": reference_speaker, "text": prompt, "visible_to": None})
        return turns, reference_speaker, "handoff"

    raise ValueError(f"unknown context kind: {kind}")


def _last_speaking_agents(speak):
    """``speak``: ``[A, T]`` bool. Agent indices active in the LAST column
    that has anyone active at all (empty list if the whole window is silent).
    """
    any_active = speak.any(dim=0)
    if not bool(any_active.any()):
        return []
    last_step = int(any_active.nonzero().max())
    return [a for a in range(speak.shape[0]) if bool(speak[a, last_step])]


def _decode_window(tokenizer, tokens, speak, num_agents):
    decoded = []
    for a in range(num_agents):
        ids = [int(tokens[0, a, s]) for s in range(tokens.shape[2]) if bool(speak[0, a, s])]
        decoded.append(tokenizer.decode(ids, skip_special_tokens=True))
    return decoded


def _ids_window(tokens, speak, num_agents):
    return [
        [
            int(tokens[0, agent, step])
            for step in range(tokens.shape[2])
            if bool(speak[0, agent, step])
        ]
        for agent in range(num_agents)
    ]


def _matrix_cells(tokenizer, tokens, speak, num_agents):
    """JSON-friendly time-major cells for judge/rendering consumers."""
    rows = []
    for step in range(tokens.shape[2]):
        rows.append([
            tokenizer.decode([int(tokens[0, agent, step])], skip_special_tokens=True)
            if bool(speak[0, agent, step])
            else ""
            for agent in range(num_agents)
        ])
    return rows


def _run_chain(model, tokenizer, args, inp, mask, private_mask, visibility, num_agents,
               reference_speaker, score_kind, max_links=3):
    """Generate one window; if a handoff away from ``reference_speaker`` is
    found -- clean OR a brief interruption (1–4 overlap columns) -- extend
    the context with the model's own output and generate again, looking for
    a SECOND and THIRD handoff in the same rollout. Dirty 5+ overlap stops
    the chain. Returns a list of per-window result dicts (length = however
    many links actually occurred, 0 if the first window has no listener
    response at all).

    IMPORTANT (08-05 fix): earlier versions stopped the rollout the instant
    ANY overlap occurred, even a single simultaneous token -- this silently
    prevented the generation from ever continuing past a minor blip, so we
    could never observe whether the conversation would have naturally kept
    going into a real 2nd/3rd handoff. Only genuinely sustained overlap
    (beyond the dirty-handoff tolerance) now stops the rollout; a brief,
    quickly-resolved overlap is treated as a real (if imperfect) handoff and
    the chain keeps extending.
    """
    cur_inp, cur_mask, cur_private = inp, mask, private_mask
    ref = reference_speaker
    kind = score_kind
    chain = []
    for _link in range(max_links):
        other_agents = [a for a in range(num_agents) if a != ref]
        tokens, speak, probs = generate_matrix(
            model, cur_inp, args.max_new_tokens,
            input_activity_mask=cur_mask, temperature=args.temperature,
            activity_threshold=args.activity_threshold, placeholder_token_id=0,
            input_private_mask=cur_private, agent_visibility=visibility,
            return_probs=True,
        )
        overlap, overlap_tokens, listener_any, first_listener, first_ref, last_ref, clean_handoff, interruption_handoff, dirty_handoff = _score_handoff(
            speak[0], kind, ref, other_agents,
        )
        decoded = _decode_window(tokenizer, tokens, speak, num_agents)
        ids = _ids_window(tokens, speak, num_agents)
        chain.append({
            "reference_speaker": ref, "overlap": overlap, "overlap_tokens": overlap_tokens,
            "listener_spoke": listener_any,
            "clean_handoff": clean_handoff,
            "interruption_handoff": interruption_handoff,
            "dirty_handoff": dirty_handoff,
            "outputs": decoded,
            "activity_rows": [
                "".join("X" if bool(speak[0, agent, step]) else "."
                        for step in range(speak.shape[2]))
                for agent in range(num_agents)
            ],
            "matrix_cells": _matrix_cells(tokenizer, tokens, speak, num_agents),
            "ids": ids,
        })
        # Continue on a clean yield or a brief interruption (1–4 overlap
        # columns). Stop on silence, unresolved overlap, or a dirty 5+ overlap.
        if not (clean_handoff or interruption_handoff):
            break
        new_holders = _last_speaking_agents(speak[0])
        if len(new_holders) != 1:
            break
        cur_inp = torch.cat([cur_inp, tokens], dim=2)
        cur_mask = torch.cat([cur_mask, speak], dim=2)
        if cur_private is not None:
            cur_private = torch.cat(
                [cur_private, torch.zeros_like(speak, dtype=torch.bool)], dim=2,
            )
        ref = new_holders[0]
        kind = "primed_handoff"  # subsequent links are always "already primed" by definition
    return chain


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--output", required=True)
    p.add_argument("--max_new_tokens", type=int, default=48,
                    help="Raised from the original 16 so a longer stretch of the "
                         "resulting conversation actually plays out per window, "
                         "closer to simulating a real end-to-end exchange rather "
                         "than a single immediate reaction.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--activity_threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_chain_links", type=int, default=3,
                    help="How many chained handoffs to look for beyond the first.")
    p.add_argument("--num_agents", type=int, default=8,
                    help="Ceiling for the model config's max_agents; actual per-trial "
                         "agent counts are swept via AGENT_COUNTS, not this flag.")
    p.add_argument("--resume_from", default=None,
                    help="Path to a previous run's `<output>.partial.json` (or a prior "
                         "completed `--output`) to resume from -- skips any "
                         "(prompt, num_agents, seed, kind) combo already present, so a "
                         "job that hits the sbatch time limit mid-sweep (see "
                         "2026-08-25 TRAINING_RUN_LOG entry: a full 4h run lost 100% of "
                         "its ~50%-complete progress with no resume support) doesn't "
                         "have to restart from trial 0.")
    p.add_argument("--partial_save_every", type=int, default=48,
                    help="Dump in-progress results to `<output>.partial.json` every N "
                         "trials, so a mid-sweep timeout/crash still leaves resumable "
                         "progress on disk instead of losing the whole run.")
    return p.parse_args()


def _load_resume_results(resume_from: str) -> list:
    with open(resume_from) as f:
        data = json.load(f)
    # Accept either a bare partial-results list or a full summary dict with
    # a "results" key (i.e. resuming from a completed prior run's output).
    return data["results"] if isinstance(data, dict) else data


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_model(args, device, dtype)

    results = []
    done = set()
    if args.resume_from:
        results = _load_resume_results(args.resume_from)
        done = {(r["prompt"], r["num_agents"], r["seed"], r["context_kind"]) for r in results}
        print(f"[resume] loaded {len(results)} completed trials from {args.resume_from}")

    partial_path = args.output + ".partial.json"
    trials_since_save = 0
    for prompt in PROMPTS:
        for num_agents in AGENT_COUNTS:
            for seed in SEEDS:
                for kind in CONTEXT_KINDS:
                    if (prompt, num_agents, seed, kind) in done:
                        continue
                    torch.manual_seed(seed)
                    turns, reference_speaker, score_kind = _build_context(kind, prompt, num_agents)
                    inp, mask, private_mask, visibility = _build_multiturn_inputs(
                        tokenizer, turns, num_agents, device,
                    )
                    chain = _run_chain(
                        model, tokenizer, args, inp, mask, private_mask, visibility,
                        num_agents, reference_speaker, score_kind, max_links=args.max_chain_links,
                    )
                    first = chain[0] if chain else {
                        "overlap": False, "dirty_handoff": False, "interruption_handoff": False,
                        "listener_spoke": False, "clean_handoff": False, "outputs": [],
                    }
                    # Strict chain: only fully-clean (zero-overlap) links count,
                    # unchanged from the original definition so past broad-sweep
                    # results stay comparable.
                    chain_length = 0
                    for link in chain:
                        if link["clean_handoff"]:
                            chain_length += 1
                        else:
                            break
                    # Lenient chain: a brief interruption (1–4 overlap
                    # columns) also continues the chain. Dirty 5+ overlap
                    # or silence breaks it. Reported alongside strict.
                    chain_length_lenient = 0
                    for link in chain:
                        if link["clean_handoff"] or link.get("interruption_handoff"):
                            chain_length_lenient += 1
                        else:
                            break
                    row = {
                        "prompt": prompt, "num_agents": num_agents, "seed": seed, "context_kind": kind,
                        "clean_handoff": first["clean_handoff"],
                        "interruption_handoff": first.get("interruption_handoff", False),
                        "dirty_handoff": first["dirty_handoff"],
                        "listener_spoke": first["listener_spoke"],
                        "overlap": first["overlap"], "chain_length": chain_length,
                        "chain_length_lenient": chain_length_lenient,
                        "chain": chain,
                    }
                    results.append(row)
                    trials_since_save += 1
                    if trials_since_save >= args.partial_save_every:
                        trials_since_save = 0
                        os.makedirs(os.path.dirname(partial_path) or ".", exist_ok=True)
                        with open(partial_path, "w") as f:
                            json.dump(results, f)
                    tag = "CLEAN" if first["clean_handoff"] else (
                        "INTERRUPTION" if first.get("interruption_handoff") else
                        ("DIRTY" if first["dirty_handoff"] else
                        ("OVERLAP" if first["overlap"] else ("SILENT" if not first["listener_spoke"] else "MIXED")))
                    )
                    extra = f" chain={chain_length}/{chain_length_lenient}" if chain_length_lenient > 1 else ""
                    print(f"[{tag}]{extra} kind={kind} prompt={prompt[:30]!r} n={num_agents} seed={seed} "
                          f"-> {first['outputs']}")

    n_clean = sum(r["clean_handoff"] for r in results)
    n_interruption = sum(r.get("interruption_handoff") for r in results)
    n_dirty = sum(r["dirty_handoff"] for r in results)
    n_overlap = sum(r["overlap"] for r in results)
    n_silent = sum(not r["listener_spoke"] for r in results)
    n_chain2plus = sum(r["chain_length"] >= 2 for r in results)
    n_chain3plus = sum(r["chain_length"] >= 3 for r in results)
    n_chain2plus_lenient = sum(r["chain_length_lenient"] >= 2 for r in results)
    n_chain3plus_lenient = sum(r["chain_length_lenient"] >= 3 for r in results)
    by_kind = {}
    for kind in CONTEXT_KINDS:
        subset = [r for r in results if r["context_kind"] == kind]
        by_kind[kind] = {
            "total_trials": len(subset),
            "clean_handoff_rate": sum(r["clean_handoff"] for r in subset) / max(len(subset), 1),
            "interruption_handoff_rate": sum(r.get("interruption_handoff") for r in subset) / max(len(subset), 1),
            "dirty_handoff_rate": sum(r["dirty_handoff"] for r in subset) / max(len(subset), 1),
            "handoff_rate_lenient": sum(
                r["clean_handoff"] or r.get("interruption_handoff") for r in subset
            ) / max(len(subset), 1),
            "overlap_rate": sum(r["overlap"] for r in subset) / max(len(subset), 1),
            "no_listener_response_rate": sum(not r["listener_spoke"] for r in subset) / max(len(subset), 1),
            "listener_response_rate": sum(r["listener_spoke"] for r in subset) / max(len(subset), 1),
            "nonoverlap_listener_rate": sum(
                r["listener_spoke"] and not r["overlap"] for r in subset
            ) / max(len(subset), 1),
            "chain_2plus_rate": sum(r["chain_length"] >= 2 for r in subset) / max(len(subset), 1),
            "chain_2plus_rate_lenient": sum(
                r["chain_length_lenient"] >= 2 for r in subset
            ) / max(len(subset), 1),
        }
    summary = {
        "checkpoint_dir": args.checkpoint_dir,
        "total_trials": len(results),
        "clean_handoff_rate": n_clean / len(results),
        "interruption_handoff_rate": n_interruption / len(results),
        "dirty_handoff_rate": n_dirty / len(results),
        "handoff_rate_lenient": (n_clean + n_interruption) / len(results),
        "overlap_rate": n_overlap / len(results),
        "no_listener_response_rate": n_silent / len(results),
        "listener_response_rate": sum(r["listener_spoke"] for r in results) / len(results),
        "nonoverlap_listener_rate": sum(
            r["listener_spoke"] and not r["overlap"] for r in results
        ) / len(results),
        "chain_2plus_rate": n_chain2plus / len(results),
        "chain_3plus_rate": n_chain3plus / len(results),
        "chain_2plus_rate_lenient": n_chain2plus_lenient / len(results),
        "chain_3plus_rate_lenient": n_chain3plus_lenient / len(results),
        **repetition_stats([
            ids
            for result in results
            for link in result.get("chain", [])
            for ids in link.get("ids", [])
        ]),
        "by_context_kind": by_kind,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
