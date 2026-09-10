#!/usr/bin/env python3
"""Factorial diagnostic for listener silence at a frozen checkpoint.

This is intentionally an evaluation experiment, not another training arm. It
isolates four variables currently confounded between the recurring in-training
probe and the broad sweep: agent count, activity threshold, generation
temperature, and rollout horizon.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.generation import generate_matrix, generation_topology_summary
from scripts.eval.demo_handoff_variation import _build_context
from scripts.eval.evaluate_checkpoint_suite import (
    HANDOFF_PROMPTS,
    _aggregate_generation_topology,
    _build_multiturn_inputs,
    _score_handoff,
)
from scripts.eval.turn_taking_probe import load_model, resolve_device


def csv_values(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--agent_counts", default="3,4")
    parser.add_argument("--activity_thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--temperatures", default="0.0,0.8")
    parser.add_argument("--horizons", default="16,48")
    parser.add_argument("--stochastic_seeds", default="0,1,2,3")
    parser.add_argument("--num_prompts", type=int, default=6)
    parser.add_argument(
        "--context_kinds",
        default="cold,primed_handoff_gap3,lengthy",
        help="Context slice run only at threshold 0.5, temperature 0.8, horizon 48.",
    )
    return parser.parse_args()


def evaluate_once(
    model,
    tokenizer,
    *,
    prompt: str,
    num_agents: int,
    context_kind: str,
    threshold: float,
    temperature: float,
    horizon: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    turns, reference_speaker, score_kind = _build_context(
        context_kind, prompt, num_agents
    )
    inputs, activity, private, visibility = _build_multiturn_inputs(
        tokenizer, turns, num_agents, next(model.parameters()).device
    )
    tokens, speak, probabilities = generate_matrix(
        model,
        inputs,
        horizon,
        input_activity_mask=activity,
        temperature=temperature,
        activity_threshold=threshold,
        placeholder_token_id=0,
        input_private_mask=private,
        agent_visibility=visibility,
        return_probs=True,
    )
    listeners = [agent for agent in range(num_agents) if agent != reference_speaker]
    (
        overlap,
        overlap_tokens,
        listener_spoke,
        first_listener,
        first_reference,
        last_reference,
        clean_handoff,
        interruption_handoff,
        dirty_handoff,
    ) = _score_handoff(speak[0], score_kind, reference_speaker, listeners)
    listener_trace = probabilities[0, listeners, :].mean(dim=0)
    reference_trace = probabilities[0, reference_speaker, :]
    generated_topology = generation_topology_summary(
        probabilities,
        speak,
        activity[:, :, -1],
    )
    listeners_spoke = int(
        sum(bool(speak[0, listener, :].any()) for listener in listeners)
    )
    return {
        "prompt": prompt,
        "num_agents": num_agents,
        "context_kind": context_kind,
        "activity_threshold": threshold,
        "temperature": temperature,
        "horizon": horizon,
        "seed": seed,
        "listener_spoke": listener_spoke,
        "listeners_spoke": listeners_spoke,
        "listener_agent_fraction": listeners_spoke / len(listeners),
        "clean_handoff": clean_handoff,
        "interruption_handoff": interruption_handoff,
        "dirty_handoff": dirty_handoff,
        "overlap": overlap,
        "overlap_tokens": overlap_tokens,
        "first_listener_step": first_listener,
        "first_reference_step": first_reference,
        "last_reference_step": last_reference,
        "listener_p_speak_initial": float(listener_trace[0]),
        "listener_p_speak_final": float(listener_trace[-1]),
        "listener_p_speak_max": float(listener_trace.max()),
        "listener_p_speak_delta": float(listener_trace[-1] - listener_trace[0]),
        "reference_p_speak_initial": float(reference_trace[0]),
        "reference_p_speak_final": float(reference_trace[-1]),
        "generation_topology": generated_topology,
    }


def summarize(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    summaries = []
    for values, items in sorted(groups.items()):
        count = len(items)
        summaries.append(
            {
                **dict(zip(group_keys, values)),
                "trials": count,
                "no_listener_response_rate": sum(
                    not item["listener_spoke"] for item in items
                )
                / count,
                "per_listener_agent_response_rate": sum(
                    item["listener_agent_fraction"] for item in items
                )
                / count,
                "clean_handoff_rate": sum(item["clean_handoff"] for item in items)
                / count,
                "interruption_handoff_rate": sum(
                    item["interruption_handoff"] for item in items
                )
                / count,
                "dirty_handoff_rate": sum(item["dirty_handoff"] for item in items)
                / count,
                "overlap_rate": sum(item["overlap"] for item in items) / count,
                "listener_p_speak_initial_mean": sum(
                    item["listener_p_speak_initial"] for item in items
                )
                / count,
                "listener_p_speak_final_mean": sum(
                    item["listener_p_speak_final"] for item in items
                )
                / count,
                "listener_p_speak_max_mean": sum(
                    item["listener_p_speak_max"] for item in items
                )
                / count,
                "generation_topology": _aggregate_generation_topology(items),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    agent_counts = csv_values(args.agent_counts, int)
    # ``turn_taking_probe.load_model`` uses this field to size max_agents.
    args.num_agents = max(agent_counts)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_model(args, device, dtype)

    prompts = HANDOFF_PROMPTS[: args.num_prompts]
    thresholds = csv_values(args.activity_thresholds, float)
    temperatures = csv_values(args.temperatures, float)
    horizons = csv_values(args.horizons, int)
    stochastic_seeds = csv_values(args.stochastic_seeds, int)

    factorial_rows = []
    for prompt in prompts:
        for num_agents in agent_counts:
            for threshold in thresholds:
                for temperature in temperatures:
                    seeds = [0] if temperature == 0 else stochastic_seeds
                    for horizon in horizons:
                        for seed in seeds:
                            factorial_rows.append(
                                evaluate_once(
                                    model,
                                    tokenizer,
                                    prompt=prompt,
                                    num_agents=num_agents,
                                    context_kind="cold",
                                    threshold=threshold,
                                    temperature=temperature,
                                    horizon=horizon,
                                    seed=seed,
                                )
                            )

    context_rows = []
    for prompt in prompts:
        for num_agents in agent_counts:
            for context_kind in csv_values(args.context_kinds, str):
                for seed in stochastic_seeds:
                    context_rows.append(
                        evaluate_once(
                            model,
                            tokenizer,
                            prompt=prompt,
                            num_agents=num_agents,
                            context_kind=context_kind,
                            threshold=0.5,
                            temperature=0.8,
                            horizon=48,
                            seed=seed,
                        )
                    )

    output = {
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "factorial_trials": len(factorial_rows),
        "context_trials": len(context_rows),
        "factorial_summary": summarize(
            factorial_rows,
            ("num_agents", "activity_threshold", "temperature", "horizon"),
        ),
        "context_summary": summarize(
            context_rows,
            ("num_agents", "context_kind"),
        ),
        "factorial_results": factorial_rows,
        "context_results": context_rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": args.output,
                "factorial_trials": len(factorial_rows),
                "context_trials": len(context_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
