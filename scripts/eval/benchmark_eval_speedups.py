#!/usr/bin/env python3
"""Correctness + wall-clock benchmark for eval speed optimizations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.eval_batching import EvalTrialSpec, bucket_eval_specs, collate_prefix_batch
from model.generation import generate_matrix
from scripts.eval.turn_taking_probe import load_model, resolve_device


def _load_examples_from_jsonl(path: Path, max_examples: int) -> list[dict]:
    examples: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prefix = record["prefix"]
            cut = record["cut"]
            agents = len(prefix["token_ids"])
            prefix_len = int(cut["column"])
            examples.append(
                {
                    "trial_id": record["trial_id"],
                    "agents": agents,
                    "continuation_columns": int(cut["continuation_columns"]),
                    "prefix_ids": [row[:prefix_len] for row in prefix["token_ids"]],
                    "prefix_activity": [row[:prefix_len] for row in prefix["activity"]],
                    "prefix_private": [row[:prefix_len] for row in prefix["is_private_mask"]],
                    "visibility": prefix["agent_visibility"],
                }
            )
            if len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError(f"no examples found in {path}")
    return examples


def _example_tensors(example: dict, device: torch.device) -> dict:
    return {
        "agents": example["agents"],
        "continuation_columns": example["continuation_columns"],
        "input_ids": torch.tensor([example["prefix_ids"]], dtype=torch.long, device=device),
        "input_activity": torch.tensor(
            [example["prefix_activity"]], dtype=torch.bool, device=device
        ),
        "input_private": torch.tensor(
            [example["prefix_private"]], dtype=torch.bool, device=device
        ),
        "agent_visibility": torch.tensor(
            [example["visibility"]], dtype=torch.bool, device=device
        ),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_generate(
    model,
    input_ids,
    max_new_tokens: int,
    *,
    input_activity_mask,
    input_private_mask,
    agent_visibility,
    use_kv_cache: bool,
    repeats: int,
    device: torch.device,
) -> list[float]:
    times: list[float] = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        generate_matrix(
            model,
            input_ids,
            max_new_tokens,
            input_activity_mask=input_activity_mask,
            temperature=0.0,
            placeholder_token_id=0,
            input_private_mask=input_private_mask,
            agent_visibility=agent_visibility,
            use_kv_cache=use_kv_cache,
        )
        _sync(device)
        times.append(time.perf_counter() - start)
    return times


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-path", default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--examples-jsonl",
        default="paper_results_adaptive_a075_20260830/paired_continuations.jsonl",
        help="Existing eval JSONL to reuse realistic prefix shapes.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--continuation-columns", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    loader_args = SimpleNamespace(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        num_agents=8,
    )

    results: dict[str, object] = {
        "checkpoint_dir": args.checkpoint_dir,
        "device": str(device),
        "dtype": str(dtype),
    }

    print("[1/4] loading eager model...", flush=True)
    model_eager, _ = load_model(loader_args, device, dtype, attn_implementation="eager")
    print("[2/4] loading sdpa model...", flush=True)
    model_sdpa, _ = load_model(loader_args, device, dtype, attn_implementation="sdpa")

    examples = _load_examples_from_jsonl(Path(args.examples_jsonl), args.max_examples)
    first = _example_tensors(examples[0], device)
    inp = first["input_ids"]
    mask = first["input_activity"]
    private = first["input_private"]
    visibility = first["agent_visibility"]
    cont_cols = first["continuation_columns"]

    print("[3/4] parity: eager sdpa cached vs uncached...", flush=True)
    with torch.no_grad():
        eager_uncached = generate_matrix(
            model_eager, inp, min(8, cont_cols), input_activity_mask=mask,
            input_private_mask=private, agent_visibility=visibility,
            temperature=0.0, placeholder_token_id=0, use_kv_cache=False,
        )
        eager_cached = generate_matrix(
            model_eager, inp, min(8, cont_cols), input_activity_mask=mask,
            input_private_mask=private, agent_visibility=visibility,
            temperature=0.0, placeholder_token_id=0, use_kv_cache=True,
        )
        sdpa_cached = generate_matrix(
            model_sdpa, inp, min(8, cont_cols), input_activity_mask=mask,
            input_private_mask=private, agent_visibility=visibility,
            temperature=0.0, placeholder_token_id=0, use_kv_cache=True,
        )
    parity = {
        "eager_cached_matches_uncached": bool(
            torch.equal(eager_cached[0], eager_uncached[0])
            and torch.equal(eager_cached[1], eager_uncached[1])
        ),
        "sdpa_cached_matches_eager_uncached": bool(
            torch.equal(sdpa_cached[0], eager_uncached[0])
            and torch.equal(sdpa_cached[1], eager_uncached[1])
        ),
    }
    results["parity"] = parity
    print(json.dumps(parity, indent=2), flush=True)
    if not all(parity.values()):
        raise SystemExit("parity check failed")

    bench_cols = min(args.continuation_columns, cont_cols)
    print(f"[4/4] timing generate_matrix ({bench_cols} columns)...", flush=True)
    timing = {}
    for label, model, use_kv in [
        ("eager_uncached", model_eager, False),
        ("eager_cached", model_eager, True),
        ("sdpa_cached", model_sdpa, True),
    ]:
        times = _time_generate(
            model, inp, bench_cols,
            input_activity_mask=mask,
            input_private_mask=private,
            agent_visibility=visibility,
            use_kv_cache=use_kv,
            repeats=args.repeats,
            device=device,
        )
        timing[label] = {"seconds": times, "best": min(times), "mean": sum(times) / len(times)}
        print(f"  {label}: best={timing[label]['best']:.3f}s mean={timing[label]['mean']:.3f}s", flush=True)

    results["generate_matrix"] = timing
    base = timing["eager_uncached"]["best"]
    results["speedup_vs_eager_uncached"] = {
        key: round(base / value["best"], 2) for key, value in timing.items()
    }

    print("[5/5] timing held-out eval paths...", flush=True)
    serial_times: list[float] = []
    for example in examples:
        prefix = _example_tensors(example, device)
        _sync(device)
        start = time.perf_counter()
        generate_matrix(
            model_sdpa,
            prefix["input_ids"],
            prefix["continuation_columns"],
            input_activity_mask=prefix["input_activity"],
            input_private_mask=prefix["input_private"],
            agent_visibility=prefix["agent_visibility"],
            temperature=0.0,
            placeholder_token_id=0,
            use_kv_cache=True,
        )
        _sync(device)
        serial_times.append(time.perf_counter() - start)

    eval_specs = [
        EvalTrialSpec(
            trial_id=example["trial_id"],
            entry={
                "num_agents": example["agents"],
                "length": len(example["prefix_ids"][0]) + example["continuation_columns"],
            },
            row={
                "input_ids": [
                    row + [0] * example["continuation_columns"]
                    for row in example["prefix_ids"]
                ],
                "input_activity_mask": [
                    row + [False] * example["continuation_columns"]
                    for row in example["prefix_activity"]
                ],
                "is_private_mask": [
                    row + [False] * example["continuation_columns"]
                    for row in example["prefix_private"]
                ],
                "agent_visibility": example["visibility"],
            },
            cut={
                "column": len(example["prefix_ids"][0]),
                "fraction": 0.5,
            },
            seed=0,
        )
        for example in examples
    ]

    batch_times: list[float] = []
    batches = bucket_eval_specs(eval_specs, batch_size=args.eval_batch_size)
    for batch in batches:
        collated = collate_prefix_batch(
            batch.specs, placeholder_token_id=0, device=device
        )
        _sync(device)
        start = time.perf_counter()
        generate_matrix(
            model_sdpa,
            collated["input_ids"],
            collated["continuation_columns"],
            input_activity_mask=collated["input_activity"],
            input_private_mask=collated["input_private"],
            agent_visibility=collated["agent_visibility"],
            temperature=0.0,
            placeholder_token_id=0,
            use_kv_cache=True,
        )
        _sync(device)
        batch_times.append(time.perf_counter() - start)

    results["held_out_eval"] = {
        "examples": len(eval_specs),
        "serial_total_sec": sum(serial_times),
        "serial_mean_sec": sum(serial_times) / len(serial_times),
        "batched_total_sec": sum(batch_times),
        "batched_mean_sec": sum(batch_times) / len(eval_specs),
        "batch_size": args.eval_batch_size,
    }
    results["held_out_eval"]["batch_speedup"] = round(
        results["held_out_eval"]["serial_total_sec"]
        / max(results["held_out_eval"]["batched_total_sec"], 1e-9),
        2,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"[done] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
