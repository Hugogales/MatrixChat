#!/usr/bin/env python3
"""Run paired human-vs-model continuation evaluation on a held-out contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.judge_adapter import (
    HELD_OUT_JUDGE_SYSTEM,
    judge_paired_continuations,
    make_client,
)
from evaluation.paired_continuation import (
    checkpoint_identifier,
    load_contract,
    load_referenced_rows,
    run_paired_continuation_evaluation,
)
from scripts.eval.turn_taking_probe import load_model, resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--contract", required=True, help="Evaluation contract JSON.")
    parser.add_argument("--output", required=True, help="Append/resume JSONL output.")
    parser.add_argument("--processed-dir", default=None, help="Override contract processed_dir.")
    parser.add_argument(
        "--split", choices=("development", "final_test"), default="development"
    )
    parser.add_argument(
        "--allow-final-test",
        action="store_true",
        help="Required explicit authorization for the sealed final_test split.",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--model-path", default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--protocol",
        choices=("matrix", "round_robin", "xml_round_robin", "pass_the_baton"),
        default="matrix",
        help=(
            "matrix: MatrixChat generate_matrix. "
            "round_robin: unfinetuned Qwen, one speaker per column. "
            "xml_round_robin: XML-tagged full turns in roster order. "
            "pass_the_baton: XML-tagged turns with hidden next-speaker routing."
        ),
    )
    parser.add_argument(
        "--round-robin-start",
        default="next_after_cut_reference",
        help="Round-robin start rule hashed into generation identity.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument("--placeholder-token-id", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.4, 0.5], choices=(0.4, 0.5)
    )
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--summary", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--judge", action="store_true", help="Enable optional paired judging.")
    parser.add_argument(
        "--judge-base-url", default="http://dh-dgxh100-2.hpc.msoe.edu:8000/v1"
    )
    parser.add_argument(
        "--judge-model", default="meta/llama-4-scout-17b-16e-instruct"
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Attention backend for MatrixChat eval loading (training stays eager).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Micro-batch size for shape-compatible held-out trials.",
    )
    parser.add_argument(
        "--trust-frozen-contract",
        action="store_true",
        help=(
            "Skip implementation-hash drift checks (convert.py edits after the "
            "contract was frozen). Safe for development screening when manifest "
            "and row content hashes still match."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    contract, processed_dir, contract_id = load_contract(
        args.contract,
        split=args.split,
        allow_final_test=args.allow_final_test,
        processed_dir=args.processed_dir,
    )
    # Complete the integrity preflight before allocating/loading a model.
    metadata_path = Path(args.metadata) if args.metadata else Path(args.output).with_suffix(".metadata.json")
    output_path = Path(args.output)
    trust_frozen_contract = bool(args.trust_frozen_contract)
    if not trust_frozen_contract and metadata_path.exists() and output_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(previous.get("contract_id") or "") == contract_id:
            trust_frozen_contract = True
    referenced = load_referenced_rows(
        contract,
        processed_dir,
        split=args.split,
        trust_frozen_contract=trust_frozen_contract,
    )
    vanilla_protocols = {"round_robin", "xml_round_robin", "pass_the_baton"}
    if args.protocol in vanilla_protocols and args.checkpoint_dir:
        raise SystemExit(
            f"{args.protocol} uses unfinetuned Qwen; omit --checkpoint-dir"
        )
    checkpoint_id = checkpoint_identifier(
        args.checkpoint_dir, model_path=args.model_path
    )
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    generate_fn = None
    if args.protocol in vanilla_protocols:
        from model.generation_round_robin import (
            generate_round_robin_vanilla,
            load_vanilla_causal_lm,
        )
        from model.generation_xml_turns import make_xml_generate_fn

        model, tokenizer = load_vanilla_causal_lm(
            args.model_path,
            device,
            dtype,
            attn_implementation="sdpa" if args.protocol != "round_robin" else "eager",
        )
        if args.protocol == "round_robin":
            generate_fn = generate_round_robin_vanilla
        else:
            generate_fn = make_xml_generate_fn(tokenizer, args.protocol)
    else:
        max_agents = max(
            (int(entry["num_agents"]) for entry, _ in referenced),
            default=1,
        )
        loader_args = SimpleNamespace(
            model_path=args.model_path,
            checkpoint_dir=args.checkpoint_dir,
            num_agents=max_agents,
        )
        model, tokenizer = load_model(
            loader_args, device, dtype, attn_implementation=args.attn_implementation
        )

    judge_fn = None
    if args.judge:
        client = make_client(args.judge_base_url)

        def judge_fn(record):
            return judge_paired_continuations(client, args.judge_model, record)

    summary = run_paired_continuation_evaluation(
        model=model,
        tokenizer=tokenizer,
        contract=contract,
        processed_dir=processed_dir,
        output_path=args.output,
        contract_id=contract_id,
        checkpoint_id=checkpoint_id,
        split=args.split,
        allow_final_test=args.allow_final_test,
        fractions=args.fractions,
        generation_seeds=args.generation_seeds,
        max_examples=args.max_examples,
        temperature=args.temperature,
        activity_threshold=args.activity_threshold,
        placeholder_token_id=args.placeholder_token_id,
        device=device,
        generate_fn=generate_fn,
        protocol=args.protocol,
        round_robin_start=args.round_robin_start,
        judge_fn=judge_fn,
        judge_config=(
            {
                "base_url": args.judge_base_url,
                "model": args.judge_model,
                "rubric_prompt_sha256": hashlib.sha256(
                    HELD_OUT_JUDGE_SYSTEM.encode("utf-8")
                ).hexdigest(),
            }
            if args.judge
            else None
        ),
        summary_path=args.summary,
        metadata_path=args.metadata,
        eval_batch_size=args.eval_batch_size,
        trust_frozen_contract=trust_frozen_contract,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
