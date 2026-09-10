#!/usr/bin/env python3
"""Profile short slices of common MatrixChat training and evaluation workflows.

Runs ~10 optimizer steps of training and ~10 held-out continuation examples,
records wall-clock phase timings, cProfile hotspots, and (on CUDA) a torch
profiler table for the heaviest GPU regions.

Example (login node, CPU smoke):
    python scripts/analysis/profile_workflows.py --mode tiny --output-dir /tmp/mc_profile

Example (GPU, production-shaped):
    sbatch scripts/analysis/profile_workflows.sbatch

Outputs under ``<output-dir>/``:
    SUMMARY.md
    training_cprofile.txt
    eval_cprofile.txt
    eval_per_example.jsonl
    generate_matrix_per_column.json
    training_torch_cuda.txt   (CUDA only)
    eval_torch_cuda.txt       (CUDA only)
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_CHECKPOINT = "checkpoints/adaptive_hfreq_a075_s176106/best_probe"
DEFAULT_CONTRACT = "logs/final_runs_20260827_recontract/development_screening.json"
DEFAULT_PROCESSED = "/home/ad.msoe.edu/garrido-lestacheh/matrixchat/processed_final_20260814_lookback192_allspk"


@dataclass
class PhaseTiming:
    name: str
    seconds: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowReport:
    workflow: str
    total_seconds: float
    phases: list[PhaseTiming]
    notes: list[str] = field(default_factory=list)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _format_cprofile(prof: cProfile.Profile, *, top: int = 40) -> str:
    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream)
    stats.sort_stats("cumtime")
    stats.print_stats(top)
    stream.write("\n--- by internal time (tottime) ---\n")
    stats.sort_stats("tottime")
    stats.print_stats(top)
    return stream.getvalue()


@contextmanager
def _cprofile_capture():
    prof = cProfile.Profile()
    prof.enable()
    try:
        yield prof
    finally:
        prof.disable()


def _maybe_torch_cuda_profile(fn: Callable[[], None], out_path: Path) -> str | None:
    if not torch.cuda.is_available():
        return None
    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        with record_function("workflow_body"):
            fn()
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=40,
        max_name_column_width=90,
    )
    cpu_table = prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=25,
        max_name_column_width=90,
    )
    text = "=== CUDA time (top 40) ===\n" + table + "\n\n=== CPU time (top 25) ===\n" + cpu_table
    _write_text(out_path, text)
    return text


def _training_argv(
    *,
    mode: str,
    steps: int,
    run_dir: Path,
    checkpoint: str | None,
    processed_dir: str,
) -> list[str]:
    argv = [
        "--num_steps",
        str(steps),
        "--val_fraction",
        "0",
        "--eval_every",
        "1000000",
        "--probe_every",
        "0",
        "--save_checkpoint",
        "false",
        "--run_name",
        "profile_workflow",
    ]
    if mode == "tiny":
        argv.extend(
            [
                "--use_tiny_model",
                "true",
                "--device",
                "cpu",
                "--torch_dtype",
                "float32",
                "--num_agents",
                "2",
                "--seq_len",
                "8",
                "--batch_size",
                "2",
            ]
        )
        return argv

    argv.extend(
        [
            "--use_tiny_model",
            "false",
            "--device",
            "cuda",
            "--torch_dtype",
            "bfloat16",
            "--model_name",
            "models/Qwen3-4B-Instruct-2507",
            "--processed_data_dir",
            processed_dir,
            "--dataset_weights",
            "ami=0.5,meld=0.15,werewolf=0.35",
            "--sampling_strategy",
            "balanced_cycle",
            "--seed",
            "176106",
            "--max_agents",
            "8",
            "--num_agents",
            "4",
            "--position_mode",
            "column",
            "--use_agent_embeddings",
            "true",
            "--agent_attention_mode",
            "qkv",
            "--agent_attention_representation",
            "learned",
            "--agent_attention_dim",
            "16",
            "--agent_dynamic_state_mode",
            "gru",
            "--turn_reward_mode",
            "balanced_ce",
            "--floor_control_adaptive_weight_alpha",
            "0.75",
            "--context_lookback_columns",
            "192",
            "--batch_size",
            "4",
            "--gradient_accumulation_steps",
            "2",
            "--max_flat_len",
            "2048",
            "--gradient_checkpointing",
            "true",
            "--lora_r",
            "128",
            "--unfreeze_n_layers",
            "2",
            "--learning_rate",
            "0.0001",
        ]
    )
    if checkpoint:
        argv.extend(["--resume_from", checkpoint])
    return argv


def profile_training(
    *,
    mode: str,
    steps: int,
    output_dir: Path,
    work_dir: Path,
    checkpoint: str | None,
    processed_dir: str,
) -> WorkflowReport:
    import main

    argv = _training_argv(
        mode=mode,
        steps=steps,
        run_dir=work_dir,
        checkpoint=checkpoint,
        processed_dir=processed_dir,
    )
    phases: list[PhaseTiming] = []
    notes: list[str] = []

    started = time.perf_counter()
    prior_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        with _cprofile_capture() as prof:
            def run_training():
                main.main(argv)

            cuda_path = output_dir / "training_torch_cuda.txt"
            cuda_ran = _maybe_torch_cuda_profile(run_training, cuda_path)
            if cuda_ran is None:
                run_training()
    finally:
        os.chdir(prior_cwd)
    total = time.perf_counter() - started
    _write_text(output_dir / "training_cprofile.txt", _format_cprofile(prof))
    if cuda_ran is None:
        notes.append("CUDA torch profiler skipped (no GPU).")

    return WorkflowReport(
        workflow=f"training_{mode}",
        total_seconds=total,
        phases=phases,
        notes=notes + [f"argv tail: ... {' '.join(argv[-6:])}"],
    )


def _load_eval_model(checkpoint: str, device: torch.device, dtype: torch.dtype):
    from scripts.eval.turn_taking_probe import load_model

    loader_args = SimpleNamespace(
        model_path="models/Qwen3-4B-Instruct-2507",
        checkpoint_dir=checkpoint,
        num_agents=8,
    )
    t0 = time.perf_counter()
    model, tokenizer = load_model(loader_args, device, dtype)
    return model, tokenizer, time.perf_counter() - t0


def profile_eval(
    *,
    examples: int,
    output_dir: Path,
    contract: str,
    checkpoint: str,
    processed_dir: str | None,
    temperature: float,
) -> WorkflowReport:
    from evaluation.paired_continuation import (
        checkpoint_identifier,
        load_contract,
        run_paired_continuation_evaluation,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    contract_data, resolved_processed, contract_id = load_contract(
        contract,
        split="development",
        processed_dir=processed_dir,
    )
    if processed_dir is None:
        processed_dir = resolved_processed

    model, tokenizer, model_load_sec = _load_eval_model(checkpoint, device, dtype)
    checkpoint_id = checkpoint_identifier(
        checkpoint, model_path="models/Qwen3-4B-Instruct-2507"
    )

    per_example: list[dict[str, Any]] = []
    original_generate = None

    def timed_generate(model, input_ids, max_new_tokens, **kwargs):
        nonlocal original_generate
        assert original_generate is not None
        col_times: list[float] = []
        module = model
        original_forward = module.forward

        def forward_timer(*args, **kw):
            t_col = time.perf_counter()
            out = original_forward(*args, **kw)
            col_times.append(time.perf_counter() - t_col)
            return out

        module.forward = forward_timer  # type: ignore[method-assign]
        try:
            t_gen = time.perf_counter()
            result = original_generate(
                model, input_ids, max_new_tokens, return_probs=True, **kwargs
            )
            gen_sec = time.perf_counter() - t_gen
        finally:
            module.forward = original_forward  # type: ignore[method-assign]
        per_example.append(
            {
                "continuation_columns": int(max_new_tokens),
                "generate_seconds": round(gen_sec, 4),
                "forward_passes": len(col_times),
                "forward_seconds_total": round(sum(col_times), 4),
                "forward_seconds_mean": round(
                    (sum(col_times) / len(col_times)) if col_times else 0.0, 4
                ),
                "forward_seconds_last_col": round(col_times[-1], 4) if col_times else 0.0,
            }
        )
        return result

    from model import generation as generation_mod

    original_generate = generation_mod.generate_matrix
    generation_mod.generate_matrix = timed_generate  # type: ignore[assignment]

    out_jsonl = output_dir / "eval_profile_output.jsonl"
    if out_jsonl.exists():
        out_jsonl.unlink()

    started = time.perf_counter()
    contract_load_sec = 0.0
    try:
        with _cprofile_capture() as prof:
            def eval_body():
                run_paired_continuation_evaluation(
                    model=model,
                    tokenizer=tokenizer,
                    contract=contract_data,
                    processed_dir=processed_dir,
                    output_path=out_jsonl,
                    contract_id=contract_id,
                    checkpoint_id=checkpoint_id,
                    split="development",
                    fractions=[0.5],
                    generation_seeds=[0],
                    temperature=temperature,
                    activity_threshold=0.5,
                    placeholder_token_id=0,
                    device=device,
                    max_examples=examples,
                )

            cuda_path = output_dir / "eval_torch_cuda.txt"
            if _maybe_torch_cuda_profile(eval_body, cuda_path) is None:
                eval_body()
    finally:
        generation_mod.generate_matrix = original_generate  # type: ignore[assignment]

    total = time.perf_counter() - started
    _write_text(output_dir / "eval_cprofile.txt", _format_cprofile(prof))
    _write_text(
        output_dir / "eval_per_example.jsonl",
        "\n".join(json.dumps(row) for row in per_example) + ("\n" if per_example else ""),
    )

    phases = [
        PhaseTiming("contract_load", contract_load_sec),
        PhaseTiming("model_load", model_load_sec),
        PhaseTiming(
            "generate_loop",
            total - contract_load_sec - model_load_sec,
            {"examples": len(per_example)},
        ),
    ]
    if per_example:
        mean_gen = sum(r["generate_seconds"] for r in per_example) / len(per_example)
        mean_cols = sum(r["continuation_columns"] for r in per_example) / len(per_example)
        phases.append(
            PhaseTiming(
                "per_example_mean",
                mean_gen,
                {
                    "mean_continuation_columns": round(mean_cols, 1),
                    "mean_forward_passes": round(
                        sum(r["forward_passes"] for r in per_example) / len(per_example), 1
                    ),
                },
            )
        )

    return WorkflowReport(
        workflow="eval_held_out",
        total_seconds=total,
        phases=phases,
        notes=[f"examples={examples}", f"output={out_jsonl}"],
    )


def profile_generate_matrix_micro(
    *,
    output_dir: Path,
    checkpoint: str,
    prefix_columns: int,
    continuation_columns: int,
    num_agents: int,
) -> WorkflowReport:
    """One synthetic conversation-shaped generate_matrix call with per-column timings."""
    from model.generation import generate_matrix
    from scripts.eval.turn_taking_probe import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, _tokenizer, model_load_sec = _load_eval_model(checkpoint, device, dtype)
    model.eval()

    agents = num_agents
    prefix = prefix_columns
    cont = continuation_columns
    placeholder = 0
    input_ids = torch.full((1, agents, prefix), placeholder, dtype=torch.long, device=device)
    input_activity = torch.ones((1, agents, prefix), dtype=torch.bool, device=device)

    col_times: list[float] = []
    original_forward = model.forward

    def forward_timer(*args, **kw):
        t0 = time.perf_counter()
        out = original_forward(*args, **kw)
        col_times.append(time.perf_counter() - t0)
        return out

    model.forward = forward_timer  # type: ignore[method-assign]
    started = time.perf_counter()
    with torch.no_grad():
        generate_matrix(
            model,
            input_ids,
            cont,
            input_activity_mask=input_activity,
            temperature=1.0,
            activity_threshold=0.5,
            placeholder_token_id=placeholder,
            return_probs=True,
        )
    total = time.perf_counter() - started
    model.forward = original_forward  # type: ignore[method-assign]

    payload = {
        "prefix_columns": prefix,
        "continuation_columns": cont,
        "num_agents": agents,
        "model_load_seconds": round(model_load_sec, 4),
        "generate_total_seconds": round(total, 4),
        "per_column_seconds": [round(t, 4) for t in col_times],
        "per_column_seconds_first": col_times[0] if col_times else None,
        "per_column_seconds_last": col_times[-1] if col_times else None,
        "per_column_seconds_mean": round(sum(col_times) / len(col_times), 4) if col_times else None,
    }
    _write_text(
        output_dir / "generate_matrix_per_column.json",
        json.dumps(payload, indent=2) + "\n",
    )

    return WorkflowReport(
        workflow="generate_matrix_micro",
        total_seconds=total + model_load_sec,
        phases=[
            PhaseTiming("model_load", model_load_sec),
            PhaseTiming("generate_matrix", total, payload),
        ],
    )


def render_summary(reports: list[WorkflowReport], output_dir: Path) -> str:
    lines = [
        "# Workflow profiling summary",
        "",
        f"Output directory: `{output_dir}`",
        "",
    ]
    for report in reports:
        lines.append(f"## {report.workflow}")
        lines.append("")
        lines.append(f"- **Total:** {report.total_seconds:.2f}s")
        for phase in report.phases:
            extra = ""
            if phase.details:
                extra = " — " + ", ".join(f"{k}={v}" for k, v in phase.details.items())
            lines.append(f"- **{phase.name}:** {phase.seconds:.2f}s{extra}")
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(
        [
            "## How to read the artifacts",
            "",
            "- `*_cprofile.txt`: Python cumulative/internal time — look for `generate_matrix`,",
            "  `MatrixQwenForCausalLM.forward`, dataloader, tokenizer decode.",
            "- `*_torch_cuda.txt`: GPU kernel time (CUDA workflows only).",
            "- `eval_per_example.jsonl`: per-example generate time vs continuation length.",
            "- `generate_matrix_per_column.json`: shows whether later columns cost more",
            "  (growing flattened context / quadratic attention).",
            "",
            "## Likely speedup levers (hypotheses to validate here)",
            "",
            "1. **Eval:** cache prefix KV states instead of re-forwarding full history each column.",
            "2. **Eval:** batch independent examples (currently serial over the contract).",
            "3. **Eval:** shorten continuation cap or use adaptive early-stop when degenerate.",
            "4. **Training:** first-step validation dominates short runs; keep `val_fraction=0` for smoke.",
            "5. **Both:** SDPA/FlashAttention if safe for this model (training uses eager by default).",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("tiny", "real"),
        default="real",
        help="tiny=CPU toy model; real=production-shaped Qwen+checkpoint on CUDA if available.",
    )
    parser.add_argument("--training-steps", type=int, default=10)
    parser.add_argument("--eval-examples", type=int, default=10)
    parser.add_argument(
        "--workflows",
        nargs="+",
        default=["training", "eval", "generate_matrix"],
        choices=["training", "eval", "generate_matrix"],
    )
    parser.add_argument("--output-dir", default="logs/profiling/latest")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--prefix-columns",
        type=int,
        default=100,
        help="Synthetic prefix length for generate_matrix microbench.",
    )
    parser.add_argument(
        "--continuation-columns",
        type=int,
        default=96,
        help="Synthetic continuation length for generate_matrix microbench.",
    )
    parser.add_argument("--num-agents", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = _ensure_dir(Path(args.output_dir))
    work_dir = _ensure_dir(output_dir / "work")
    reports: list[WorkflowReport] = []

    if "training" in args.workflows:
        reports.append(
            profile_training(
                mode=args.mode,
                steps=args.training_steps,
                output_dir=output_dir,
                work_dir=work_dir,
                checkpoint=args.checkpoint if args.mode == "real" else None,
                processed_dir=args.processed_dir or DEFAULT_PROCESSED,
            )
        )

    if args.mode == "real":
        if "eval" in args.workflows:
            reports.append(
                profile_eval(
                    examples=args.eval_examples,
                    output_dir=output_dir,
                    contract=args.contract,
                    checkpoint=args.checkpoint,
                    processed_dir=args.processed_dir,
                    temperature=args.temperature,
                )
            )
        if "generate_matrix" in args.workflows:
            reports.append(
                profile_generate_matrix_micro(
                    output_dir=output_dir,
                    checkpoint=args.checkpoint,
                    prefix_columns=args.prefix_columns,
                    continuation_columns=args.continuation_columns,
                    num_agents=args.num_agents,
                )
            )
    else:
        if "eval" in args.workflows or "generate_matrix" in args.workflows:
            reports.append(
                WorkflowReport(
                    workflow="skipped_cuda_workflows",
                    total_seconds=0.0,
                    phases=[],
                    notes=["eval/generate_matrix skipped in --mode tiny (CPU toy training only)."],
                )
            )

    summary = render_summary(reports, output_dir)
    _write_text(output_dir / "SUMMARY.md", summary)
    print(summary)


if __name__ == "__main__":
    main()
