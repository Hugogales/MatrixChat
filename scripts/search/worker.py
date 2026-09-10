#!/usr/bin/env python3
"""Single-GPU rotating worker for the month-long search."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BROAD_EVAL_SCRIPT = _ROOT / "scripts" / "eval" / "demo_handoff_variation.py"

from scripts.search.search_state import (  # noqa: E402
    SearchPaths,
    claim_ticket,
    publish_result,
    read_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def bool_text(value) -> str:
    return "true" if bool(value) else "false"


def ensure_candidate_links(paths: SearchPaths, candidate_id: str) -> None:
    candidate_dir = paths.candidate_dir(candidate_id)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    links = {
        "metrics.jsonl": _ROOT / "logs" / candidate_id / "metrics.jsonl",
        "probe_history.jsonl": (
            _ROOT / "logs" / candidate_id / "checkpoint_probe_history.jsonl"
        ),
        "checkpoints": _ROOT / "checkpoints" / candidate_id,
    }
    for name, target in links.items():
        link = candidate_dir / name
        if link.exists() or link.is_symlink():
            continue
        try:
            link.symlink_to(target)
        except OSError:
            pass


def training_command(config: dict, ticket: dict) -> list[str]:
    candidate_id = config["candidate_id"]
    checkpoint = _ROOT / "checkpoints" / candidate_id / "last" / "training_state.pt"
    # Legacy (pre-phase-three) imported/queued configs may not have these
    # keys at all -- default to exact old behavior (no attention
    # conditioning, single learning rate, no L2-SP, no oversampling, the old
    # hardcoded unfreeze-2-layers) rather than raising.
    learning_rate = float(config["learning_rate"])
    interface_lr = learning_rate * float(config.get("interface_lr_mult", 1.0))
    base_lr = learning_rate * float(config.get("base_lr_mult", 1.0))
    unfreeze_n_layers = int(config.get("unfreeze_n_layers", 2))
    command = [
        sys.executable,
        str(_ROOT / "main.py"),
        "--use_tiny_model",
        "false",
        "--model_name",
        config["model_path"],
        "--processed_data_dir",
        str(Path(config["processed_data_dir"]).expanduser()),
        "--dataset_weights",
        config["dataset_weights"],
        "--sampling_strategy",
        config["sampling_strategy"],
        "--seed",
        str(config["seed"]),
        "--max_agents",
        "8",
        "--position_mode",
        "column",
        "--use_agent_embeddings",
        "true",
        "--agent_attention_mode",
        str(config.get("agent_attention_mode", "none")),
        "--agent_attention_representation",
        str(config.get("agent_attention_representation", "learned")),
        "--agent_attention_dim",
        str(config.get("agent_attention_dim", 32)),
        "--agent_attention_init_std",
        str(config.get("agent_attention_init_std", 1e-3)),
        "--agent_same_attention_bias",
        bool_text(config.get("agent_same_attention_bias", False)),
        "--agent_same_attention_bias_init",
        str(config.get("agent_same_attention_bias_init", 0.0)),
        "--agent_dynamic_state_mode",
        str(config.get("agent_dynamic_state_mode", "none")),
        "--agent_dynamic_state_dim",
        str(config.get("agent_dynamic_state_dim", 64)),
        "--agent_dynamic_state_init_std",
        str(config.get("agent_dynamic_state_init_std", 1e-3)),
        "--attn_mask_mode",
        "matrix_causal",
        "--placeholder_token_id",
        "0",
        "--lambda_content",
        "1.0",
        "--lambda_activity",
        str(config["lambda_activity"]),
        "--lambda_reward",
        str(config["lambda_reward"]),
        "--activity_pos_weight",
        str(config["activity_pos_weight"]),
        # 2026-08-24 reward rearchitecture (model/turn_reward.py's
        # floor_control_reward); floor_control_ref_* are FIXED reference
        # rates (not searched -- see FloorControlConfig's docstring) and are
        # simply ignored by main.py when turn_reward_mode="linear".
        "--turn_reward_mode",
        str(config.get("turn_reward_mode", "linear")),
        "--floor_control_ref_silence",
        "0.0784",
        "--floor_control_ref_same",
        "0.7805",
        "--floor_control_ref_handoff",
        "0.0675",
        "--floor_control_ref_overlap",
        "0.0735",
        "--floor_control_adaptive_weight_alpha",
        str(config.get("floor_control_adaptive_weight_alpha", 0.0)),
        "--floor_control_adaptive_prior_strength",
        str(config.get("floor_control_adaptive_prior_strength", 32.0)),
        "--speak_grace",
        "5",
        "--speak_tau",
        "200",
        "--silence_grace",
        "0",
        "--silence_tau",
        "5",
        "--overlap_base_weight",
        str(config["overlap_base_weight"]),
        "--overlap_max_weight",
        str(config["overlap_max_weight"]),
        "--overlap_grace",
        str(config["overlap_grace"]),
        "--overlap_tau",
        str(config["overlap_tau"]),
        "--handoff_bonus_weight",
        str(config["handoff_bonus_weight"]),
        "--lora_enable",
        "true",
        "--lora_r",
        str(config["lora_r"]),
        "--lora_alpha",
        str(2 * int(config["lora_r"])),
        "--lora_dropout",
        "0.05",
        "--lora_target_modules",
        "q_proj,k_proj,v_proj,o_proj",
        "--freeze_base_model",
        "true",
        "--unfreeze_first_n_layers",
        str(unfreeze_n_layers),
        "--unfreeze_last_n_layers",
        str(unfreeze_n_layers),
        "--learning_rate",
        str(learning_rate),
        "--interface_lr",
        str(interface_lr),
        "--base_lr",
        str(base_lr),
        "--lambda_l2sp",
        str(config.get("lambda_l2sp", 0.0)),
        "--weight_decay",
        "0",
        "--lr_scheduler",
        "constant",
        "--warmup_steps",
        "0",
        "--max_grad_norm",
        str(config["max_grad_norm"]),
        "--batch_size",
        str(config["batch_size"]),
        "--gradient_accumulation_steps",
        str(config["gradient_accumulation_steps"]),
        "--gradient_checkpointing",
        bool_text(config["gradient_checkpointing"]),
        "--empty_cache_every",
        "100",
        "--chain_rich_oversample_factor",
        str(config.get("chain_rich_oversample_factor", 1.0)),
        "--quality_oversample_factor",
        str(config.get("quality_oversample_factor", 1.0)),
        "--quality_min_score",
        str(config.get("quality_min_score", 0.55)),
        "--num_epochs",
        str(config["num_epochs"]),
        "--num_steps",
        str(ticket["target_steps"]),
        "--max_runtime_minutes",
        str(ticket["slice_minutes"]),
        "--num_workers",
        "8",
        "--log_every",
        "20",
        "--eval_every",
        "100",
        "--val_fraction",
        "0.05",
        "--max_flat_len",
        str(config["max_flat_len"]),
        "--checkpoint_every",
        "100",
        "--probe_every",
        str(config["probe_every"]),
        "--probe_num_agents",
        "3",
        "--probe_max_new_tokens",
        "16",
        "--sample_every",
        "0",
        "--torch_dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--save_checkpoint",
        "true",
        "--run_name",
        candidate_id,
    ]
    warm_start_mode = str(config.get("warm_start_mode", "scratch"))
    # Warm-start only on the first train slice. Resuming must use
    # --resume_from alone; main.py rejects both flags together.
    if warm_start_mode == "from_leader" and not checkpoint.is_file():
        leader_root = config.get(
            "warm_start_leader_checkpoint",
            "checkpoints/final40_h100_c106_s176106",
        )
        command.extend(
            [
                "--init_from",
                str(_ROOT / leader_root),
                "--init_checkpoint_prefer",
                str(config.get("init_checkpoint_prefer", "root")),
            ]
        )
    if checkpoint.is_file():
        previous = read_json(
            _ROOT / "hyperparameters" / f"{candidate_id}.json", {}
        )
        expected = {
            "lora_r": config["lora_r"],
            "processed_data_dir": config["processed_data_dir"],
            "activity_pos_weight": config["activity_pos_weight"],
            "batch_size": config["batch_size"],
            "gradient_accumulation_steps": config[
                "gradient_accumulation_steps"
            ],
        }
        mismatched = {
            key: (previous.get(key), value)
            for key, value in expected.items()
            if previous.get(key) != value
        }
        # Architecture-affecting keys added 2026-08-07: mismatches here would
        # load a checkpoint's saved module weights (or lack thereof) under a
        # different agent-attention wiring than they were trained with.
        # Legacy candidates first-saved before these keys existed genuinely
        # have no entry for them in `previous` at all -- but since main.py's
        # own argparse defaults are exactly these same fallback values, a
        # MISSING key means "matches the default", not "unknown, assume
        # mismatch". Use `previous.get(key, value)` (defaults to the current
        # expected value when absent) so only an ACTUAL differing saved
        # value counts as a mismatch, never a merely-absent legacy key --
        # a bug that made 2026-08-07 falsely reject a legacy candidate's
        # resume with e.g. `{'agent_attention_mode': (None, 'none')}` despite
        # both sides genuinely being "none".
        expected_arch = {
            "agent_attention_mode": config.get("agent_attention_mode", "none"),
            "agent_attention_representation": config.get(
                "agent_attention_representation", "learned"
            ),
            "agent_attention_dim": config.get("agent_attention_dim", 32),
            "agent_same_attention_bias": config.get(
                "agent_same_attention_bias", False
            ),
            # agent_dynamic_state_mode adds/removes a whole GRU module (see
            # model/dynamic_agent_state.py) -- architecture-affecting exactly
            # like the agent_attention_* keys above, so a resume across a
            # changed value must be refused the same way.
            "agent_dynamic_state_mode": config.get(
                "agent_dynamic_state_mode", "none"
            ),
        }
        for key, value in expected_arch.items():
            if previous.get(key, value) != value:
                mismatched[key] = (previous.get(key), value)
        # unfreeze_n_layers is a controller-only synthetic key (maps to two
        # real main.py args): saved hyperparameters JSONs use the real
        # argparse field names, not this key, so compare against those
        # directly instead of a key that never exists in `previous` (a bug
        # that made 2026-08-07 falsely reject every resume needing >1 train
        # slice with a guaranteed-mismatched `(None, 2)` -- see
        # TRAINING_RUN_LOG.md).
        expected_unfreeze = unfreeze_n_layers
        for real_key in ("unfreeze_first_n_layers", "unfreeze_last_n_layers"):
            if previous.get(real_key) != expected_unfreeze:
                mismatched[real_key] = (previous.get(real_key), expected_unfreeze)
        if mismatched:
            raise RuntimeError(
                f"refusing incompatible resume for {candidate_id}: {mismatched}"
            )
        command.extend(
            ["--resume_from", str(_ROOT / "checkpoints" / candidate_id)]
        )
    return command


def read_training_step(candidate_id: str) -> int:
    path = _ROOT / "checkpoints" / candidate_id / "last" / "training_state.pt"
    if not path.is_file():
        return 0
    import torch

    state = torch.load(path, map_location="cpu")
    return int(state.get("step", 0))


def log_tail(path: Path, limit: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return text[-limit:]


def subprocess_env() -> dict[str, str]:
    """Return the shared environment required by train and eval children."""
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=(
            str(_ROOT)
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        ),
    )
    return env


def run_train(paths: SearchPaths, ticket: dict, worker_id: str) -> dict:
    candidate_id = ticket["candidate_id"]
    config = read_json(paths.candidate_dir(candidate_id) / "config.json")
    if not config:
        raise FileNotFoundError(f"missing config for {candidate_id}")
    ensure_candidate_links(paths, candidate_id)
    logs = paths.candidate_dir(candidate_id) / "worker_logs"
    logs.mkdir(parents=True, exist_ok=True)
    output = logs / f"{ticket['ticket_id']}.log"
    started = time.time()
    with output.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            training_command(config, ticket),
            cwd=_ROOT,
            env=subprocess_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    step = read_training_step(candidate_id)
    result = {
        "status": "ok" if completed.returncode == 0 else "error",
        "returncode": completed.returncode,
        "step": step,
        "target_steps": ticket["target_steps"],
        "duration_seconds": time.time() - started,
        "log_path": str(output),
        "worker_id": worker_id,
    }
    if completed.returncode != 0:
        result["error"] = log_tail(output)
    return result


def run_evaluate(paths: SearchPaths, ticket: dict, worker_id: str) -> dict:
    candidate_id = ticket["candidate_id"]
    checkpoint_dir = Path(
        ticket.get("checkpoint_dir")
        or (_ROOT / "checkpoints" / candidate_id / "last")
    ).resolve()
    if not (checkpoint_dir / "model_state.pt").is_file():
        raise FileNotFoundError(f"missing last checkpoint for {candidate_id}")
    broad_dir = paths.candidate_dir(candidate_id) / "broad_sweeps"
    broad_dir.mkdir(parents=True, exist_ok=True)
    output = broad_dir / f"{ticket['checkpoint_step']}.json"
    logs = paths.candidate_dir(candidate_id) / "worker_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{ticket['ticket_id']}.log"
    config = read_json(paths.candidate_dir(candidate_id) / "config.json", {})
    command = [
        sys.executable,
        str(BROAD_EVAL_SCRIPT),
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--model_path",
        config.get("model_path", "models/Qwen3-4B-Instruct-2507"),
        "--output",
        str(output),
        "--max_new_tokens",
        str(ticket.get("max_new_tokens", 48)),
        "--device",
        "cuda",
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=_ROOT,
            env=subprocess_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    result = {
        "status": "ok" if completed.returncode == 0 and output.is_file() else "error",
        "returncode": completed.returncode,
        "checkpoint_step": ticket["checkpoint_step"],
        "duration_seconds": time.time() - started,
        "output_path": str(output),
        "log_path": str(log_path),
        "worker_id": worker_id,
    }
    if result["status"] != "ok":
        result["error"] = log_tail(log_path)
    return result


def execute_ticket(paths: SearchPaths, ticket: dict, worker_id: str) -> dict:
    if ticket["type"] == "train_slice":
        return run_train(paths, ticket, worker_id)
    if ticket["type"] == "broad_evaluate":
        return run_evaluate(paths, ticket, worker_id)
    raise ValueError(f"unknown ticket type: {ticket['type']}")


def main():
    args = parse_args()
    paths = SearchPaths(Path(args.search_dir).resolve())
    paths.initialize()
    worker_id = args.worker_id or (
        f"{os.environ.get('SLURM_JOB_ID', 'local')}-"
        f"{os.environ.get('SLURM_PROCID', os.getpid())}"
    )
    while True:
        if (paths.root / "shutdown_requested").exists():
            return
        claimed = claim_ticket(paths, worker_id)
        if claimed is None:
            if args.once:
                return
            time.sleep(args.poll_seconds)
            continue
        _, ticket = claimed
        try:
            result = execute_ticket(paths, ticket, worker_id)
        except Exception as exc:  # worker must report, not silently lose a lease
            result = {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "worker_id": worker_id,
            }
        publish_result(paths, ticket, result)
        if args.once:
            return


if __name__ == "__main__":
    main()
