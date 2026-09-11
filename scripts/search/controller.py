#!/usr/bin/env python3
"""Restart-safe controller for the month-long rotating HPO search.

The controller is CPU-only and is the sole writer of canonical
``search_state.json``. Five H100 workers communicate through atomic ticket and
result files managed by :mod:`scripts.search.search_state`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.search.search_state import (  # noqa: E402
    SearchPaths,
    acquire_directory_lock,
    append_jsonl,
    atomic_write_json,
    consume_results,
    enqueue_ticket,
    finish_ticket,
    read_json,
    expand_processed_dirs,
    reclaim_stale_claims,
    release_directory_lock,
    sbatch_search_exports,
    utc_now,
)
from scripts.search.judge import judge_broad_sweep  # noqa: E402
from scripts.search.strategist import propose as strategist_propose  # noqa: E402
from training.race import broad_sweep_score, promotion_blocked_by_probe  # noqa: E402

try:
    from scripts.search import bayes_opt  # noqa: E402
except ImportError:  # pragma: no cover - optuna is an optional dependency
    bayes_opt = None

_STUDY_CACHE: dict[str, "object"] = {}


def get_study(paths: "SearchPaths", state: dict):
    if bayes_opt is None:
        raise RuntimeError("optuna is not installed; cannot use optimizer='tpe'")
    key = str(paths.root)
    if key not in _STUDY_CACHE:
        _STUDY_CACHE[key] = bayes_opt.load_study(
            paths.root,
            seed=int(state["config"]["search_seed"]),
            n_startup_trials=int(state["config"].get("tpe_startup_trials", 5)),
        )
    return _STUDY_CACHE[key]


def _maybe_tell_bayes(
    paths: "SearchPaths", state: dict, candidate: dict, score: float | None
) -> None:
    """Report a candidate's terminal score back to the TPE study, once.

    Called exactly at the moment a candidate's lifecycle for its sampled
    configuration ends (pruned, finalist, or failed) so Optuna's tell() is
    never invoked twice for the same trial.
    """
    if str(state["config"].get("optimizer", "random")) != "tpe":
        return
    trial_number = candidate.get("optuna_trial_number")
    if trial_number is None or candidate.get("optuna_told"):
        return
    try:
        study = get_study(paths, state)
        bayes_opt.tell(study, trial_number, score)
        candidate["optuna_told"] = True
    except Exception as exc:  # pragma: no cover - defensive, never blocks the loop
        append_jsonl(
            paths.decisions,
            {
                "timestamp": utc_now(),
                "action": "bayes_tell_error",
                "candidate_id": candidate["candidate_id"],
                "error": str(exc),
            },
        )


DEFAULT_RUN_CONFIG = {
    "model_path": "models/Qwen3-4B-Instruct-2507",
    "processed_dirs": {
        "0": "~/matrixchat/processed_meld_ami_nolaugh_noname",
        "64": "~/matrixchat/processed_meld_ami_nolaugh_noname_lookback",
        "128": "~/matrixchat/processed_meld_ami_nolaugh_noname_lookback128",
        "192": "~/matrixchat/processed_meld_ami_nolaugh_noname_lookback192",
    },
    "rungs": [
        {"target_steps": 500, "priority": 100},
        {"target_steps": 1500, "priority": 200},
        {"target_steps": 5000, "priority": 300},
    ],
    "initial_candidates": 10,
    "target_active_candidates": 10,
    "max_candidates": 200,
    "train_slice_minutes": 45,
    "probe_every": 250,
    "eval_max_new_tokens": 48,
    "promotion_fraction": 1 / 3,
    "min_rung_population": 3,
    "search_seed": 20260728,
    "candidate_prefix": "month",
    "judge_enabled": True,
    "judge_base_url": "http://dh-dgxh100-2.hpc.msoe.edu:8000/v1",
    "judge_model": "meta/llama-4-scout-17b-16e-instruct",
    "judge_trials_by_rung": [12, 18, 24],
    "judge_quality_exponent": 0.5,
    "judge_calibration_path": "",
    "require_calibrated_judge": False,
    "strategist_enabled": True,
    "strategist_quota": 5,
    "optimizer": "random",
    "tpe_startup_trials": 5,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-runtime-hours", type=float, default=116.0)
    parser.add_argument("--stale-lease-hours", type=float, default=3.0)
    parser.add_argument("--controller-lease-seconds", type=float, default=600.0)
    parser.add_argument("--controller-sbatch", default="scripts/search/controller.sbatch")
    parser.add_argument("--no-self-chain", action="store_true")
    return parser.parse_args()


def load_run_config(path: str | None) -> dict:
    config = json.loads(json.dumps(DEFAULT_RUN_CONFIG))
    if path:
        override = read_json(Path(path), {})
        config.update(override)
    expand_processed_dirs(config)
    return config


def initial_state(config: dict) -> dict:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "generation": 0,
        "config": config,
        "next_candidate_number": 1,
        "processed_results": [],
        "candidates": {},
        "controller": {},
    }


def _sample_random_point(state: dict, number: int) -> dict:
    rng = random.Random(state["config"]["search_seed"] + number)
    base = rng.uniform(0.6, 2.0)
    maximum = rng.uniform(max(base + 0.8, 2.0), 6.0)
    config = state["config"]
    # Choice lists are configurable per-search (e.g. an 80GB-H100 search can
    # widen batch_size/max_flat_len/lora_r well beyond the defaults tuned for
    # smaller GPUs) but default to exactly the original fixed lists so any
    # search that doesn't set these overrides samples identically to before.
    batch_size_choices = config.get("batch_size_choices", [1, 2, 4])
    max_flat_len_choices = config.get("max_flat_len_choices", [1536, 2048, 3072])
    lora_r_choices = config.get("lora_r_choices", [16, 32, 64])
    gradient_accumulation_choices = config.get(
        "gradient_accumulation_choices", [1, 2, 4, 8]
    )
    return {
        "context_lookback_columns": rng.choice([64, 128, 192]),
        "activity_pos_weight": round(rng.uniform(0.70, 0.90), 3),
        "overlap_base_weight": round(base, 3),
        "overlap_max_weight": round(maximum, 3),
        "overlap_grace": rng.choice([0, 1]),
        "overlap_tau": rng.choice([1, 2, 3, 5]),
        "handoff_bonus_weight": round(rng.uniform(0.25, 1.75), 3),
        "lambda_activity": rng.choice(
            config.get("lambda_activity_choices", [0.5, 0.75, 1.0])
        ),
        "lambda_reward": rng.choice([0.05, 0.10, 0.20]),
        "gradient_accumulation_steps": rng.choice(gradient_accumulation_choices),
        "batch_size": rng.choice(batch_size_choices),
        "max_flat_len": rng.choice(max_flat_len_choices),
        "lora_r": rng.choice(lora_r_choices),
        "learning_rate": rng.choice([5e-5, 7.5e-5, 1e-4]),
        "dataset_weights": rng.choice(
            config.get(
                "dataset_weight_choices",
                [
                    "meld=0.3334,ami=0.3333,werewolf=0.3333",
                    "meld=0.15,ami=0.50,werewolf=0.35",
                ],
            )
        ),
        "sampling_strategy": rng.choice(["probabilistic", "balanced_cycle"]),
        "agent_attention_mode": rng.choice(
            config.get(
                "agent_attention_mode_choices", ["none", "qkv", "qkv_gated"]
            )
        ),
        "agent_attention_representation": rng.choice(
            config.get(
                "agent_attention_representation_choices", ["learned", "simplex"]
            )
        ),
        "agent_attention_dim": rng.choice(
            config.get("agent_attention_dim_choices", [16, 32])
        ),
        "agent_attention_init_std": rng.choice(
            config.get("agent_attention_init_std_choices", [0.001, 0.01])
        ),
        "agent_same_attention_bias": rng.choice(
            config.get("agent_same_attention_bias_choices", [False, True])
        ),
        "agent_same_attention_bias_init": rng.choice(
            config.get("agent_same_attention_bias_init_choices", [0.0, 0.5])
        ),
        "lambda_l2sp": rng.choice(
            config.get("lambda_l2sp_choices", [0.0, 0.02])
        ),
        "chain_rich_oversample_factor": rng.choice(
            config.get("chain_rich_oversample_factor_choices", [1.0, 3.0])
        ),
        "quality_oversample_factor": rng.choice(
            config.get("quality_oversample_factor_choices", [1.0, 2.0, 3.0])
        ),
        "warm_start_mode": rng.choice(
            config.get("warm_start_mode_choices", ["scratch", "from_leader"])
        ),
        "interface_lr_mult": rng.choice(
            config.get("interface_lr_mult_choices", [1.0, 3.0])
        ),
        "base_lr_mult": rng.choice(
            config.get("base_lr_mult_choices", [0.2, 1.0])
        ),
        "unfreeze_n_layers": rng.choice(
            config.get("unfreeze_n_layers_choices", [0, 2])
        ),
        "content_supervision_mode": rng.choice(
            config.get(
                "content_supervision_mode_choices", ["target_only", "all_speakers"]
            )
        ),
        "turn_reward_mode": rng.choice(
            config.get("turn_reward_mode_choices", ["linear", "balanced_ce"])
        ),
        "agent_dynamic_state_mode": rng.choice(
            config.get("agent_dynamic_state_mode_choices", ["none", "gru"])
        ),
        "floor_control_adaptive_weight_alpha": rng.choice(
            config.get(
                "floor_control_adaptive_weight_alpha_choices",
                [0.0, 0.25, 0.5, 0.75, 1.0],
            )
        ),
    }


def _dataset_families(config: dict) -> dict[str, list[str]]:
    choices = config.get(
        "dataset_weight_choices",
        [
            "meld=0.3334,ami=0.3333,werewolf=0.3333",
            "meld=0.15,ami=0.50,werewolf=0.35",
        ],
    )
    families = {
        "ami_heavy": [
            choice
            for choice in choices
            if "ami=0.5" in choice or "ami=0.50" in choice
        ],
        "structure": [
            choice
            for choice in choices
            if any(token in choice for token in ("bazinga", "molweni", "when2speak"))
            and "ami=0.5" not in choice
            and "ami=0.50" not in choice
        ],
        "legacy_equal": [choice for choice in choices if "ami=0.3333" in choice],
        "ami_light": [
            choice
            for choice in choices
            if "ami=0.2" in choice or "ami=0.20" in choice
        ],
    }
    return {name: values for name, values in families.items() if values}


def _sample_stratified_point(state: dict, number: int) -> dict:
    """Cover reward x dynstate x warm-start x data-family before TPE biases."""
    config = state["config"]
    rng = random.Random(config["search_seed"] + number * 9973)
    sampled = _sample_random_point(state, number)
    sampled["turn_reward_mode"] = rng.choice(
        config.get("turn_reward_mode_choices", ["linear", "balanced_ce"])
    )
    sampled["agent_dynamic_state_mode"] = rng.choice(
        config.get("agent_dynamic_state_mode_choices", ["none", "gru"])
    )
    sampled["warm_start_mode"] = rng.choice(
        config.get("warm_start_mode_choices", ["scratch", "from_leader"])
    )
    families = _dataset_families(config)
    family_name = rng.choice(list(families))
    sampled["dataset_weights"] = rng.choice(families[family_name])
    return sampled


def sample_candidate(
    state: dict,
    paths: "SearchPaths | None" = None,
    overrides: dict | None = None,
) -> dict:
    number = int(state["next_candidate_number"])
    state["next_candidate_number"] = number + 1
    prefix = str(state["config"].get("candidate_prefix", "month")).strip()
    candidate_id = f"{prefix}_cand_{number:05d}"

    optimizer = str(state["config"].get("optimizer", "random"))
    trial_number = None
    exploration_quota = int(state["config"].get("exploration_quota", 0))
    if number <= exploration_quota:
        sampled = _sample_stratified_point(state, number)
    elif optimizer == "tpe" and paths is not None and bayes_opt is not None:
        study = get_study(paths, state)
        trial_number, sampled = bayes_opt.ask(study, state["config"])
    else:
        sampled = _sample_random_point(state, number)

    config = {
        "candidate_id": candidate_id,
        "seed": 1000 + number,
        "model_path": state["config"]["model_path"],
        "processed_data_dir": state["config"]["processed_dirs"][
            str(sampled["context_lookback_columns"])
        ],
        "gradient_checkpointing": False,
        "num_epochs": 2,
        "max_grad_norm": 1.0,
        "probe_every": state["config"]["probe_every"],
        **sampled,
    }
    # Only lookback192 has a prepared all_speakers processed directory so
    # far (built 2026-08-07). content_supervision_mode is sampled
    # independently of context_lookback_columns, so force the pairing that
    # actually has data on disk rather than KeyError-ing on a directory
    # that doesn't exist.
    if config.get("content_supervision_mode") == "all_speakers":
        config["context_lookback_columns"] = 192
        config["processed_data_dir"] = state["config"]["processed_dirs"].get(
            "192_allspk", state["config"]["processed_dirs"]["192"]
        )
    # Memory-aware sampling. The attention implementation is quadratic in
    # flattened length; a true batch of several long examples can exceed even
    # 80GB. This is a deterministic function of batch_size, not an
    # independent search dimension, so it is applied after sampling either
    # way. Thresholds depend on the actual GPU memory the workers run on.
    gpu_memory_gb = float(state["config"].get("gpu_memory_gb", 80))
    if gpu_memory_gb <= 40:
        # V100 (32GB): empirically, even true batch_size=2 is unsafe for
        # this model/sequence-length combination. Fold any sampled batch
        # parallelism into gradient accumulation instead, preserving the
        # same effective batch size the search intended.
        if config["batch_size"] > 1:
            config["gradient_accumulation_steps"] *= config["batch_size"]
            config["batch_size"] = 1
        config["gradient_checkpointing"] = True
        config["max_flat_len"] = min(config["max_flat_len"], 1536)
    elif gpu_memory_gb >= 80:
        # H100 (80GB): empirically corrected on 2026-08-04 after the first
        # widened-search-space attempt OOM'd 22/22 failures, including the
        # SMALLEST new combination (batch_size=2, max_flat_len=3072, no
        # checkpointing). This model uses eager (quadratic-memory)
        # attention, so a long flattened sequence's attention-weight tensor
        # dominates memory regardless of the extra 48GB over a V100 --
        # checkpointing (which avoids retaining that tensor across the
        # whole forward pass) is required far more readily than initially
        # assumed. Err conservative: any max_flat_len beyond the old
        # generic 2048 ceiling, or any batch_size >= 4, forces
        # checkpointing; only small-batch/short-sequence combinations skip
        # it, matching the pre-2026-08-04 generic thresholds almost
        # exactly but still allowing batch_size up to 4 to go uncapped in
        # max_flat_len when unchecked (rather than lowering it), since
        # that combination is the one this GPU's extra memory can safely
        # spend on widening.
        if config["max_flat_len"] > 2048 or config["batch_size"] >= 4:
            config["gradient_checkpointing"] = True
        # A 2nd correction the same day: batch_size=8 + max_flat_len=4096 +
        # checkpointing STILL intermittently OOM'd partway through a train
        # slice (one candidate completed ~500 steps, an identically
        # -configured one OOM'd at step ~456 with only 296MiB free out of
        # 79GB) -- checkpointing avoids retaining activations across
        # layers, but doesn't lower this combination's steady-state working
        # set enough to leave real headroom against normal run-to-run
        # variance. Cap max_flat_len tighter once batch_size reaches 8.
        if config["batch_size"] >= 8:
            config["max_flat_len"] = min(config["max_flat_len"], 2048)
        # 5th empirical correction (2026-08-10 phase three): true batch8,
        # max_flat_len=2048, checkpointing=True also OOM'd twice at lora_r=32
        # (one gated-QKV/all-speakers candidate at step31 and one
        # no-QKV/all-speakers candidate at step253, each requesting another
        # ~15GiB with 65-73GiB already in use). This rules out rank as the
        # remaining discriminator: batch8 itself has no reliable margin on
        # the current eager-attention training stack. Preserve the intended
        # effective batch but always fold true batch8+ to batch4.
        if config["batch_size"] >= 8:
            config["gradient_accumulation_steps"] *= 2
            config["batch_size"] = 4
        # A 3rd correction on 2026-08-05: 3 more candidates OOM'd overnight,
        # ALL at batch_size=2/max_flat_len=2048/no-checkpointing -- the exact
        # combination the two corrections above still let skip
        # checkpointing, on the (never actually verified) assumption it
        # matched the pre-2026-08-04 generic thresholds' safe zone. It
        # doesn't: the generic path (below) forces checkpointing at
        # batch_size >= 2, not >= 4, and real evidence now shows even
        # batch_size=2/max_flat_len=2048 fills all 79GB by step 4
        # (torch.OutOfMemoryError, 78.19GiB in use). Every widened
        # combination tested without checkpointing has now failed, so stop
        # trying to find a safe skip-checkpointing zone for this GPU class
        # and always force it on instead.
        config["gradient_checkpointing"] = True
    elif config["batch_size"] >= 4:
        config["gradient_checkpointing"] = True
        config["max_flat_len"] = min(config["max_flat_len"], 1536)
    elif config["batch_size"] >= 2:
        config["gradient_checkpointing"] = True
        config["max_flat_len"] = min(config["max_flat_len"], 2048)
    if overrides:
        config.update(overrides)
    config.update(state["config"].get("fixed_candidate_overrides") or {})
    candidate = {
        "candidate_id": candidate_id,
        "created_at": utc_now(),
        "config": config,
        "status": "new",
        "rung": 0,
        "step": 0,
        "accumulated_train_seconds": 0.0,
        "scores": {},
        "history": [],
        "optuna_trial_number": trial_number,
    }
    return candidate


def outstanding_candidate_ids(paths: SearchPaths) -> set[str]:
    ids = set()
    for directory in (paths.pending, paths.claimed):
        for path in directory.glob("*.json"):
            ticket = read_json(path, {})
            if ticket.get("candidate_id"):
                ids.add(ticket["candidate_id"])
    return ids


def enqueue_train(paths: SearchPaths, state: dict, candidate: dict) -> None:
    rung = state["config"]["rungs"][candidate["rung"]]
    enqueue_ticket(
        paths,
        {
            "type": "train_slice",
            "candidate_id": candidate["candidate_id"],
            "priority": rung["priority"],
            "target_steps": rung["target_steps"],
            "slice_minutes": state["config"]["train_slice_minutes"],
        },
    )
    candidate["status"] = "queued_train"


def enqueue_evaluation(paths: SearchPaths, state: dict, candidate: dict) -> None:
    rung = state["config"]["rungs"][candidate["rung"]]
    ticket = {
        "type": "broad_evaluate",
        "candidate_id": candidate["candidate_id"],
        "priority": rung["priority"] + 500,
        "checkpoint_step": candidate["step"],
        "max_new_tokens": state["config"]["eval_max_new_tokens"],
    }
    if candidate.get("evaluation_checkpoint_dir"):
        ticket["checkpoint_dir"] = candidate["evaluation_checkpoint_dir"]
    enqueue_ticket(paths, ticket)
    candidate["status"] = "queued_eval"


def _lenient_handoff_link(link: dict) -> bool:
    """A lenient chain link is a clean yield or a brief interruption.

    Historical probe JSON stored brief overlap as ``dirty_handoff``. New
    results store that event as ``interruption_handoff`` and reserve
    ``dirty_handoff`` for 5+ overlapping columns.
    """
    if "interruption_handoff" in link:
        return bool(link.get("clean_handoff") or link.get("interruption_handoff"))
    return bool(link.get("clean_handoff") or link.get("dirty_handoff"))


def normalize_clean_chain_metrics(probe: dict) -> dict:
    """Recompute strict and lenient chains from per-link classifications."""
    results = probe.get("results") or []
    for row in results:
        clean_links = 0
        lenient_links = 0
        for link in row.get("chain", []):
            if link.get("clean_handoff"):
                clean_links += 1
            else:
                break
        for link in row.get("chain", []):
            if _lenient_handoff_link(link):
                lenient_links += 1
            else:
                break
        row["chain_length"] = clean_links
        row["chain_length_lenient"] = lenient_links
    if results:
        probe["chain_2plus_rate"] = sum(
            row["chain_length"] >= 2 for row in results
        ) / len(results)
        probe["chain_3plus_rate"] = sum(
            row["chain_length"] >= 3 for row in results
        ) / len(results)
        probe["chain_2plus_rate_lenient"] = sum(
            row["chain_length_lenient"] >= 2 for row in results
        ) / len(results)
        probe["chain_3plus_rate_lenient"] = sum(
            row["chain_length_lenient"] >= 3 for row in results
        ) / len(results)
        probe["interruption_handoff_rate"] = sum(
            bool(row.get("interruption_handoff")) for row in results
        ) / len(results)
        probe["dirty_handoff_rate"] = sum(
            bool(row.get("dirty_handoff")) for row in results
        ) / len(results)
        probe["handoff_rate_lenient"] = sum(
            _lenient_handoff_link(row) for row in results
        ) / len(results)
    for kind, bucket in (probe.get("by_context_kind") or {}).items():
        rows = [row for row in results if row.get("context_kind") == kind]
        if rows:
            bucket["chain_2plus_rate"] = sum(
                row["chain_length"] >= 2 for row in rows
            ) / len(rows)
            bucket["chain_3plus_rate"] = sum(
                row["chain_length"] >= 3 for row in rows
            ) / len(rows)
            bucket["chain_2plus_rate_lenient"] = sum(
                row["chain_length_lenient"] >= 2 for row in rows
            ) / len(rows)
            bucket["chain_3plus_rate_lenient"] = sum(
                row["chain_length_lenient"] >= 3 for row in rows
            ) / len(rows)
            bucket["interruption_handoff_rate"] = sum(
                bool(row.get("interruption_handoff")) for row in rows
            ) / len(rows)
            bucket["dirty_handoff_rate"] = sum(
                bool(row.get("dirty_handoff")) for row in rows
            ) / len(rows)
            bucket["handoff_rate_lenient"] = sum(
                _lenient_handoff_link(row) for row in rows
            ) / len(rows)
    return probe


def retryable_evaluation_failure(result: dict) -> bool:
    """Return whether an evaluation failure is infrastructure, not a bad trial."""
    if result.get("ticket_type") != "broad_evaluate":
        return False
    error = str(result.get("error", ""))
    markers = (
        "ModuleNotFoundError",
        "No module named",
        "FileNotFoundError",
        "ConnectionError",
        "timed out",
    )
    return any(marker in error for marker in markers)


def ingest_results(paths: SearchPaths, state: dict) -> None:
    processed = state["processed_results"]
    for _, result in consume_results(paths, processed):
        ticket_id = result["ticket_id"]
        candidate = state["candidates"].get(result.get("candidate_id"))
        if candidate is None:
            processed.append(ticket_id)
            finish_ticket(paths, ticket_id)
            continue
        candidate["history"].append(result)
        if result.get("status") != "ok":
            retries = int(candidate.get("evaluation_retries", 0))
            if retryable_evaluation_failure(result) and retries < 2:
                candidate["evaluation_retries"] = retries + 1
                candidate["last_evaluation_failure"] = result.get(
                    "error", "worker failure"
                )
                enqueue_evaluation(paths, state, candidate)
            else:
                candidate["status"] = "failed"
                candidate["failure"] = result.get("error", "worker failure")
                # Most training failures (e.g. OOM) are hyperparameter-caused,
                # so a score of 0.0 is informative signal. Infrastructure-only
                # evaluation failures are retried above instead of poisoning
                # the optimizer with a false zero.
                _maybe_tell_bayes(paths, state, candidate, 0.0)
        elif result["ticket_type"] == "train_slice":
            candidate["step"] = int(result.get("step", candidate["step"]))
            candidate["accumulated_train_seconds"] += float(
                result.get("duration_seconds", 0.0)
            )
            target = state["config"]["rungs"][candidate["rung"]]["target_steps"]
            if candidate["step"] >= target:
                enqueue_evaluation(paths, state, candidate)
            else:
                candidate["status"] = "ready_train"
        elif result["ticket_type"] == "broad_evaluate":
            probe = normalize_clean_chain_metrics(
                read_json(Path(result["output_path"]), {})
            )
            atomic_write_json(Path(result["output_path"]), probe)
            judge = None
            judge_error = None
            if state["config"].get("judge_enabled", True):
                try:
                    calibration_path = state["config"].get("judge_calibration_path")
                    if state["config"].get("require_calibrated_judge") and (
                        not calibration_path
                        or not read_json(Path(calibration_path), {}).get("launch_ready")
                    ):
                        raise RuntimeError("judge calibration has not passed launch gates")
                    trial_counts = state["config"]["judge_trials_by_rung"]
                    judge = judge_broad_sweep(
                        probe,
                        base_url=state["config"]["judge_base_url"],
                        model=state["config"]["judge_model"],
                        max_trials=int(
                            trial_counts[min(candidate["rung"], len(trial_counts) - 1)]
                        ),
                        seed=state["config"]["search_seed"] + candidate["rung"],
                    )
                    judge_dir = (
                        paths.candidate_dir(candidate["candidate_id"]) / "judge"
                    )
                    judge_dir.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(
                        judge_dir / f"{result.get('checkpoint_step', candidate['step'])}.json",
                        judge,
                    )
                except Exception as exc:
                    judge_error = str(exc)
                    judge = {
                        "schema_valid": False,
                        "position_consistent": True,
                        "needs_review": True,
                        "axis_lcb": {},
                        "error": judge_error,
                    }
            score = broad_sweep_score(
                probe,
                judge=judge,
                judge_quality_exponent=float(
                    state["config"].get("judge_quality_exponent", 0.5)
                ),
            )
            # Rung 0 is an early-learning snapshot. Repetition/low diversity
            # must persist into a second evaluation before becoming hard gates;
            # otherwise every 500-step candidate is pruned before ASHA can
            # test whether it improves with budget.
            if candidate["rung"] == 0 and set(score["gates"]).issubset(
                {"repetition", "low_diversity", "silence", "chain_hacking"}
            ):
                score["provisional_gates"] = list(score["gates"])
                score["disqualified"] = False
                score["score"] = score["raw_score"] * 0.25
            candidate["scores"][str(candidate["rung"])] = {
                **score,
                "checkpoint_step": result.get("checkpoint_step"),
                "probe_path": result["output_path"],
                "judge_error": judge_error,
                "scored_at": utc_now(),
            }
            if candidate.get("evaluation_only"):
                candidate["status"] = "imported_complete"
                if (
                    str(state["config"].get("optimizer", "random")) == "tpe"
                    and candidate.get("seed_optimizer", True)
                    and not candidate.get("optuna_told")
                ):
                    try:
                        study = get_study(paths, state)
                        trial_number = bayes_opt.add_completed_trial(
                            study,
                            candidate["config"],
                            float(score["score"]),
                            state["config"],
                        )
                        candidate["optuna_trial_number"] = trial_number
                        candidate["optuna_told"] = True
                    except Exception as exc:
                        append_jsonl(
                            paths.decisions,
                            {
                                "timestamp": utc_now(),
                                "action": "import_seed_error",
                                "candidate_id": candidate["candidate_id"],
                                "error": str(exc),
                            },
                        )
            else:
                candidate["status"] = "rung_complete"
        processed.append(ticket_id)
        finish_ticket(paths, ticket_id)


def snapshot_checkpoint_before_promotion(
    candidate_id: str, rung: int, step: int, checkpoints_root: Path | None = None
) -> str | None:
    """Copy checkpoints/<id>/last/ to an immutable per-rung snapshot.

    Returns the snapshot path as a string, or None if there was nothing to
    copy (e.g. a legacy/test candidate with no on-disk checkpoint). Copy
    (not move) so the worker can keep training from checkpoints/<id>/last/
    uninterrupted; failures are logged by the caller via decision_log.jsonl
    rather than raised, so a filesystem hiccup never blocks the controller
    loop.
    """
    root = checkpoints_root or (_ROOT / "checkpoints")
    source = root / candidate_id / "last"
    if not source.is_dir():
        return None
    destination = root / candidate_id / f"rung{rung}_step{step}"
    if destination.exists():
        return str(destination)  # already snapshotted (e.g. re-run after a crash)
    shutil.copytree(source, destination)
    return str(destination)


def promote_and_prune(paths: SearchPaths, state: dict) -> None:
    fraction = float(state["config"]["promotion_fraction"])
    minimum = int(state["config"]["min_rung_population"])
    last_rung = len(state["config"]["rungs"]) - 1
    for rung in range(last_rung + 1):
        rows = [
            candidate
            for candidate in state["candidates"].values()
            if candidate["rung"] == rung and candidate["status"] == "rung_complete"
        ]
        if rung == last_rung:
            for candidate in rows:
                score_row = candidate["scores"][str(rung)]
                if promotion_blocked_by_probe(score_row, rung=rung):
                    candidate["status"] = "pruned"
                    append_jsonl(
                        paths.decisions,
                        {
                            "timestamp": utc_now(),
                            "action": "prune",
                            "candidate_id": candidate["candidate_id"],
                            "rung": rung,
                            "score": score_row["score"],
                            "reason": "broad_sweep_red_flags",
                        },
                    )
                    _maybe_tell_bayes(paths, state, candidate, score_row["score"])
                    continue
                candidate["status"] = "finalist"
                _maybe_tell_bayes(
                    paths, state, candidate, candidate["scores"][str(rung)]["score"]
                )
            continue
        if len(rows) < minimum:
            continue
        rows.sort(
            key=lambda candidate: candidate["scores"][str(rung)]["score"],
            reverse=True,
        )
        promote_count = max(1, math.ceil(len(rows) * fraction))
        for index, candidate in enumerate(rows):
            score_row = candidate["scores"][str(rung)]
            if index < promote_count and not promotion_blocked_by_probe(
                score_row, rung=rung
            ):
                # A promoted candidate resumes training from the SAME
                # checkpoints/<id>/last/ directory the worker just evaluated,
                # which gets progressively overwritten as it trains toward
                # the next rung. Snapshot it now, before that happens --
                # otherwise a candidate's peak-scoring checkpoint (e.g. a
                # strong rung1 result later trained past its best point) is
                # silently unrecoverable once training continues.
                checkpoints_root = state["config"].get("checkpoints_root")
                try:
                    snapshot_path = snapshot_checkpoint_before_promotion(
                        candidate["candidate_id"],
                        rung,
                        candidate["step"],
                        checkpoints_root=Path(checkpoints_root)
                        if checkpoints_root
                        else None,
                    )
                except OSError as exc:
                    snapshot_path = None
                    append_jsonl(
                        paths.decisions,
                        {
                            "timestamp": utc_now(),
                            "action": "checkpoint_snapshot_error",
                            "candidate_id": candidate["candidate_id"],
                            "rung": rung,
                            "error": str(exc),
                        },
                    )
                candidate["rung"] += 1
                candidate["status"] = "ready_train"
                append_jsonl(
                    paths.decisions,
                    {
                        "timestamp": utc_now(),
                        "action": "promote",
                        "candidate_id": candidate["candidate_id"],
                        "to_rung": candidate["rung"],
                        "checkpoint_snapshot": snapshot_path,
                    },
                )
            else:
                candidate["status"] = "pruned"
                append_jsonl(
                    paths.decisions,
                    {
                        "timestamp": utc_now(),
                        "action": "prune",
                        "candidate_id": candidate["candidate_id"],
                        "rung": rung,
                        "score": candidate["scores"][str(rung)]["score"],
                    },
                )
                _maybe_tell_bayes(
                    paths, state, candidate, candidate["scores"][str(rung)]["score"]
                )


def maintain_population(paths: SearchPaths, state: dict) -> None:
    config = state["config"]
    if config.get("hold_population_until_imports_complete") and any(
        candidate.get("evaluation_only")
        and candidate["status"] not in ("imported_complete", "failed")
        for candidate in state["candidates"].values()
    ):
        return
    active = [
        candidate
        for candidate in state["candidates"].values()
        if candidate["status"]
        not in ("pruned", "failed", "finalist", "imported_complete")
    ]
    desired = int(config["target_active_candidates"])
    while len(active) < desired and len(state["candidates"]) < int(config["max_candidates"]):
        proposal = None
        quota = max(1, int(config.get("strategist_quota", 5)))
        should_ask_strategist = (
            config.get("strategist_enabled", True)
            and int(state["next_candidate_number"]) % quota == 0
            and any(candidate.get("scores") for candidate in state["candidates"].values())
        )
        if should_ask_strategist:
            try:
                proposal = strategist_propose(
                    state,
                    base_url=config["judge_base_url"],
                    model=config["judge_model"],
                )
            except Exception as exc:
                append_jsonl(
                    paths.decisions,
                    {
                        "timestamp": utc_now(),
                        "action": "strategist_error",
                        "error": str(exc),
                    },
                )
        candidate = sample_candidate(
            state, paths=paths, overrides=(proposal or {}).get("changes")
        )
        if proposal:
            candidate["strategist_proposal"] = proposal
        state["candidates"][candidate["candidate_id"]] = candidate
        candidate_dir = paths.candidate_dir(candidate["candidate_id"])
        candidate_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(candidate_dir / "config.json", candidate["config"])
        append_jsonl(
            paths.decisions,
            {
                "timestamp": utc_now(),
                "action": "sample",
                "candidate_id": candidate["candidate_id"],
                "config": candidate["config"],
                "strategist_proposal": proposal,
            },
        )
        active.append(candidate)

    outstanding = outstanding_candidate_ids(paths)
    for candidate in active:
        if candidate["candidate_id"] in outstanding:
            continue
        if candidate["status"] in ("new", "ready_train") or (
            candidate["status"] == "queued_train"
            and candidate["candidate_id"] not in outstanding
        ):
            enqueue_train(paths, state, candidate)


def write_reports(paths: SearchPaths, state: dict) -> None:
    rows = []
    for candidate in state["candidates"].values():
        scores = candidate.get("scores", {})
        latest = scores.get(str(candidate["rung"])) or (
            scores.get(str(candidate["rung"] - 1)) if candidate["rung"] else None
        )
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "status": candidate["status"],
                "rung": candidate["rung"],
                "step": candidate["step"],
                "score": (latest or {}).get("score"),
                "behavior": (latest or {}).get("behavior"),
                "diversity": (latest or {}).get("diversity"),
                "gates": ",".join((latest or {}).get("gates", [])),
            }
        )
        atomic_write_json(
            paths.candidate_dir(candidate["candidate_id"]) / "status.json",
            {
                "candidate_id": candidate["candidate_id"],
                "status": candidate["status"],
                "rung": candidate["rung"],
                "step": candidate["step"],
                "accumulated_train_seconds": candidate[
                    "accumulated_train_seconds"
                ],
                "scores": candidate.get("scores", {}),
                "failure": candidate.get("failure"),
                "updated_at": utc_now(),
            },
        )
    rows.sort(key=lambda row: row["score"] if row["score"] is not None else -1, reverse=True)
    atomic_write_json(paths.reports / "leaderboard.json", rows)
    csv_path = paths.reports / "leaderboard.csv"
    temporary = csv_path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["candidate_id"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    lines = [
        f"# Month Search ({utc_now()})",
        "",
        "| Candidate | Status | Rung | Step | Score | Behavior | Diversity | Gates |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows[:20]:
        lines.append(
            f"| {row['candidate_id']} | {row['status']} | {row['rung']} | "
            f"{row['step']} | {row['score'] if row['score'] is not None else '-'} | "
            f"{row['behavior'] if row['behavior'] is not None else '-'} | "
            f"{row['diversity'] if row['diversity'] is not None else '-'} | "
            f"{row['gates'] or '-'} |"
        )
    (paths.reports / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if rows and rows[0]["score"] is not None:
        leader = state["candidates"][rows[0]["candidate_id"]]
        atomic_write_json(
            paths.best / "behavior.json",
            {
                "updated_at": utc_now(),
                "candidate_id": leader["candidate_id"],
                "checkpoint_path": str(_ROOT / "checkpoints" / leader["candidate_id"]),
                "step": leader["step"],
                "score": rows[0]["score"],
                "config": leader["config"],
            },
        )


def heartbeat(paths: SearchPaths, state: dict) -> None:
    state["controller"] = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "generation": state["generation"],
        "heartbeat_at": utc_now(),
        "heartbeat_unix": time.time(),
    }
    atomic_write_json(paths.lease, state["controller"])


def submit_replacement(sbatch_path: str, search_dir: Path) -> str | None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    result = subprocess.run(
        [
            "sbatch",
            f"--dependency=afterany:{job_id}",
            f"--export={sbatch_search_exports(search_dir)}",
            sbatch_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip().split()[-1]


def controller_iteration(paths: SearchPaths, state: dict, stale_hours: float) -> None:
    reclaim_stale_claims(paths, max_age_seconds=stale_hours * 3600.0)
    ingest_results(paths, state)
    promote_and_prune(paths, state)
    maintain_population(paths, state)
    write_reports(paths, state)
    state["updated_at"] = utc_now()
    atomic_write_json(paths.state, state)
    heartbeat(paths, state)


def main():
    args = parse_args()
    paths = SearchPaths(Path(args.search_dir).resolve())
    paths.initialize()
    if paths.controller_lock.exists():
        lease = read_json(paths.lease, {})
        heartbeat_unix = float(lease.get("heartbeat_unix", 0.0) or 0.0)
        if heartbeat_unix and time.time() - heartbeat_unix < args.controller_lease_seconds:
            raise SystemExit("another live controller owns controller.lock")
        release_directory_lock(paths.controller_lock)
    if not acquire_directory_lock(paths.controller_lock):
        raise SystemExit("another controller owns controller.lock")
    try:
        config = load_run_config(args.config) if args.config else None
        state = read_json(paths.state)
        if state is None:
            if config is None:
                raise SystemExit("--config is required for a new search directory")
            state = initial_state(config)
            atomic_write_json(
                paths.root / "launch_config.json",
                {"path": str(Path(args.config).resolve())},
            )
        elif config is not None:
            existing_prefix = state.get("config", {}).get("candidate_prefix")
            new_prefix = config.get("candidate_prefix")
            if existing_prefix != new_prefix:
                raise SystemExit(
                    "search_state candidate_prefix="
                    f"{existing_prefix!r} disagrees with --config prefix="
                    f"{new_prefix!r}; use a fresh SEARCH_DIR"
                )
        expand_processed_dirs(state["config"])
        state["generation"] = int(state.get("generation", 0)) + 1
        started = time.time()
        while True:
            controller_iteration(paths, state, args.stale_lease_hours)
            if args.once:
                break
            if (time.time() - started) / 3600.0 >= args.max_runtime_hours:
                if not args.no_self_chain:
                    replacement = submit_replacement(args.controller_sbatch, paths.root)
                    append_jsonl(
                        paths.decisions,
                        {
                            "timestamp": utc_now(),
                            "action": "controller_self_chain",
                            "replacement_job_id": replacement,
                        },
                    )
                break
            time.sleep(args.poll_seconds)
    finally:
        release_directory_lock(paths.controller_lock)


if __name__ == "__main__":
    main()
