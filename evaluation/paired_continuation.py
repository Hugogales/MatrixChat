"""Reproducible paired held-out continuation evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .continuation_metrics import continuation_metrics
from .contract import (
    CONTRACT_VERSION,
    _content_hash,
    _entry,
    _implementation_hashes,
    _row_identity_hash,
)
from .paired_stats import paired_effect_summary, summarize_by_source_fraction


DEFAULT_SPLIT = "development"
ALLOWED_SPLITS = ("development", "final_test")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{hashlib.sha256(_canonical_bytes(value)).hexdigest()[:20]}"


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_contract(
    contract_path: str | os.PathLike[str],
    *,
    split: str = DEFAULT_SPLIT,
    allow_final_test: bool = False,
    processed_dir: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], Path, str]:
    """Load a contract and verify its current processed manifest before use."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS}, got {split!r}")
    if split == "final_test" and not allow_final_test:
        raise PermissionError(
            "final_test is sealed; pass --allow-final-test explicitly to evaluate it"
        )
    contract_path = Path(contract_path).expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(contract.get("version", -1)) != CONTRACT_VERSION:
        raise ValueError(
            f"unsupported contract version {contract.get('version')!r}; "
            f"expected {CONTRACT_VERSION}"
        )
    if split not in contract or not isinstance(contract[split], list):
        raise ValueError(f"contract has no {split!r} entry list")
    root = Path(processed_dir or contract["processed_dir"]).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    expected = str(contract.get("manifest_sha256") or "")
    if actual != expected:
        raise ValueError(
            f"processed manifest hash mismatch: contract={expected}, current={actual}"
        )
    return contract, root, stable_id("contract", contract)


def load_referenced_rows(
    contract: Mapping[str, Any],
    processed_dir: str | os.PathLike[str],
    *,
    split: str = DEFAULT_SPLIT,
    trust_frozen_contract: bool = False,
) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    """Load only referenced HF shard rows and verify every content hash."""
    from datasets import load_from_disk

    root = Path(processed_dir).resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    expected_manifest = str(contract.get("manifest_sha256") or "")
    actual_manifest = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest != expected_manifest:
        raise ValueError(
            "processed manifest hash mismatch: "
            f"contract={expected_manifest}, current={actual_manifest}"
        )
    manifest = json.loads(manifest_bytes)
    sources = manifest.get("sources") or {}
    provenance = contract.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("contract has no source_provenance")
    datasets: dict[str, Any] = {}
    loaded = []
    for entry in contract[split]:
        source = str(entry["source"])
        if source not in sources:
            raise ValueError(f"contract source {source!r} is absent from current manifest")
        if source not in datasets:
            shard_path = (root / sources[source].get("path", source)).resolve()
            if root not in shard_path.parents:
                raise ValueError(f"source path escapes processed directory: {shard_path}")
            datasets[source] = load_from_disk(str(shard_path))
            source_provenance = provenance.get(source)
            if not isinstance(source_provenance, Mapping):
                raise ValueError(f"contract has no provenance for source {source!r}")
            expected_fingerprint = str(
                source_provenance.get("hf_dataset_fingerprint") or ""
            )
            actual_fingerprint = str(
                getattr(datasets[source], "_fingerprint", "") or ""
            )
            if actual_fingerprint != expected_fingerprint:
                raise ValueError(
                    f"HF dataset fingerprint mismatch for {source}: "
                    f"contract={expected_fingerprint}, current={actual_fingerprint}"
                )
            expected_implementations = dict(
                source_provenance.get("implementation_sha256") or {}
            )
            if not trust_frozen_contract:
                actual_implementations = _implementation_hashes(source)
                if actual_implementations != expected_implementations:
                    raise ValueError(
                        f"implementation hash mismatch for source {source!r}"
                    )
        index = int(entry["row_index"])
        try:
            row = dict(datasets[source][index])
        except (IndexError, KeyError) as exc:
            raise ValueError(f"missing referenced row {source}[{index}]") from exc
        actual = _content_hash(row)
        expected = str(entry["content_sha256"])
        if actual != expected:
            raise ValueError(
                f"row content hash mismatch for {source}[{index}]: "
                f"contract={expected}, current={actual}"
            )
        actual_identity = _row_identity_hash(row)
        expected_identity = str(entry.get("row_identity_sha256") or "")
        if actual_identity != expected_identity:
            raise ValueError(
                f"row identity hash mismatch for {source}[{index}]: "
                f"contract={expected_identity}, current={actual_identity}"
            )
        row_split = str(row.get("dataset_split", row.get("split")) or "train").lower()
        expected_split = "validation" if split == "development" else "test"
        if row_split != expected_split or row_split != str(entry.get("split")):
            raise ValueError(
                f"split mismatch for {source}[{index}]: "
                f"contract={entry.get('split')}, current={row_split}"
            )
        rebuilt = _entry(
            source,
            index,
            row,
            row_split,
            int(contract["min_prefix_new_columns"]),
            int(contract["min_continuation_columns"]),
        )
        integrity_fields = (
            "source_id",
            "conversation_id",
            "group_id",
            "chunk_index",
            "num_agents",
            "length",
            "timed",
            "num_overlap_columns",
            "num_pause_columns",
            "context_lookback_columns",
            "has_private_content",
            "private_cell_count",
            "is_chain_rich",
            "num_speaker_changes",
            "eligible_cut_points",
        )
        for field in integrity_fields:
            if entry.get(field) != rebuilt.get(field):
                raise ValueError(
                    f"row metadata mismatch for {source}[{index}] field {field!r}: "
                    f"contract={entry.get(field)!r}, current={rebuilt.get(field)!r}"
                )
        loaded.append((entry, row))
    return loaded


def checkpoint_identifier(
    checkpoint_dir: str | os.PathLike[str] | None,
    *,
    model_path: str | os.PathLike[str] | None = None,
) -> str:
    """Content-identify a checkpoint once without depending on directory mtimes."""
    if checkpoint_dir is None:
        return stable_id("checkpoint", {"base_model": str(model_path or "")})
    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"checkpoint directory is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"checkpoint_{digest.hexdigest()[:20]}"


def _matrix(row: Mapping[str, Any], name: str, agents: int, length: int, default: Any):
    values = row.get(name)
    if values is None:
        return [[default for _ in range(length)] for _ in range(agents)]
    matrix = [list(agent)[:length] for agent in values]
    if len(matrix) != agents or any(len(agent) != length for agent in matrix):
        raise ValueError(f"{name} does not match contract [A][T]=[{agents}][{length}]")
    return matrix


def _decode_part(
    tokenizer: Any,
    token_ids: Sequence[Sequence[int]],
    activity: Sequence[Sequence[Any]],
) -> tuple[list[str], list[list[str]]]:
    agents = len(token_ids)
    columns = len(token_ids[0]) if agents else 0
    decoded = [
        tokenizer.decode(
            [int(token) for token, active in zip(token_ids[agent], activity[agent]) if active],
            skip_special_tokens=True,
        )
        for agent in range(agents)
    ]
    cells = [
        [
            tokenizer.decode([int(token_ids[agent][column])], skip_special_tokens=True)
            if bool(activity[agent][column])
            else ""
            for agent in range(agents)
        ]
        for column in range(columns)
    ]
    return decoded, cells


def _part(
    tokenizer: Any,
    token_ids: Sequence[Sequence[int]],
    activity: Sequence[Sequence[Any]],
    private_mask: Sequence[Sequence[Any]],
    agent_visibility: Sequence[Sequence[Any]],
    *,
    speak_probabilities: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    ids = [[int(value) for value in row] for row in token_ids]
    active = [[bool(value) for value in row] for row in activity]
    private = [[bool(value) for value in row] for row in private_mask]
    visibility = [[bool(value) for value in row] for row in agent_visibility]
    decoded, cells = _decode_part(tokenizer, ids, active)
    result = {
        "token_ids": ids,
        "activity": active,
        "is_private_mask": private,
        "agent_visibility": visibility,
        "decoded_by_agent": decoded,
        "cells": cells,
    }
    if speak_probabilities is not None:
        result["speak_probabilities"] = [
            [float(value) for value in row] for row in speak_probabilities
        ]
    return result


def _torch_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _select_entries(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    max_examples: int | None,
) -> Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if max_examples is None:
        return rows
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    return rows[:max_examples]


def planned_trials(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    contract_id: str,
    checkpoint_id: str,
    generation_id: str,
    fractions: Sequence[float],
    generation_seeds: Sequence[int],
    max_examples: int | None = None,
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], int]]:
    """Return deterministic eligible trial specifications."""
    wanted = {float(value) for value in fractions}
    unknown = wanted - {0.4, 0.5}
    if unknown:
        raise ValueError(f"unsupported cut fractions: {sorted(unknown)}")
    specs = []
    for entry, row in _select_entries(rows, max_examples):
        for cut in entry.get("eligible_cut_points") or []:
            if float(cut["fraction"]) not in wanted:
                continue
            for seed in generation_seeds:
                identity = {
                    "contract_id": contract_id,
                    "checkpoint_id": checkpoint_id,
                    "generation_id": generation_id,
                    "source": entry["source"],
                    "row_index": int(entry["row_index"]),
                    "fraction": float(cut["fraction"]),
                    "column": int(cut["column"]),
                    "seed": int(seed),
                }
                specs.append((stable_id("trial", identity), entry, row, cut, int(seed)))
    return specs


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(f"incomplete JSONL line {line_number} in {source}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL line {line_number} in {source}") from exc
            if not isinstance(record, dict) or not record.get("trial_id"):
                raise ValueError(f"JSONL line {line_number} has no trial_id")
            records.append(record)
    if len({record["trial_id"] for record in records}) != len(records):
        raise ValueError(f"duplicate trial IDs in {source}")
    return records


def _lock_owner_alive(lock: Path) -> bool:
    """Best-effort liveness check for a lock's recorded PID.

    Only meaningful when the checking process runs on the same host that
    wrote the lock (true for this project's single-process CLI usage). If the
    lock content is missing/unparseable, treat the owner as unknown/alive so
    we never silently steal a lock we can't positively prove is dead.
    """
    try:
        pid = int(lock.read_text().strip().splitlines()[0])
    except (OSError, ValueError, IndexError):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _output_lock(output_path: Path):
    lock = output_path.with_suffix(output_path.suffix + ".lock")
    if lock.exists() and not _lock_owner_alive(lock):
        # A prior run was killed uncleanly (OOM, node fault, SIGTERM without
        # cleanup) and left an orphaned lock -- a real, previously-hit failure
        # mode on this cluster, not a hypothetical. Reclaim it rather than
        # blocking every future resume attempt forever.
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"evaluation output is locked: {lock}") from exc
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        lock.unlink(missing_ok=True)


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    encoded = _canonical_bytes(record) + b"\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short JSONL append: wrote {written}/{len(encoded)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _numeric_leaves(value: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    leaves = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            leaves.update(_numeric_leaves(child, path))
        elif isinstance(child, (bool, int, float)):
            leaves[path] = float(child)
    return leaves


def build_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, float], int] = {}
    for record in records:
        key = (str(record["source"]), float(record["cut"]["fraction"]))
        groups[key] = groups.get(key, 0) + 1
    summary: dict[str, Any] = {
        "record_count": len(records),
        "trial_ids": [record["trial_id"] for record in records],
        "groups": [
            {"source": source, "fraction": fraction, "count": count}
            for (source, fraction), count in sorted(groups.items())
        ],
    }
    metric_pairs: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        human = _numeric_leaves(record["human"]["metrics"])
        generated = _numeric_leaves(record["model"]["metrics"])
        for key in sorted(human.keys() & generated.keys()):
            metric_pairs.setdefault(key, []).append(
                {
                    "source": record["source"],
                    "fraction": record["cut"]["fraction"],
                    "human": human[key],
                    "model": generated[key],
                }
            )
    summary["paired_metrics"] = {
        key: {
            "overall": paired_effect_summary(
                [pair["human"] for pair in pairs],
                [pair["model"] for pair in pairs],
            ),
            "by_source_fraction": summarize_by_source_fraction(
                pairs, before_key="human", after_key="model"
            ),
        }
        for key, pairs in sorted(metric_pairs.items())
    }
    judged = [record for record in records if record.get("judge")]
    if judged:
        summary["judge_aggregate"] = paired_effect_summary(
            [record["judge"]["human"]["aggregate"] for record in judged],
            [record["judge"]["model"]["aggregate"] for record in judged],
        )
    return summary


def run_paired_continuation_evaluation(
    *,
    model: Any,
    tokenizer: Any,
    contract: Mapping[str, Any],
    processed_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    contract_id: str,
    checkpoint_id: str,
    split: str = DEFAULT_SPLIT,
    allow_final_test: bool = False,
    fractions: Sequence[float] = (0.4, 0.5),
    generation_seeds: Sequence[int] = (0,),
    max_examples: int | None = None,
    temperature: float = 0.0,
    activity_threshold: float = 0.5,
    placeholder_token_id: int = 0,
    device: str | torch.device | None = None,
    generate_fn: Callable[..., Any] | None = None,
    protocol: str = "matrix",
    round_robin_start: str = "next_after_cut_reference",
    judge_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    judge_config: Mapping[str, Any] | None = None,
    summary_path: str | os.PathLike[str] | None = None,
    metadata_path: str | os.PathLike[str] | None = None,
    eval_batch_size: int = 4,
    trust_frozen_contract: bool = False,
) -> dict[str, Any]:
    """Generate and append lossless paired records, safely resuming by trial ID."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS}, got {split!r}")
    if split == "final_test" and not allow_final_test:
        raise PermissionError(
            "final_test is sealed; pass allow_final_test=True explicitly"
        )
    if not generation_seeds:
        raise ValueError("at least one generation seed is required")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 <= activity_threshold <= 1.0:
        raise ValueError("activity_threshold must be between zero and one")
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be at least 1")
    if generate_fn is None:
        from model.generation import generate_matrix

        generate_fn = generate_matrix

    normalized_judge_config = None
    if judge_fn is not None:
        normalized_judge_config = dict(judge_config or {})
        missing_judge_identity = {
            key
            for key in ("model", "base_url")
            if not str(normalized_judge_config.get(key) or "").strip()
        }
        if missing_judge_identity:
            raise ValueError(
                "judge_config must identify judge model and base_url; missing "
                + ", ".join(sorted(missing_judge_identity))
            )
        if "rubric_prompt_sha256" not in normalized_judge_config:
            from .judge_adapter import HELD_OUT_JUDGE_SYSTEM

            normalized_judge_config["rubric_prompt_sha256"] = hashlib.sha256(
                HELD_OUT_JUDGE_SYSTEM.encode("utf-8")
            ).hexdigest()
    protocol_name = str(protocol or "matrix")
    allowed_protocols = {"matrix", "round_robin", "xml_round_robin", "pass_the_baton"}
    if protocol_name not in allowed_protocols:
        raise ValueError(f"protocol must be one of {sorted(allowed_protocols)}, got {protocol_name!r}")
    generation_config = {
        "temperature": float(temperature),
        "activity_threshold": float(activity_threshold),
        "placeholder_token_id": int(placeholder_token_id),
        "fractions": [float(value) for value in fractions],
        "generation_seeds": [int(value) for value in generation_seeds],
    }
    # Keep the historical matrix identity unchanged; only non-matrix
    # protocols hash extra fields so resume cannot mix JSONL files.
    if protocol_name != "matrix":
        generation_config["protocol"] = protocol_name
        generation_config["round_robin_start"] = str(round_robin_start)
    if protocol_name in {"xml_round_robin", "pass_the_baton"}:
        generation_config["turn_format"] = "xml_agent_tags"
        generation_config["hidden_routing"] = protocol_name == "pass_the_baton"
    evaluation_config = {
        **generation_config,
        "split": split,
        "max_examples": max_examples,
        "judge": normalized_judge_config,
    }
    generation_id = stable_id("generation", generation_config)
    evaluation_identity = {
        "contract_id": contract_id,
        "checkpoint_id": checkpoint_id,
        "split": split,
        "generation_config": generation_config,
        "judge_config": normalized_judge_config,
        "max_examples": max_examples,
    }
    evaluation_id = stable_id("evaluation", evaluation_identity)
    config_identity = _canonical_bytes(evaluation_config).decode("utf-8")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = Path(summary_path) if summary_path else output.with_suffix(".summary.json")
    metadata = Path(metadata_path) if metadata_path else output.with_suffix(".metadata.json")
    trust_frozen_contract = bool(trust_frozen_contract)
    if not trust_frozen_contract and metadata.exists() and output.exists():
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        if str(previous.get("contract_id") or "") == contract_id:
            trust_frozen_contract = True
    rows = load_referenced_rows(
        contract,
        processed_dir,
        split=split,
        trust_frozen_contract=trust_frozen_contract,
    )
    specs = planned_trials(
        rows,
        contract_id=contract_id,
        checkpoint_id=checkpoint_id,
        generation_id=generation_id,
        fractions=fractions,
        generation_seeds=generation_seeds,
        max_examples=max_examples,
    )
    run_metadata = {
        "evaluation_id": evaluation_id,
        "evaluation_identity": evaluation_identity,
        "config_identity": config_identity,
        "contract_id": contract_id,
        "checkpoint_id": checkpoint_id,
        "generation_id": generation_id,
        "split": split,
        "judge_enabled": judge_fn is not None,
        "judge_config": normalized_judge_config,
        "config": evaluation_config,
        "planned_trial_count": len(specs),
    }
    target_device = torch.device(device) if device is not None else _torch_device(model)
    with _output_lock(output):
        existing = read_jsonl(output)
        if existing and not metadata.exists():
            raise ValueError("existing output has no metadata; refusing unsafe resume")
        if metadata.exists():
            previous = json.loads(metadata.read_text(encoding="utf-8"))
            identity_keys = (
                "evaluation_id",
                "evaluation_identity",
                "config_identity",
                "contract_id",
                "checkpoint_id",
                "generation_id",
                "split",
                "judge_enabled",
                "judge_config",
                "config",
                "planned_trial_count",
            )
            if any(previous.get(key) != run_metadata[key] for key in identity_keys):
                raise ValueError("existing metadata does not match this evaluation run")
        else:
            atomic_write_json(
                metadata,
                {
                    **run_metadata,
                    "output_path": str(output.resolve()),
                    "summary_path": str(summary.resolve()),
                    "completed_trial_count": 0,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        expected_trials = {
            trial_id: {
                "source": entry["source"],
                "row_index": int(entry["row_index"]),
                "source_id": int(entry["source_id"]),
                "conversation_id": entry.get("conversation_id", ""),
                "group_id": entry.get("group_id", ""),
                "content_sha256": entry["content_sha256"],
                "row_identity_sha256": entry["row_identity_sha256"],
                "cut": {
                    "fraction": float(cut["fraction"]),
                    "column": int(cut["column"]),
                    "continuation_columns": int(entry["length"]) - int(cut["column"]),
                },
                "generation_seed": seed,
            }
            for trial_id, entry, _row, cut, seed in specs
        }
        for record in existing:
            record_identity = {
                "evaluation_id": evaluation_id,
                "contract_id": contract_id,
                "checkpoint_id": checkpoint_id,
                "generation_id": generation_id,
                "split": split,
            }
            if any(record.get(key) != value for key, value in record_identity.items()):
                raise ValueError(
                    f"existing trial {record['trial_id']} belongs to a different evaluation run"
                )
            if (
                record.get("config_identity") != config_identity
                or record.get("config") != evaluation_config
            ):
                raise ValueError(
                    f"existing trial {record['trial_id']} has incompatible config identity"
                )
            expected_trial = expected_trials.get(str(record["trial_id"]))
            if expected_trial is None or any(
                record.get(key) != value for key, value in expected_trial.items()
            ):
                raise ValueError(
                    f"existing trial {record['trial_id']} does not match the trial plan"
                )
            if (record.get("judge") is not None) != (judge_fn is not None):
                raise ValueError(
                    f"existing trial {record['trial_id']} has incompatible judge mode"
                )
        completed = {record["trial_id"] for record in existing}
        from evaluation.eval_batching import EvalTrialSpec, bucket_eval_specs, collate_prefix_batch

        pending_specs = [
            EvalTrialSpec(trial_id=trial_id, entry=entry, row=row, cut=cut, seed=seed)
            for trial_id, entry, row, cut, seed in specs
            if trial_id not in completed
        ]
        batches = bucket_eval_specs(pending_specs, batch_size=eval_batch_size)

        def _build_record(
            trial_id: str,
            entry: Mapping[str, Any],
            row: Mapping[str, Any],
            cut: Mapping[str, Any],
            seed: int,
            model_ids: list[list[int]],
            model_activity: list[list[bool]],
            model_probabilities: list[list[float]],
        ) -> dict[str, Any]:
            agents = int(entry["num_agents"])
            length = int(entry["length"])
            column = int(cut["column"])
            continuation_columns = length - column
            ids = _matrix(row, "input_ids", agents, length, placeholder_token_id)
            activity = _matrix(row, "input_activity_mask", agents, length, False)
            private = _matrix(row, "is_private_mask", agents, length, False)
            visibility = row.get("agent_visibility")
            if visibility is None:
                visibility = [[True] * agents for _ in range(agents)]
            visibility = [[bool(value) for value in values] for values in visibility]
            if len(visibility) != agents or any(len(values) != agents for values in visibility):
                raise ValueError("agent_visibility must be [A][A]")

            prefix_ids = [values[:column] for values in ids]
            prefix_activity = [values[:column] for values in activity]
            prefix_private = [values[:column] for values in private]
            human_ids = [values[column:] for values in ids]
            human_activity = [values[column:] for values in activity]
            human_private = [values[column:] for values in private]
            model_private = [[False] * continuation_columns for _ in range(agents)]

            prefix_part = _part(
                tokenizer, prefix_ids, prefix_activity, prefix_private, visibility
            )
            human_part = _part(
                tokenizer, human_ids, human_activity, human_private, visibility
            )
            model_part = _part(
                tokenizer,
                model_ids,
                model_activity,
                model_private,
                visibility,
                speak_probabilities=model_probabilities,
            )
            human_part["metrics"] = continuation_metrics(
                human_ids, human_activity, prefix_activity=prefix_activity
            )
            model_part["metrics"] = continuation_metrics(
                model_ids, model_activity, prefix_activity=prefix_activity
            )
            record: dict[str, Any] = {
                "trial_id": trial_id,
                "evaluation_id": evaluation_id,
                "config_identity": config_identity,
                "contract_id": contract_id,
                "checkpoint_id": checkpoint_id,
                "generation_id": generation_id,
                "split": split,
                "source": entry["source"],
                "row_index": int(entry["row_index"]),
                "source_id": int(entry["source_id"]),
                "conversation_id": entry.get("conversation_id", ""),
                "group_id": entry.get("group_id", ""),
                "content_sha256": entry["content_sha256"],
                "row_identity_sha256": entry["row_identity_sha256"],
                "cut": {
                    "fraction": float(cut["fraction"]),
                    "column": column,
                    "continuation_columns": continuation_columns,
                },
                "generation_seed": seed,
                "prefix": prefix_part,
                "human": human_part,
                "model": model_part,
                "config": evaluation_config,
            }
            if judge_fn is not None:
                record["judge"] = dict(judge_fn(record))
            return record

        for batch in batches:
            trial_specs = batch.specs

            if protocol_name == "matrix" and len(trial_specs) > 1:
                collated = collate_prefix_batch(
                    trial_specs,
                    placeholder_token_id=placeholder_token_id,
                    device=target_device,
                )
                seed = int(trial_specs[0].seed)
                devices = [target_device.index or 0] if target_device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    if target_device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    generated_ids, generated_activity, probabilities = generate_fn(
                        model,
                        collated["input_ids"],
                        collated["continuation_columns"],
                        input_activity_mask=collated["input_activity"],
                        temperature=temperature,
                        activity_threshold=activity_threshold,
                        placeholder_token_id=placeholder_token_id,
                        input_private_mask=collated["input_private"],
                        agent_visibility=collated["agent_visibility"],
                        return_probs=True,
                    )
                model_ids_batch = generated_ids.detach().cpu().tolist()
                model_activity_batch = generated_activity.detach().cpu().tolist()
                model_probabilities_batch = probabilities.detach().cpu().tolist()
                for row_idx, spec in enumerate(trial_specs):
                    continuation_columns = collated["continuation_columns"]
                    model_ids = model_ids_batch[row_idx]
                    model_activity = model_activity_batch[row_idx]
                    model_probabilities = model_probabilities_batch[row_idx]
                    if any(len(values) != continuation_columns for values in model_ids):
                        raise ValueError("generate_matrix returned the wrong continuation length")
                    record = _build_record(
                        spec.trial_id,
                        spec.entry,
                        spec.row,
                        spec.cut,
                        spec.seed,
                        model_ids,
                        model_activity,
                        model_probabilities,
                    )
                    _append_record(output, record)
                    existing.append(record)
                    completed.add(spec.trial_id)
                continue

            for spec in trial_specs:
                trial_id = spec.trial_id
                entry = spec.entry
                row = spec.row
                cut = spec.cut
                seed = spec.seed
                agents = int(entry["num_agents"])
                length = int(entry["length"])
                column = int(cut["column"])
                ids = _matrix(row, "input_ids", agents, length, placeholder_token_id)
                activity = _matrix(row, "input_activity_mask", agents, length, False)
                private = _matrix(row, "is_private_mask", agents, length, False)
                visibility = row.get("agent_visibility")
                if visibility is None:
                    visibility = [[True] * agents for _ in range(agents)]
                visibility = [[bool(value) for value in values] for values in visibility]
                if len(visibility) != agents or any(len(values) != agents for values in visibility):
                    raise ValueError("agent_visibility must be [A][A]")

                prefix_ids = [values[:column] for values in ids]
                prefix_activity = [values[:column] for values in activity]
                prefix_private = [values[:column] for values in private]
                continuation_columns = length - column

                input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=target_device)
                input_activity = torch.tensor(
                    [prefix_activity], dtype=torch.bool, device=target_device
                )
                input_private = torch.tensor(
                    [prefix_private], dtype=torch.bool, device=target_device
                )
                agent_visibility = torch.tensor(
                    [visibility], dtype=torch.bool, device=target_device
                )
                devices = [target_device.index or 0] if target_device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    if target_device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    generated_ids, generated_activity, probabilities = generate_fn(
                        model,
                        input_ids,
                        continuation_columns,
                        input_activity_mask=input_activity,
                        temperature=temperature,
                        activity_threshold=activity_threshold,
                        placeholder_token_id=placeholder_token_id,
                        input_private_mask=input_private,
                        agent_visibility=agent_visibility,
                        return_probs=True,
                    )
                model_ids = generated_ids.detach().cpu().tolist()[0]
                model_activity = generated_activity.detach().cpu().tolist()[0]
                model_probabilities = probabilities.detach().cpu().tolist()[0]
                if any(len(values) != continuation_columns for values in model_ids):
                    raise ValueError("generate_matrix returned the wrong continuation length")
                record = _build_record(
                    trial_id,
                    entry,
                    row,
                    cut,
                    seed,
                    model_ids,
                    model_activity,
                    model_probabilities,
                )
                _append_record(output, record)
                existing.append(record)
                completed.add(trial_id)

        final_summary = {
            **build_summary(existing),
            "evaluation_id": evaluation_id,
            "contract_id": contract_id,
            "checkpoint_id": checkpoint_id,
            "generation_id": generation_id,
            "split": split,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(summary, final_summary)
        atomic_write_json(
            metadata,
            {
                **run_metadata,
                "output_path": str(output.resolve()),
                "summary_path": str(summary.resolve()),
                "completed_trial_count": len(existing),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return final_summary


__all__ = [
    "ALLOWED_SPLITS",
    "DEFAULT_SPLIT",
    "atomic_write_json",
    "build_summary",
    "checkpoint_identifier",
    "load_contract",
    "load_referenced_rows",
    "planned_trials",
    "read_jsonl",
    "run_paired_continuation_evaluation",
    "stable_id",
]
