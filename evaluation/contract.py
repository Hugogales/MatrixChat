"""Build a reproducible contract for processed held-out MatrixChat examples.

The builder intentionally reads only the saved Hugging Face datasets described
by ``processed_dir/manifest.json``.  It records references and hashes, not row
payloads, so the resulting contract is safe to inspect without materializing
sealed test examples in training code.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


CONTRACT_VERSION = 2
CUT_FRACTIONS = (0.4, 0.5)
MATRIX_CONTENT_FIELDS = (
    "input_ids",
    "input_activity_mask",
    "labels",
    "activity_labels",
    "agent_ids",
    "is_private_mask",
    "agent_visibility",
)
_KNOWN_SPLITS = {"train", "validation", "test"}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_hash(row: Mapping[str, Any]) -> str:
    content = {field: row.get(field) for field in MATRIX_CONTENT_FIELDS}
    return _sha256_bytes(_canonical_bytes(content))


def _row_identity_hash(row: Mapping[str, Any]) -> str:
    """Hash the complete serialized row, including identity and metadata."""
    return _sha256_bytes(_canonical_bytes(dict(row)))


def _implementation_hashes(source: str) -> Dict[str, str]:
    project_root = Path(__file__).resolve().parents[1]
    candidates = {
        "converter": project_root / "data" / "convert.py",
        "schema": project_root / "data" / "schema.py",
        "source_adapter": project_root / "data" / "adapters" / f"{source}.py",
    }
    return {
        name: _sha256_bytes(path.read_bytes())
        for name, path in candidates.items()
        if path.is_file()
    }


def _normalise_split(value: Any) -> str:
    split = str(value or "train").strip().lower()
    if split not in _KNOWN_SPLITS:
        raise ValueError(f"Unknown dataset split {value!r}; expected train/validation/test")
    return split


def _eligible_cut_points(
    length: int,
    lookback_columns: int,
    min_prefix_columns: int,
    min_continuation_columns: int,
) -> list[Dict[str, Any]]:
    new_length = length - lookback_columns
    if new_length < 0:
        raise ValueError("context_lookback_columns cannot exceed row length")
    points = []
    for fraction in CUT_FRACTIONS:
        prefix_new_columns = int(new_length * fraction)
        column = lookback_columns + prefix_new_columns
        if (
            prefix_new_columns >= min_prefix_columns
            and length - column >= min_continuation_columns
        ):
            points.append(
                {
                    "fraction": fraction,
                    "column": column,
                    "prefix_new_columns": prefix_new_columns,
                    "continuation_columns": length - column,
                }
            )
    return points


def _has_private_content(row: Mapping[str, Any]) -> bool:
    mask = row.get("is_private_mask")
    return bool(mask and any(bool(cell) for agent in mask for cell in agent))


def _private_cell_count(row: Mapping[str, Any]) -> int:
    mask = row.get("is_private_mask")
    if not mask:
        return 0
    return sum(bool(cell) for agent in mask for cell in agent)


def _entry(
    source: str,
    row_index: int,
    row: Mapping[str, Any],
    split: str,
    min_prefix_columns: int,
    min_continuation_columns: int,
) -> Dict[str, Any]:
    length = int(row["length"])
    chunk_index = int(row.get("chunk_index") or 0)
    overlap_columns = int(row.get("num_overlap_columns") or 0)
    lookback_columns = int(row.get("context_lookback_columns") or 0)
    explicit_timed = row.get("is_timed", row.get("timed"))
    timed = (
        bool(explicit_timed)
        if explicit_timed is not None
        else source in {"ami", "werewolf"}
        or bool(chunk_index or overlap_columns or lookback_columns)
    )
    return {
        "source": source,
        "row_index": row_index,
        "source_id": int(row["source_id"]),
        "conversation_id": str(row.get("conversation_id") or ""),
        "group_id": str(row.get("group_id") or ""),
        "chunk_index": chunk_index,
        "split": split,
        "content_sha256": _content_hash(row),
        "row_identity_sha256": _row_identity_hash(row),
        "num_agents": int(row["num_agents"]),
        "length": length,
        "timed": timed,
        "num_overlap_columns": overlap_columns,
        "num_pause_columns": int(row.get("num_pause_columns") or 0),
        "context_lookback_columns": lookback_columns,
        "has_private_content": _has_private_content(row),
        "private_cell_count": _private_cell_count(row),
        "is_chain_rich": bool(row.get("is_chain_rich", False)),
        "num_speaker_changes": int(row.get("num_speaker_changes") or 0),
        "eligible_cut_points": _eligible_cut_points(
            length, lookback_columns, min_prefix_columns, min_continuation_columns
        ),
    }


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple:
    return (
        entry["source"],
        entry["conversation_id"],
        entry["group_id"],
        entry["chunk_index"],
        entry["row_index"],
    )


def _cap_per_source(
    entries: Iterable[Dict[str, Any]], max_per_source: Optional[int]
) -> list[Dict[str, Any]]:
    ordered = sorted(entries, key=_entry_sort_key)
    if max_per_source is None:
        return ordered
    counts: Dict[str, int] = defaultdict(int)
    selected = []
    for entry in ordered:
        source = entry["source"]
        if counts[source] < max_per_source:
            selected.append(entry)
            counts[source] += 1
    return selected


def _record_identifier_split(
    seen: Dict[str, Dict[str, str]],
    kind: str,
    identifier: str,
    split: str,
    location: str,
) -> None:
    if not identifier:
        return
    previous = seen[kind].get(identifier)
    if previous is not None and previous != split:
        raise ValueError(
            f"Split leakage: {kind} {identifier!r} occurs in both "
            f"{previous!r} and {split!r} ({location})"
        )
    seen[kind][identifier] = split


def build_eval_contract(
    processed_dir: str | os.PathLike[str],
    *,
    min_prefix_columns: int = 4,
    min_prefix_new_columns: Optional[int] = None,
    min_continuation_columns: int = 4,
    max_per_source: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a deterministic evaluation contract from processed Arrow shards.

    ``validation`` rows become development entries and exact ``test`` rows
    become sealed ``final_test`` entries.  Empty/absent split labels count as
    training rows for leakage checks and are never placed in either held-out
    collection.
    """
    if min_prefix_new_columns is not None:
        min_prefix_columns = min_prefix_new_columns
    if min_prefix_columns < 0 or min_continuation_columns < 0:
        raise ValueError("Minimum prefix/continuation columns must be non-negative")
    if max_per_source is not None and max_per_source <= 0:
        raise ValueError("max_per_source must be positive when provided")

    root = Path(processed_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Processed manifest has no sources")

    from datasets import load_from_disk

    development = []
    final_test = []
    seen: Dict[str, Dict[str, str]] = {
        "conversation_id": {},
        "group_id": {},
    }
    source_provenance: Dict[str, Dict[str, Any]] = {}
    for source in sorted(sources):
        info = sources[source]
        relative_path = info.get("path", source)
        shard_path = (root / relative_path).resolve()
        if root not in shard_path.parents:
            raise ValueError(f"Source {source!r} path escapes processed_dir: {relative_path!r}")
        dataset = load_from_disk(str(shard_path))
        source_provenance[source] = {
            "hf_dataset_fingerprint": str(getattr(dataset, "_fingerprint", "") or ""),
            "implementation_sha256": _implementation_hashes(source),
        }
        for row_index, row in enumerate(dataset):
            split = _normalise_split(row.get("dataset_split", row.get("split")))
            location = f"{source}[{row_index}]"
            _record_identifier_split(
                seen,
                "conversation_id",
                (
                    f"{source}:{row.get('conversation_id')}"
                    if row.get("conversation_id")
                    else ""
                ),
                split, location,
            )
            _record_identifier_split(
                seen,
                "group_id",
                f"{source}:{row.get('group_id')}" if row.get("group_id") else "",
                split, location,
            )
            if split not in {"validation", "test"}:
                continue
            entry = _entry(
                source,
                row_index,
                row,
                split,
                min_prefix_columns,
                min_continuation_columns,
            )
            (development if split == "validation" else final_test).append(entry)

    return {
        "version": CONTRACT_VERSION,
        "processed_dir": str(root),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "source_provenance": source_provenance,
        "cut_fractions": list(CUT_FRACTIONS),
        "min_prefix_new_columns": min_prefix_columns,
        "min_continuation_columns": min_continuation_columns,
        "max_per_source": max_per_source,
        "development": _cap_per_source(development, max_per_source),
        "final_test": _cap_per_source(final_test, max_per_source),
    }


def write_eval_contract(
    contract: Mapping[str, Any],
    output_path: str | os.PathLike[str],
) -> Path:
    """Atomically write ``contract`` as stable, human-readable JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(contract, handle, ensure_ascii=False, indent=2, sort_keys=True)
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
