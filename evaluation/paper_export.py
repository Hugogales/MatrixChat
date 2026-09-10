"""Per-example tables and transcripts for paper-facing held-out evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .analysis import DEFAULT_METRIC_SPECS, _finite, _record_metric_sides
from .paired_continuation import atomic_write_json, read_jsonl


EXPORT_METRICS = tuple(DEFAULT_METRIC_SPECS)


def _decoded(side: Mapping[str, Any]) -> list[str]:
    return [str(text) for text in (side.get("decoded_by_agent") or [])]


def _pct_diff(human: float, model: float) -> float | None:
    if human == 0.0:
        return None
    return 100.0 * (model - human) / abs(human)


def record_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per example with human/model/difference for every registered metric."""
    rows = []
    for record in records:
        human_metrics, model_metrics = _record_metric_sides(record)
        row: dict[str, Any] = {
            "trial_id": record.get("trial_id"),
            "source": record.get("source"),
            "conversation_id": record.get("conversation_id"),
            "group_id": record.get("group_id"),
            "source_id": record.get("source_id"),
            "split": record.get("split"),
            "generation_seed": record.get("generation_seed"),
            "cut_fraction": _finite((record.get("cut") or {}).get("fraction")),
            "checkpoint_id": record.get("checkpoint_id"),
        }
        for name in EXPORT_METRICS:
            human = human_metrics.get(name)
            model = model_metrics.get(name)
            row[f"{name}.human"] = human
            row[f"{name}.model"] = model
            if human is None or model is None:
                row[f"{name}.diff"] = None
                row[f"{name}.pct_diff"] = None
            else:
                row[f"{name}.diff"] = model - human
                row[f"{name}.pct_diff"] = _pct_diff(human, model)
        rows.append(row)
    return rows


def transcript_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compact decoded text for both sides, suitable for paper inspection."""
    exported = []
    for record in records:
        human = record.get("human") or {}
        model = record.get("model") or {}
        prefix = record.get("prefix") or {}
        exported.append(
            {
                "trial_id": record.get("trial_id"),
                "source": record.get("source"),
                "conversation_id": record.get("conversation_id"),
                "group_id": record.get("group_id"),
                "cut_fraction": _finite((record.get("cut") or {}).get("fraction")),
                "prefix_decoded_by_agent": _decoded(prefix),
                "human_decoded_by_agent": _decoded(human),
                "model_decoded_by_agent": _decoded(model),
                "human_num_columns": (human.get("metrics") or {}).get("num_columns"),
                "model_num_columns": (model.get("metrics") or {}).get("num_columns"),
                "judge": record.get("judge"),
            }
        )
    return exported


def export_paper_tables(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    loaded = list(records) if records is not None else read_jsonl(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = record_rows(loaded)
    csv_path = destination / "per_example_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else ["trial_id"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    transcripts_path = destination / "per_example_transcripts.jsonl"
    with transcripts_path.open("w", encoding="utf-8") as handle:
        for item in transcript_records(loaded):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    atomic_write_json(
        destination / "export_manifest.json",
        {
            "record_count": len(loaded),
            "metrics": list(EXPORT_METRICS),
            "per_example_metrics": str(csv_path.resolve()),
            "per_example_transcripts": str(transcripts_path.resolve()),
        },
    )
    return {
        "per_example_metrics": str(csv_path),
        "per_example_transcripts": str(transcripts_path),
    }
