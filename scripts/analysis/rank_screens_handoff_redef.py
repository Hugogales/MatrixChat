#!/usr/bin/env python3
"""CPU re-rank 60-example temperature-1 screens under the current handoff rules.

Reads lossless JSONL, recomputes metrics in memory, and writes cluster-weighted
clean / interruption / dirty rates. Does not rewrite the source JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.analysis import _cluster_id
from evaluation.continuation_metrics import recompute_record_metrics
from evaluation.paired_continuation import atomic_write_json, read_jsonl


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _flag(record: dict, side: str, flag: str) -> float:
    metrics = ((record.get(side) or {}).get("metrics") or {}).get("boundary_handoff") or {}
    return float(metrics.get(flag) or 0.0)


def _cluster_weighted(records: list[dict], flag: str) -> dict:
    human_clusters: dict[str, list[float]] = defaultdict(list)
    model_clusters: dict[str, list[float]] = defaultdict(list)
    human_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    model_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        cluster = _cluster_id(record)
        source = str(record.get("source") or "")
        human = _flag(record, "human", flag)
        model = _flag(record, "model", flag)
        human_clusters[cluster].append(human)
        model_clusters[cluster].append(model)
        human_source[source][cluster].append(human)
        model_source[source][cluster].append(model)

    def collapse(groups: dict[str, list[float]]) -> float:
        return _mean([_mean(values) for values in groups.values()])

    by_source = {}
    for source, groups in sorted(model_source.items()):
        by_source[source] = {
            "model": collapse(groups),
            "human": collapse(human_source[source]),
            "clusters": len(groups),
            "raw": sum(len(values) for values in groups.values()),
        }
    return {
        "model": collapse(model_clusters),
        "human": collapse(human_clusters),
        "clusters": len(model_clusters),
        "raw": len(records),
        "by_source": by_source,
    }


def _score_path(jsonl: Path) -> dict | None:
    meta_path = jsonl.parent / "eval_metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    config = meta.get("config") or {}
    if (
        meta.get("completed_trial_count") != 60
        or config.get("temperature") != 1.0
        or (config.get("split") or meta.get("split")) != "development"
    ):
        return None
    records = [recompute_record_metrics(record) for record in read_jsonl(jsonl)]
    if len(records) != 60:
        return None
    row = {
        "path": str(jsonl),
        "run": jsonl.parent.name,
        "parent": jsonl.parent.parent.name,
        "checkpoint_id": meta.get("checkpoint_id"),
        "strict": _cluster_weighted(records, "strict"),
        "interruption": _cluster_weighted(records, "interruption"),
        "dirty": _cluster_weighted(records, "dirty"),
    }
    ami = float(((row["strict"].get("by_source") or {}).get("ami") or {}).get("model") or 0.0)
    row["ami_dead"] = ami < 0.05
    row["rank_score"] = row["strict"]["model"]
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        default=str(_ROOT / "logs"),
        help="Root to search for eval_metadata.json files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSON ranking output path.",
    )
    args = parser.parse_args(argv)
    rows = []
    for meta in sorted(Path(args.logs_root).glob("**/eval_metadata.json")):
        jsonl = meta.parent / "paired_continuations.jsonl"
        if not jsonl.exists():
            continue
        try:
            row = _score_path(jsonl)
        except Exception as exc:
            rows.append({"path": str(jsonl), "error": str(exc)})
            continue
        if row is not None:
            rows.append(row)
            print(
                f"{row['run']:48s} strict={row['strict']['model']:.3f} "
                f"int={row['interruption']['model']:.3f} dirty={row['dirty']['model']:.3f}"
                f"{' AMI-DEAD' if row['ami_dead'] else ''}",
                flush=True,
            )
    scored = [row for row in rows if "strict" in row]
    scored.sort(key=lambda row: (-row["rank_score"], row["run"]))
    payload = {
        "n_screens": len(scored),
        "n_errors": sum("error" in row for row in rows),
        "ranking": scored,
        "errors": [row for row in rows if "error" in row],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"n_screens": len(scored), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
