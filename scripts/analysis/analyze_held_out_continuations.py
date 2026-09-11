#!/usr/bin/env python3
"""Analyze paired held-out continuation JSONL for publication reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.analysis import analyze_jsonl, filter_records
from evaluation.continuation_metrics import recompute_record_metrics
from evaluation.paper_export import export_paper_tables
from evaluation.paired_continuation import read_jsonl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", help="Paired held-out continuation JSONL.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for atomic summary.json, stats.json, and plots.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Paired bootstrap resamples per overall/stratified effect.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic bootstrap seed.")
    parser.add_argument(
        "--plot-metrics",
        nargs="+",
        default=None,
        metavar="METRIC",
        help=(
            "Preregistered metric paths to plot (default: meaningful primary "
            "outcomes including silence, overlap, run length, and balance)."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib import and plot generation.",
    )
    parser.add_argument(
        "--no-export-tables",
        action="store_true",
        help="Skip per-example CSV/transcript export.",
    )
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help=(
            "Recompute human/model continuation metrics from stored "
            "token_ids and activity before analysis (does not rewrite the JSONL)."
        ),
    )
    parser.add_argument(
        "--exclude-sources",
        nargs="+",
        default=None,
        metavar="SOURCE",
        help="Drop these sources (e.g. meld) from cluster-weighted tests and means.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary, stats = analyze_jsonl(
        args.input,
        args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        plot_metrics=args.plot_metrics,
        make_plots=not args.no_plots,
        recompute_metrics=args.recompute_metrics,
        exclude_sources=args.exclude_sources,
    )
    result = {
        "summary": str((Path(args.output_dir) / "summary.json").resolve()),
        "stats": str((Path(args.output_dir) / "stats.json").resolve()),
        "record_count": summary["record_count"],
        "test_count": stats["multiple_comparison_correction"]["test_count"],
        "plot_count": len(summary["plots"]),
    }
    if not args.no_export_tables:
        export_records = read_jsonl(args.input)
        if args.recompute_metrics:
            export_records = [recompute_record_metrics(record) for record in export_records]
        export_records = filter_records(export_records, exclude_sources=args.exclude_sources)
        result["exports"] = export_paper_tables(
            args.input, args.output_dir, records=export_records
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
