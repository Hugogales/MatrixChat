#!/usr/bin/env python3
"""One-number composite score for a single DGX/manual race arm's probe output.

Wraps the exact same ``training.race.broad_sweep_score`` function the
month-long search's controller uses to rank candidates, so a manual DGX
checkup and an automated month-search candidate are judged by the IDENTICAL
formula -- one number to track during an hourly checkup instead of five
separate raw rates (clean_handoff_rate, overlap_rate, distinct_1, ...).

Works against either:
  - a small-suite probe: ``scripts/eval/evaluate_checkpoint_suite.py``'s
    ``--output`` file, or a training run's ``metrics.jsonl`` ``"probe"``
    entry (from ``main.py --probe_every``)
  - a full broad-sweep probe: ``scripts/eval/demo_handoff_variation.py``'s
    output (has raw per-trial ``"results"``, ``total_trials``, and
    ``by_context_kind`` -- gives a materially more trustworthy score, and is
    the only kind that supports ``--judge``)

Usage:
    python scripts/analysis/score_probe.py --metrics logs/race_20260729_CJ/metrics.jsonl
    python scripts/analysis/score_probe.py --probe logs/probe_results/CC_step500.json
    python scripts/analysis/score_probe.py --probe demo_variation_CG.json --judge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.race import broad_sweep_score  # noqa: E402

# scripts/eval/evaluate_checkpoint_suite.py's HANDOFF_PROMPTS length -- the
# fixed size of the base (non-scaling) small-suite handoff sweep, which is
# what clean_handoff_rate/nonoverlap_listener_rate/etc. are computed over.
# Only used as a fallback when a probe dict doesn't already carry its own
# "total_trials" (small-suite and in-training --probe_every probes don't;
# full demo_handoff_variation.py broad sweeps do).
SMALL_SUITE_TRIALS = 10


def load_probe(args: argparse.Namespace) -> dict:
    if args.probe:
        return json.loads(Path(args.probe).read_text(encoding="utf-8"))
    if args.metrics:
        latest = None
        with Path(args.metrics).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("probe"):
                    latest = record["probe"]
        if latest is None:
            raise SystemExit(
                f"no 'probe' entries found in {args.metrics} "
                "(was --probe_every enabled for this run?)"
            )
        return latest
    raise SystemExit("pass either --probe or --metrics")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--probe", help="Path to a probe/broad-sweep JSON file.")
    parser.add_argument(
        "--metrics", help="Path to a run's metrics.jsonl; uses the LAST 'probe' entry."
    )
    parser.add_argument(
        "--total-trials",
        type=int,
        default=None,
        help="Override the trial count used for Wilson confidence bounds "
        f"(default: {SMALL_SUITE_TRIALS} for small-suite probes, or the "
        "probe's own total_trials for broad sweeps).",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also call the remote Llama judge (requires the probe to have "
        "raw 'results' -- i.e. a full broad sweep, not a small suite).",
    )
    parser.add_argument("--judge-max-trials", type=int, default=16)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--output", help="Optional path to write the full score breakdown as JSON."
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    probe = load_probe(args)
    if "total_trials" not in probe or args.total_trials:
        probe = {**probe, "total_trials": args.total_trials or SMALL_SUITE_TRIALS}

    judge = None
    if args.judge:
        if "results" not in probe:
            raise SystemExit(
                "--judge requires a full broad-sweep probe with raw 'results' "
                "(small-suite/in-training probes don't save per-trial transcripts)"
            )
        from scripts.search.judge import (  # noqa: E402
            DEFAULT_BASE_URL,
            DEFAULT_MODEL,
            judge_broad_sweep,
        )

        judge = judge_broad_sweep(
            probe,
            base_url=args.judge_base_url or DEFAULT_BASE_URL,
            model=args.judge_model or DEFAULT_MODEL,
            max_trials=args.judge_max_trials,
        )

    result = broad_sweep_score(probe, judge=judge)

    headline = f"score            = {result['score']:.3f}"
    if result["disqualified"]:
        headline += "   DISQUALIFIED: " + ", ".join(result["gates"])
    print(headline)
    print(
        f"  behavior       = {result['behavior']:.4f}  "
        f"(global={result['global_behavior']:.4f}, "
        f"worst_context={result['worst_context_behavior']:.4f})"
    )
    print(
        f"  judge_quality  = {result['judge_quality']:.4f}"
        + ("  (neutral -- no judge run, pass --judge for a real value)" if judge is None else "")
    )
    print(f"  diversity      = {result['diversity']:.4f}")
    print(
        f"  calibration    = {result['calibration']:.4f}  "
        "(neutral -- this CLI does not wire in training val_components)"
    )
    print(f"  raw_score      = {result['raw_score']:.3f}")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"score": result, "judge": judge}, indent=2), encoding="utf-8"
        )
        print(f"wrote full breakdown to {args.output}")


if __name__ == "__main__":
    main()
