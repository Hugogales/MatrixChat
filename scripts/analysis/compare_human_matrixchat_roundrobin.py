#!/usr/bin/env python3
"""Cluster-weighted Human / MatrixChat / protocol-baseline comparison table.

Pairwise tests stay Human vs each system (from that system's analysis
summary).  This script only stacks the cluster-weighted means side by side.

"--round-robin-summary" is the XML-tagged full-turn round-robin baseline's
analysis (the paper's default "Round-robin" comparison); the older
token-level round-robin baseline is retired from paper tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TABLE_ROWS = (
    ("boundary_handoff.strict", "Clean/strict"),
    ("boundary_handoff.interruption", "Interruption 1–4"),
    ("boundary_handoff.dirty", "Dirty 5+"),
    ("boundary_handoff.no_handoff", "No handoff"),
    ("columns.overlap_rate", "Overlap rate"),
    ("columns.exactly_one_rate", "Exactly-one"),
    ("columns.silence_rate", "Silence rate"),
    ("speaking_runs.summary.mean", "Mean turn (cols)"),
    ("speaker_changes.rate", "Speaker-change"),
    ("participation.active_token_share_gap", "Participation gap"),
    ("distinct2", "Distinct-2"),
    ("repeated4gram_fraction", "Repeated-4gram"),
    ("judge.aggregate", "Judge score"),
)

SOURCE_CLEAN = (
    ("ami", "AMI clean (raw)"),
    ("werewolf", "Werewolf clean (raw)"),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _overall(summary: dict, metric: str) -> tuple[float | None, float | None]:
    block = (summary.get("metrics") or {}).get(metric) or {}
    overall = block.get("overall") or {}
    human = overall.get("before_mean")
    model = overall.get("after_mean")
    return (
        None if human is None else float(human),
        None if model is None else float(model),
    )


def _source_after(summary: dict, metric: str, source: str) -> float | None:
    block = (summary.get("metrics") or {}).get(metric) or {}
    for row in block.get("by_source") or []:
        if str(row.get("source") or "") == source:
            value = row.get("after_mean")
            return None if value is None else float(value)
    return None


def _source_before(summary: dict, metric: str, source: str) -> float | None:
    block = (summary.get("metrics") or {}).get(metric) or {}
    for row in block.get("by_source") or []:
        if str(row.get("source") or "") == source:
            value = row.get("before_mean")
            return None if value is None else float(value)
    return None


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def build_table(
    human_summary: dict,
    matrix_summary: dict,
    robin_summary: dict,
    extra_summaries: dict[str, dict] | None = None,
) -> list[dict]:
    extras = extra_summaries or {}
    rows: list[dict] = []
    for metric, label in TABLE_ROWS:
        human, _ = _overall(human_summary, metric)
        if human is None:
            human, _ = _overall(matrix_summary, metric)
        _h, matrix = _overall(matrix_summary, metric)
        _h2, robin = _overall(robin_summary, metric)
        row = {
            "metric": metric,
            "label": label,
            "human": human,
            "matrixchat": matrix,
            "round_robin": robin,
        }
        for name, summary in extras.items():
            _ignored, value = _overall(summary, metric)
            row[name] = value
        rows.append(row)
    for source, label in SOURCE_CLEAN:
        row = {
            "metric": f"boundary_handoff.strict.{source}",
            "label": label,
            "human": _source_before(matrix_summary, "boundary_handoff.strict", source),
            "matrixchat": _source_after(matrix_summary, "boundary_handoff.strict", source),
            "round_robin": _source_after(robin_summary, "boundary_handoff.strict", source),
        }
        for name, summary in extras.items():
            row[name] = _source_after(summary, "boundary_handoff.strict", source)
        rows.append(row)
    judge_h, judge_m = _overall(matrix_summary, "judge.aggregate")
    judge_h_rr, judge_r = _overall(robin_summary, "judge.aggregate")
    row = {
        "metric": "judge.delta",
        "label": "Judge Δ vs human",
        "human": None,
        "matrixchat": None if judge_h is None or judge_m is None else judge_m - judge_h,
        "round_robin": None if judge_h_rr is None or judge_r is None else judge_r - judge_h_rr,
    }
    for name, summary in extras.items():
        human_j, model_j = _overall(summary, "judge.aggregate")
        row[name] = None if human_j is None or model_j is None else model_j - human_j
    rows.append(row)
    return rows


def _fmt_p(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 0.001:
        text = f"{value:.3f}"
        return f"{text} (n.s.)" if value >= 0.05 else text
    return f"{value:.2e}"


def build_paired_test_table(summary: dict, stats: dict, *, system_label: str) -> str:
    """Human vs one system: cluster-weighted means, Δ, CI, BH p."""
    lines = [
        f"| Metric (cluster-weighted, AMI+Werewolf) | Human | {system_label} | Δ | 95% CI | BH $p$ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    metrics_block = summary.get("metrics") or {}
    for metric, label in TABLE_ROWS:
        overall = (metrics_block.get(metric) or {}).get("overall") or {}
        human = overall.get("before_mean")
        model = overall.get("after_mean")
        delta = overall.get("mean_difference")
        ci = overall.get("bootstrap_ci") or {}
        test = (stats.get("tests") or {}).get(metric) or {}
        if human is None and model is None:
            continue
        ci_text = "—"
        if ci.get("ci_low") is not None and ci.get("ci_high") is not None:
            ci_text = f"[{_fmt(ci['ci_low'])}, {_fmt(ci['ci_high'])}]"
        lines.append(
            f"| {label} | {_fmt(human)} | {_fmt(model)} | {_fmt(delta)} | {ci_text} | {_fmt_p(test.get('bh_adjusted_p_value'))} |"
        )
    return "\n".join(lines) + "\n"


COLUMN_LABELS = (
    ("human", "Human"),
    ("matrixchat", "MatrixChat"),
    ("round_robin", "Round-robin"),
    ("pass_the_baton", "Pass-the-baton"),
)


def render_markdown(rows: list[dict], extra_keys: tuple[str, ...] = ()) -> str:
    keys = ["human", "matrixchat", "round_robin", *extra_keys]
    labels = [label for key, label in COLUMN_LABELS if key in keys]
    header = "| Metric (cluster-weighted, AMI+Werewolf) | " + " | ".join(labels) + " |"
    divider = "| --- | " + " | ".join(["---:"] * len(labels)) + " |"
    lines = [header, divider]
    for row in rows:
        values = " | ".join(_fmt(row.get(key)) for key in keys)
        lines.append(f"| {row['label']} | {values} |")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrixchat-summary",
        default=str(
            _ROOT
            / "logs/final_runs_20260819_handoff_redef/full_development_t1"
            / "final40_h100_c106_s176106_t1_judged/analysis_judged/summary.json"
        ),
    )
    parser.add_argument(
        "--round-robin-summary",
        required=True,
        help="XML-tagged full-turn round-robin analysis summary.json (the paper's default Round-robin baseline).",
    )
    parser.add_argument("--pass-the-baton-summary", default=None)
    parser.add_argument("--matrixchat-stats", default=None)
    parser.add_argument("--round-robin-stats", default=None)
    parser.add_argument("--pass-the-baton-stats", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _stats_path(summary_path: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    path = Path(summary_path)
    if path.name == "summary.json":
        return str(path.with_name("stats.json"))
    return str(path.parent / "stats.json")


def main(argv=None) -> None:
    args = parse_args(argv)
    matrix = _load(Path(args.matrixchat_summary))
    robin = _load(Path(args.round_robin_summary))
    extras: dict[str, dict] = {}
    extra_keys: list[str] = []
    extra_stats: list[tuple[str, str, dict, str]] = []
    if args.pass_the_baton_summary:
        extras["pass_the_baton"] = _load(Path(args.pass_the_baton_summary))
        extra_keys.append("pass_the_baton")
        extra_stats.append(
            (
                "Pass-the-baton",
                _stats_path(args.pass_the_baton_summary, args.pass_the_baton_stats),
                extras["pass_the_baton"],
                "human_vs_pass_the_baton.md",
            )
        )
    rows = build_table(matrix, matrix, robin, extras or None)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(rows, extra_keys=tuple(extra_keys))
    (output / "human_matrixchat_roundrobin.md").write_text(markdown, encoding="utf-8")
    (output / "human_matrixchat_roundrobin.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    matrix_stats_path = _stats_path(args.matrixchat_summary, args.matrixchat_stats)
    robin_stats_path = _stats_path(args.round_robin_summary, args.round_robin_stats)
    pieces = [markdown]
    if Path(matrix_stats_path).is_file():
        matrix_stats = _load(Path(matrix_stats_path))
        human_vs_mc = build_paired_test_table(matrix, matrix_stats, system_label="MatrixChat")
        (output / "human_vs_matrixchat.md").write_text(human_vs_mc, encoding="utf-8")
        pieces.append("## Human vs MatrixChat\n\n" + human_vs_mc)
    if Path(robin_stats_path).is_file():
        robin_stats = _load(Path(robin_stats_path))
        human_vs_rr = build_paired_test_table(robin, robin_stats, system_label="Round-robin")
        (output / "human_vs_roundrobin.md").write_text(human_vs_rr, encoding="utf-8")
        pieces.append("## Human vs Round-robin\n\n" + human_vs_rr)
    for label, stats_path, summary, filename in extra_stats:
        if Path(stats_path).is_file():
            stats = _load(Path(stats_path))
            table = build_paired_test_table(summary, stats, system_label=label)
            (output / filename).write_text(table, encoding="utf-8")
            pieces.append(f"## Human vs {label}\n\n" + table)
    print("\n".join(pieces))


if __name__ == "__main__":
    main()
