#!/usr/bin/env python3
"""Render publication-style RESULTS.md from judged AMI+Werewolf analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.analysis.compare_human_matrixchat_roundrobin import (  # noqa: E402
    TABLE_ROWS,
    _fmt,
    _overall,
    _source_after,
    _source_before,
    build_paired_test_table,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_p(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 0.001:
        text = f"{value:.3f}"
        return f"{text} (n.s.)" if value >= 0.05 else text
    return f"{value:.2e}"


def _metric_row(summary: dict, stats: dict, metric: str, label: str) -> dict:
    block = (summary.get("metrics") or {}).get(metric) or {}
    overall = block.get("overall") or {}
    ci = overall.get("bootstrap_ci") or {}
    test = (stats.get("tests") or {}).get(metric) or {}
    return {
        "label": label,
        "human": overall.get("before_mean"),
        "model": overall.get("after_mean"),
        "delta": overall.get("mean_difference"),
        "ci_low": ci.get("ci_low"),
        "ci_high": ci.get("ci_high"),
        "bh_p": test.get("bh_adjusted_p_value"),
    }


def _table_md(rows: list[dict], *, human_col: str = "Human", model_col: str = "MatrixChat") -> str:
    lines = [
        f"| Rate (cluster-weighted, AMI+Werewolf) | {human_col} | {model_col} | $\\Delta$ | 95% CI | BH $p$ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        ci = "—"
        if row.get("ci_low") is not None and row.get("ci_high") is not None:
            ci = f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]"
        lines.append(
            f"| {row['label']} | {_fmt(row.get('human'))} | {_fmt(row.get('model'))} | "
            f"{_fmt(row.get('delta'))} | {ci} | {_fmt_p(row.get('bh_p'))} |"
        )
    return "\n".join(lines)


def _comparison_delta(baseline: dict, current: dict, metric: str, label: str) -> str | None:
    _h1, base_model = _overall(baseline, metric)
    _h2, cur_model = _overall(current, metric)
    if base_model is None or cur_model is None:
        return None
    delta = cur_model - base_model
    direction = "better" if metric in {
        "boundary_handoff.strict",
        "columns.exactly_one_rate",
        "distinct2",
        "judge.aggregate",
    } else "worse"
    if metric in {
        "boundary_handoff.dirty",
        "boundary_handoff.no_handoff",
        "columns.overlap_rate",
        "repeated4gram_fraction",
        "speaking_runs.summary.mean",
    }:
        direction = "better" if delta < 0 else "worse"
    return (
        f"| {label} | {_fmt(base_model)} | {_fmt(cur_model)} | {_fmt(delta)} | "
        f"{'↓' if delta < 0 else '↑' if delta > 0 else '='} ({direction} for model) |"
    )


def render(
    summary: dict,
    stats: dict,
    *,
    checkpoint_label: str,
    baseline_summary: dict | None,
    figures_dir: str,
) -> str:
    n_records = stats.get("raw_record_count") or summary.get("raw_record_count")
    n_clusters = stats.get("independent_cluster_count") or summary.get("independent_cluster_count")

    occupancy_metrics = (
        ("columns.silence_rate", "Silence"),
        ("columns.exactly_one_rate", "Exactly one speaker"),
        ("columns.overlap_rate", "Overlap (≥ 2 speakers)"),
    )
    handoff_metrics = (
        ("boundary_handoff.strict", "Clean / strict"),
        ("boundary_handoff.interruption", "Interruption (1–4)"),
        ("boundary_handoff.dirty", "Dirty (5+)"),
        ("boundary_handoff.no_handoff", "No handoff"),
    )
    lexical_metrics = (
        ("distinct2", "Distinct-2"),
        ("repeated4gram_fraction", "Repeated-4gram"),
    )

    occ_rows = [_metric_row(summary, stats, m, l) for m, l in occupancy_metrics]
    hand_rows = [_metric_row(summary, stats, m, l) for m, l in handoff_metrics]
    lex_rows = [_metric_row(summary, stats, m, l) for m, l in lexical_metrics]
    judge_row = _metric_row(summary, stats, "judge.aggregate", "Judge score")
    part_row = _metric_row(
        summary, stats, "participation.active_token_share_gap", "Participation gap"
    )
    turn_row = _metric_row(summary, stats, "speaking_runs.summary.mean", "Mean turn (cols)")
    spk_row = _metric_row(summary, stats, "speaker_changes.rate", "Speaker-change")

    ami_clean_h = _source_before(summary, "boundary_handoff.strict", "ami")
    ami_clean_m = _source_after(summary, "boundary_handoff.strict", "ami")
    ww_clean_h = _source_before(summary, "boundary_handoff.strict", "werewolf")
    ww_clean_m = _source_after(summary, "boundary_handoff.strict", "werewolf")

    lines = [
        "# Results",
        "",
        f"**Checkpoint:** `{checkpoint_label}`.",
        "",
        "All inferential numbers below are from the frozen **development** split after dropping **MELD**. "
        "MELD is scripted sitcom dialogue, not naturally occurring multi-party speech, so it is excluded "
        "from the measurement sample for silence / exactly-one / overlap, participation, handoff, "
        "lexical metrics, the LLM judge, and every paired statistical test. "
        f"The remaining sample is **{n_records}** paired continuations in **{n_clusters}** conversation clusters. "
        "Training still used only the train split. Each example is cut at fraction 0.5 of the new region "
        "after lookback; the model is teacher-forced on the prefix and then generates exactly as many "
        "columns as the held-out human half, at temperature 1.0.",
        "",
        "Metrics are **cluster-weighted**. Tests are paired Wilcoxon signed-rank on cluster means, with "
        "Benjamini–Hochberg correction inside two preregistered families. Intervals are 95% percentile "
        "paired bootstrap CIs on the model-minus-human difference. Positive Δ means the model is larger "
        "than the human continuation.",
        "",
        "## Floor occupancy",
        "",
        _table_md(occ_rows),
        "",
        f"![Occupancy violins]({figures_dir}/occupancy.png)",
        "",
        "## Participation",
        "",
        "| Metric (cluster-weighted, AMI+Werewolf) | Human | MatrixChat | Δ | 95% CI | BH $p$ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| {part_row['label']} | {_fmt(part_row['human'])} | {_fmt(part_row['model'])} | "
        f"{_fmt(part_row['delta'])} | "
        f"[{_fmt(part_row['ci_low'])}, {_fmt(part_row['ci_high'])}] | {_fmt_p(part_row['bh_p'])} |",
        f"| {spk_row['label']} | {_fmt(spk_row['human'])} | {_fmt(spk_row['model'])} | "
        f"{_fmt(spk_row['delta'])} | "
        f"[{_fmt(spk_row['ci_low'])}, {_fmt(spk_row['ci_high'])}] | {_fmt_p(spk_row['bh_p'])} |",
        f"| {turn_row['label']} | {_fmt(turn_row['human'])} | {_fmt(turn_row['model'])} | "
        f"{_fmt(turn_row['delta'])} | "
        f"[{_fmt(turn_row['ci_low'])}, {_fmt(turn_row['ci_high'])}] | {_fmt_p(turn_row['bh_p'])} |",
        "",
        f"![Participation violins]({figures_dir}/participation.png)",
        "",
        "## Handoffs at the cut",
        "",
        _table_md(hand_rows),
        "",
        f"Raw AMI clean: {_fmt(ami_clean_m)} for the model vs {_fmt(ami_clean_h)} for humans. "
        f"Werewolf clean: {_fmt(ww_clean_m)} vs {_fmt(ww_clean_h)}.",
        "",
        f"![Handoff bars]({figures_dir}/handoff.png)",
        "",
        "## Lexical diversity",
        "",
        _table_md(lex_rows),
        "",
        f"![Lexical violins]({figures_dir}/lexical.png)",
        "",
        "## LLM-as-judge",
        "",
        _table_md([judge_row], human_col="Human", model_col="MatrixChat"),
        "",
        f"![Judge score violins]({figures_dir}/judge.png)",
        "",
        "## Human vs MatrixChat (full test table)",
        "",
        "The complete AMI+Werewolf paired tests used for BH correction are in `human_vs_matrixchat.md`.",
        "",
    ]

    if baseline_summary is not None:
        lines.extend(
            [
                "## Comparison to paper baseline (candidate 106, seed 176106)",
                "",
                "Side-by-side cluster-weighted **model** means on the same AMI+Werewolf development "
                "sample (MELD excluded). Δ(current − baseline) is the change in MatrixChat's continuation "
                "behavior; arrows indicate whether a decrease is generally desirable for that metric.",
                "",
                "| Metric | Paper baseline | Current | Δ (current − baseline) | Direction |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for metric, label in TABLE_ROWS:
            row = _comparison_delta(baseline_summary, summary, metric, label)
            if row:
                lines.append(row)
        lines.append("")

    lines.extend(
        [
            "## Example conversations",
            "",
            "Highest-judge MatrixChat continuations with at least two clean handoffs are in "
            "[`best_conversations/best_conversations.md`](best_conversations/best_conversations.md).",
            "",
            "## What this does and does not show",
            "",
            "This bundle measures held-out continuation quality on AMI meetings + Werewolf games under "
            "the frozen publication contract. Compare the **Comparison to paper baseline** section above "
            "for a direct read on whether recent recipe changes (adaptive human-frequency floor-control "
            "weighting, dynstate+GRU agent representation, balanced CE turn reward, best-probe checkpoint "
            "selection) moved structural and judge metrics relative to the earlier candidate-106 paper run.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figures-dir", default="figures_violins")
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--baseline-stats", default=None)
    parser.add_argument("--baseline-label", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = _load(Path(args.summary))
    stats = _load(Path(args.stats))
    baseline = _load(Path(args.baseline_summary)) if args.baseline_summary else None
    text = render(
        summary,
        stats,
        checkpoint_label=args.checkpoint_label,
        baseline_summary=baseline,
        figures_dir=args.figures_dir,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
