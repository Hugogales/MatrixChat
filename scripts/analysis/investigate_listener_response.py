#!/usr/bin/env python3
"""Reproducible listener-response diagnostics for one MatrixChat candidate."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="h100p3_cand_00106")
    parser.add_argument("--search-dir", default="searches/h100_2026_08_phase3")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean, y_mean = mean(xs), mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0 or y_var == 0:
        return None
    return numerator / math.sqrt(x_var * y_var)


def align_probe_records(rows: list[dict]) -> list[dict]:
    validations = [row for row in rows if row.get("val_components")]
    records = []
    for row in rows:
        probe = row.get("probe")
        if not probe:
            continue
        prior = [item for item in validations if item["step"] <= row["step"]]
        validation = max(prior, key=lambda item: item["step"]) if prior else None
        calibration = probe.get("calibration_summary") or {}
        record = {
            "step": row["step"],
            "clean_handoff_rate": probe["clean_handoff_rate"],
            "no_listener_response_rate": probe["no_listener_response_rate"],
            "overlap_rate": probe["overlap_rate"],
            "handoff_listener_p_speak_delta_mean": calibration.get(
                "handoff_listener_p_speak_delta_mean"
            ),
        }
        if validation:
            values = validation["val_components"]
            record.update(
                {
                    "validation_step": validation["step"],
                    "val_content_loss": values["content_loss"],
                    "val_activity_loss": values["activity_loss"],
                    "val_activity_accuracy": values["activity_accuracy"],
                    "val_predicted_speak_rate": values["predicted_speak_rate"],
                    "val_true_speak_rate": values["speak_rate"],
                    "val_reward_p0_mean": values["reward_p0_mean"],
                    "val_reward_p_same_mean": values["reward_p_same_mean"],
                    "val_reward_p_handoff_mean": values["reward_p_handoff_mean"],
                    "val_reward_p_overlap_mean": values["reward_p_overlap_mean"],
                }
            )
        records.append(record)
    return records


def temporal_correlations(records: list[dict]) -> dict[str, float | None]:
    # Step 1 is the untrained startup regime and would dominate several
    # correlations without describing the trained model's oscillation.
    trained = [record for record in records if record["step"] > 1]
    target = [record["no_listener_response_rate"] for record in trained]
    keys = [
        "clean_handoff_rate",
        "handoff_listener_p_speak_delta_mean",
        "val_content_loss",
        "val_activity_loss",
        "val_activity_accuracy",
        "val_predicted_speak_rate",
        "val_reward_p0_mean",
        "val_reward_p_same_mean",
        "val_reward_p_handoff_mean",
        "val_reward_p_overlap_mean",
    ]
    result = {}
    for key in keys:
        pairs = [
            (record["no_listener_response_rate"], record.get(key))
            for record in trained
            if record.get(key) is not None
        ]
        result[key] = pearson(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
        )
    return result


def summarize_broad_sweep(path: Path) -> dict:
    sweep = json.loads(path.read_text(encoding="utf-8"))
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in sweep["results"]:
        key = f"agents={row['num_agents']}|context={row['context_kind']}"
        buckets[key].append(row)

    def summarize(rows: list[dict]) -> dict:
        count = len(rows)
        listener_fractions = []
        for row in rows:
            first = row.get("chain", [{}])[0]
            activity_rows = first.get("activity_rows", [])
            reference = int(first.get("reference_speaker", 0))
            listener_rows = [
                activity
                for index, activity in enumerate(activity_rows)
                if index != reference
            ]
            if listener_rows:
                listener_fractions.append(
                    sum("X" in activity for activity in listener_rows)
                    / len(listener_rows)
                )
        return {
            "trials": count,
            "no_listener_response_rate": sum(
                not row["listener_spoke"] for row in rows
            )
            / count,
            "per_listener_agent_response_rate": (
                mean(listener_fractions) if listener_fractions else None
            ),
            "clean_handoff_rate": sum(row["clean_handoff"] for row in rows) / count,
            "overlap_rate": sum(row["overlap"] for row in rows) / count,
        }

    by_agents: dict[str, dict] = {}
    for agent_count in sorted({row["num_agents"] for row in sweep["results"]}):
        by_agents[str(agent_count)] = summarize(
            [row for row in sweep["results"] if row["num_agents"] == agent_count]
        )
    return {
        "headline": {
            key: sweep[key]
            for key in (
                "total_trials",
                "clean_handoff_rate",
                "no_listener_response_rate",
                "overlap_rate",
                "repeated_4gram_fraction",
            )
        },
        "by_agents": by_agents,
        "by_agents_and_context": {
            key: summarize(rows) for key, rows in sorted(buckets.items())
        },
    }


def cross_candidate_rows(search_dir: Path, step: int) -> list[dict]:
    rows = []
    for candidate_dir in sorted((search_dir / "candidates").glob("h100p3_cand_*")):
        config_path = candidate_dir / "config.json"
        sweep_path = candidate_dir / "broad_sweeps" / f"{step}.json"
        if not config_path.is_file() or not sweep_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        rows.append(
            {
                "candidate_id": candidate_dir.name,
                "no_listener_response_rate": sweep["no_listener_response_rate"],
                "clean_handoff_rate": sweep["clean_handoff_rate"],
                "overlap_rate": sweep["overlap_rate"],
                **config,
            }
        )
    return rows


def grouped_cross_candidate_summary(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    return {
        value: {
            "candidates": len(items),
            "mean_no_listener_response_rate": mean(
                item["no_listener_response_rate"] for item in items
            ),
            "mean_clean_handoff_rate": mean(item["clean_handoff_rate"] for item in items),
            "mean_overlap_rate": mean(item["overlap_rate"] for item in items),
        }
        for value, items in sorted(groups.items())
    }


def plot(records: list[dict], broad: dict, output: Path, candidate: str) -> None:
    trained = [record for record in records if record["step"] > 1]
    steps = [record["step"] for record in trained]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    axes[0, 0].plot(
        steps,
        [record["no_listener_response_rate"] for record in trained],
        marker="o",
        label="No listener response",
    )
    axes[0, 0].plot(
        steps,
        [record["clean_handoff_rate"] for record in trained],
        marker="o",
        label="Clean handoff",
    )
    axes[0, 0].set(title="Deterministic three-agent probe", xlabel="Optimizer step", ylabel="Rate")
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 0].legend()

    axes[0, 1].scatter(
        [record["handoff_listener_p_speak_delta_mean"] for record in trained],
        [record["no_listener_response_rate"] for record in trained],
    )
    axes[0, 1].set(
        title="Listener probability movement predicts silence",
        xlabel="Mean listener P(speak) change over generation",
        ylabel="No-listener response rate",
    )

    agent_counts = sorted(broad["by_agents"], key=int)
    axes[1, 0].bar(
        agent_counts,
        [
            broad["by_agents"][agent_count]["no_listener_response_rate"]
            for agent_count in agent_counts
        ],
        label="No listener response",
    )
    axes[1, 0].plot(
        agent_counts,
        [
            broad["by_agents"][agent_count]["clean_handoff_rate"]
            for agent_count in agent_counts
        ],
        color="black",
        marker="o",
        label="Clean handoff",
    )
    axes[1, 0].set(
        title="Step-1500 broad sweep depends strongly on agent count",
        xlabel="Agents in generated conversation",
        ylabel="Rate",
    )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    axes[1, 1].plot(
        steps,
        [record["val_content_loss"] for record in trained],
        marker="o",
        label="Validation content loss",
    )
    twin = axes[1, 1].twinx()
    twin.plot(
        steps,
        [record["val_predicted_speak_rate"] for record in trained],
        color="tab:orange",
        marker=".",
        label="Validation predicted speak rate",
    )
    axes[1, 1].set(
        title="Content overfits while aggregate activity stays calibrated",
        xlabel="Optimizer step",
        ylabel="Validation content loss",
    )
    twin.set_ylabel("Predicted speak rate")
    lines = axes[1, 1].lines + twin.lines
    axes[1, 1].legend(lines, [line.get_label() for line in lines], loc="center left")

    figure.suptitle(f"{candidate}: listener-response investigation")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    candidate = args.candidate
    search_dir = Path(args.search_dir)
    metrics_path = Path(args.logs_dir) / candidate / "metrics.jsonl"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.logs_dir) / candidate / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = align_probe_records(read_jsonl(metrics_path))
    broad_500 = summarize_broad_sweep(
        search_dir / "candidates" / candidate / "broad_sweeps" / "500.json"
    )
    broad_1500 = summarize_broad_sweep(
        search_dir / "candidates" / candidate / "broad_sweeps" / "1500.json"
    )
    cross_rows = cross_candidate_rows(search_dir, step=1500)
    grouping_keys = (
        "activity_pos_weight",
        "lambda_activity",
        "lambda_reward",
        "handoff_bonus_weight",
        "agent_attention_mode",
        "agent_attention_representation",
        "content_supervision_mode",
        "context_lookback_columns",
        "lora_r",
    )
    report = {
        "candidate_id": candidate,
        "temporal_probe_records": records,
        "temporal_correlations_excluding_step1": temporal_correlations(records),
        "broad_sweep_step500": broad_500,
        "broad_sweep_step1500": broad_1500,
        "cross_candidate_step1500_count": len(cross_rows),
        "cross_candidate_step1500_groups": {
            key: grouped_cross_candidate_summary(cross_rows, key)
            for key in grouping_keys
        },
        "methodological_notes": [
            "The periodic probe is greedy, uses 3 agents, 16 generated columns, and 10 fixed prompts.",
            "The broad sweep samples at temperature 0.8, uses 2-5 agents, 48 generated columns, six context kinds, and 576 trials.",
            "Agent-count aggregates are event-level 'any listener spoke' rates and mechanically become easier as more listeners are available.",
            "Cross-candidate summaries are observational and promotion-selected; they are experiment-generating evidence, not causal estimates.",
        ],
    }
    report_path = output_dir / "listener_response_investigation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(
        records,
        broad_1500,
        output_dir / "listener_response_investigation.png",
        candidate,
    )
    print(report_path)


if __name__ == "__main__":
    main()
