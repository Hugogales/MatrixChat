#!/usr/bin/env python3
"""Measure human same-speaker/handoff topology in processed MatrixChat data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_from_disk


CATEGORIES = ("silence", "same", "handoff", "acquisition", "overlap")


def parse_weights(value: str) -> dict[str, float]:
    items = {}
    for piece in value.split(","):
        name, weight = piece.split("=", 1)
        items[name.strip()] = float(weight)
    total = sum(items.values())
    return {name: weight / total for name, weight in items.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--dataset-weights", default="meld=0.15,ami=0.50,werewolf=0.35")
    parser.add_argument("--max-flat-len", type=int, default=2048)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--model-metrics", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot", default=None)
    return parser.parse_args()


def topology_counts(dataset, max_flat_len: int) -> tuple[int, Counter]:
    counts: Counter = Counter()
    examples = 0
    for example in dataset:
        agents = int(example["num_agents"])
        length = int(example["length"])
        if max_flat_len and agents * length > max_flat_len:
            continue
        examples += 1
        current = example["input_activity_mask"]
        labels = example["activity_labels"]
        for column in range(length):
            if not all(labels[agent][column] in (0, 1) for agent in range(agents)):
                continue
            previous = [
                agent for agent in range(agents) if bool(current[agent][column])
            ]
            following = [
                agent for agent in range(agents) if labels[agent][column] == 1
            ]
            counts["total"] += 1
            if not following:
                counts["silence"] += 1
            elif len(following) >= 2:
                counts["overlap"] += 1
            elif len(previous) != 1:
                # Reward-compatible p_handoff currently folds a sole speaker
                # taking the floor after silence/overlap into its handoff bucket.
                counts["acquisition"] += 1
            elif following[0] == previous[0]:
                counts["same"] += 1
            else:
                counts["handoff"] += 1
    return examples, counts


def summarize(examples: int, counts: Counter) -> dict:
    total = counts["total"]
    rates = {category: counts[category] / total for category in CATEGORIES}
    reward_handoff = rates["handoff"] + rates["acquisition"]
    exactly_one = rates["same"] + reward_handoff
    owner_conditioned = rates["same"] + rates["handoff"]
    return {
        "examples": examples,
        "supervised_transitions": total,
        "counts": {category: counts[category] for category in CATEGORIES},
        "rates": rates,
        "reward_compatible_rates": {
            "p0": rates["silence"],
            "p_same": rates["same"],
            "p_handoff": reward_handoff,
            "p_overlap": rates["overlap"],
        },
        "ignore_silence_overlap": {
            "same": rates["same"] / exactly_one,
            "handoff_including_acquisition": reward_handoff / exactly_one,
        },
        "strict_unique_owner_exactly_one": {
            "same": rates["same"] / owner_conditioned,
            "handoff": rates["handoff"] / owner_conditioned,
        },
    }


def source_weighted(summaries: dict[str, dict], weights: dict[str, float]) -> dict:
    active = {name: weight for name, weight in weights.items() if name in summaries}
    total_weight = sum(active.values())
    active = {name: weight / total_weight for name, weight in active.items()}
    rates = {
        category: sum(
            active[name] * summaries[name]["rates"][category] for name in active
        )
        for category in CATEGORIES
    }
    reward_handoff = rates["handoff"] + rates["acquisition"]
    exactly_one = rates["same"] + reward_handoff
    owner_conditioned = rates["same"] + rates["handoff"]
    return {
        "source_weights": active,
        "rates": rates,
        "reward_compatible_rates": {
            "p0": rates["silence"],
            "p_same": rates["same"],
            "p_handoff": reward_handoff,
            "p_overlap": rates["overlap"],
        },
        "ignore_silence_overlap": {
            "same": rates["same"] / exactly_one,
            "handoff_including_acquisition": reward_handoff / exactly_one,
        },
        "strict_unique_owner_exactly_one": {
            "same": rates["same"] / owner_conditioned,
            "handoff": rates["handoff"] / owner_conditioned,
        },
    }


def latest_model_topology(metrics_path: Path) -> dict | None:
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation = [row for row in rows if row.get("val_components")]
    if not validation:
        return None
    row = validation[-1]
    values = row["val_components"]
    same = values["reward_p_same_mean"]
    handoff = values["reward_p_handoff_mean"]
    return {
        "step": row["step"],
        "reward_compatible_rates": {
            "p0": values["reward_p0_mean"],
            "p_same": same,
            "p_handoff": handoff,
            "p_overlap": values["reward_p_overlap_mean"],
        },
        "ignore_silence_overlap": {
            "same": same / (same + handoff),
            "handoff": handoff / (same + handoff),
        },
    }


def make_plot(report: dict, output: Path) -> None:
    validation = report["splits"]["validation"]["source_weighted"]
    human = validation["reward_compatible_rates"]
    labels = ["p(silence)", "p(same)", "p(handoff)", "p(overlap)"]
    keys = ["p0", "p_same", "p_handoff", "p_overlap"]
    series = [("Human validation mixture", [human[key] for key in keys])]
    if report.get("model"):
        model = report["model"]["reward_compatible_rates"]
        series.append(
            (
                f"Model validation, step {report['model']['step']}",
                [model[key] for key in keys],
            )
        )

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    width = 0.36
    positions = list(range(len(labels)))
    for index, (name, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        axis.bar([position + offset for position in positions], values, width, label=name)
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Probability per supervised next-column transition")
    axis.set_title("Human versus model turn-topology probabilities")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    weights = parse_weights(args.dataset_weights)
    report = {
        "processed_dir": str(processed_dir.resolve()),
        "max_flat_len": args.max_flat_len,
        "dataset_weights": weights,
        "definitions": {
            "same": "Exactly one next speaker, equal to the unique current speaker.",
            "handoff": "Exactly one next speaker, different from the unique current speaker.",
            "acquisition": "Exactly one next speaker when the current column has no unique owner.",
            "reward_compatible_p_handoff": "handoff + acquisition, matching turn_reward.py.",
        },
        "splits": {},
    }
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        source_summaries = {}
        for source in weights:
            dataset = load_from_disk(str(processed_dir / source))
            if "dataset_split" in dataset.column_names:
                dataset = dataset.filter(
                    lambda example: example.get("dataset_split") == split
                )
            if len(dataset) == 0:
                continue
            examples, counts = topology_counts(dataset, args.max_flat_len)
            if counts["total"]:
                source_summaries[source] = summarize(examples, counts)
        report["splits"][split] = {
            "by_source": source_summaries,
            "source_weighted": source_weighted(source_summaries, weights),
        }
    if args.model_metrics:
        report["model"] = latest_model_topology(Path(args.model_metrics))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        make_plot(report, plot_path)
    print(output)


if __name__ == "__main__":
    main()
