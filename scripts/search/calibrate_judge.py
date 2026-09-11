#!/usr/bin/env python3
"""Validate the Llama judge against a human-labeled matrix-transcript gold set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.search.judge import (
    AXES,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    judge_trial,
    make_client,
    pairwise_judge,
)
from scripts.search.search_state import atomic_write_json, read_json, utc_now


def weighted_kappa(expected: list[int], predicted: list[int], maximum: int = 3) -> float:
    if len(expected) != len(predicted) or not expected:
        return 0.0
    n = len(expected)
    observed = [[0.0] * (maximum + 1) for _ in range(maximum + 1)]
    left = [0.0] * (maximum + 1)
    right = [0.0] * (maximum + 1)
    for a, b in zip(expected, predicted):
        observed[a][b] += 1.0 / n
        left[a] += 1.0 / n
        right[b] += 1.0 / n
    weighted_observed = 0.0
    weighted_expected = 0.0
    for a in range(maximum + 1):
        for b in range(maximum + 1):
            weight = abs(a - b) / maximum
            weighted_observed += weight * observed[a][b]
            weighted_expected += weight * left[a] * right[b]
    if weighted_expected <= 1e-12:
        return 1.0 if weighted_observed <= 1e-12 else 0.0
    return 1.0 - weighted_observed / weighted_expected


def tag_macro_f1(expected: list[list[str]], predicted: list[list[str]]) -> dict:
    tags = sorted({tag for rows in expected + predicted for tag in rows})
    scores = {}
    for tag in tags:
        tp = sum(tag in a and tag in b for a, b in zip(expected, predicted))
        fp = sum(tag not in a and tag in b for a, b in zip(expected, predicted))
        fn = sum(tag in a and tag not in b for a, b in zip(expected, predicted))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores[tag] = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "per_tag_f1": scores,
        "macro_f1": sum(scores.values()) / max(len(scores), 1),
    }


def calibrate(gold: dict, base_url: str, model: str) -> dict:
    client = make_client(base_url)
    judged = []
    failures = []
    for index, item in enumerate(gold.get("items", [])):
        try:
            judged.append(
                {
                    "index": index,
                    "expected": item["labels"],
                    "predicted": judge_trial(client, model, item["trial"]),
                }
            )
        except Exception as exc:
            failures.append({"index": index, "error": str(exc)})
    kappas = {
        axis: weighted_kappa(
            [row["expected"][axis] for row in judged],
            [row["predicted"][axis] for row in judged],
        )
        for axis in AXES
    }
    tags = tag_macro_f1(
        [row["expected"].get("failure_tags", []) for row in judged],
        [row["predicted"].get("failure_tags", []) for row in judged],
    )
    parser_success = len(judged) / max(len(judged) + len(failures), 1)
    pairwise = []
    for pair in gold.get("pairs", []):
        left = gold["items"][int(pair["left"])]["trial"]
        right = gold["items"][int(pair["right"])]["trial"]
        try:
            forward = pairwise_judge(client, model, left, right)
            reverse = pairwise_judge(client, model, right, left)
            reverse_mapped = {"A": "B", "B": "A", "TIE": "TIE"}[
                reverse["winner"]
            ]
            pairwise.append(
                {
                    **pair,
                    "forward": forward,
                    "reverse": reverse,
                    "reverse_mapped": reverse_mapped,
                    "position_consistent": forward["winner"] == reverse_mapped,
                    "correct": (
                        forward["winner"] == pair["expected"]
                        and reverse_mapped == pair["expected"]
                    ),
                }
            )
        except Exception as exc:
            pairwise.append({**pair, "error": str(exc), "position_consistent": False, "correct": False})
    position_consistency = (
        sum(row["position_consistent"] for row in pairwise) / len(pairwise)
        if pairwise
        else 0.0
    )
    pairwise_accuracy = (
        sum(row["correct"] for row in pairwise) / len(pairwise)
        if pairwise
        else 0.0
    )
    launch_ready = (
        len(judged) >= int(gold.get("minimum_items", 150))
        and parser_success >= 0.90
        and all(value >= 0.60 for value in kappas.values())
        and tags["macro_f1"] >= 0.60
        and position_consistency >= 0.80
        and pairwise_accuracy >= 0.80
    )
    return {
        "generated_at": utc_now(),
        "judge_model": model,
        "base_url": base_url,
        "gold_version": gold.get("version"),
        "gold_items": len(gold.get("items", [])),
        "judged_items": len(judged),
        "parser_success_rate": parser_success,
        "weighted_kappa": kappas,
        **tags,
        "position_consistency": position_consistency,
        "pairwise_accuracy": pairwise_accuracy,
        "pairwise_results": pairwise,
        "launch_ready": launch_ready,
        "failure_counts": dict(
            Counter(error["error"] for error in failures)
        ),
        "judgments": judged,
        "failures": failures,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main():
    args = parse_args()
    result = calibrate(read_json(Path(args.gold), {}), args.base_url, args.model)
    atomic_write_json(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k != "judgments"}, indent=2))


if __name__ == "__main__":
    main()
