"""Analyze all arms in a MatrixChat training-race registry.

Produces machine-readable diagnoses, an append-only decision log, a Markdown
summary, and cross-run validation/throughput plots. It never cancels/submits
jobs itself; the hourly Cursor monitor applies actions after reviewing output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.race import diagnose_run, validation_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", required=True)
    parser.add_argument("--append_decisions", action="store_true")
    return parser.parse_args()


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def slurm_status(job_id) -> str:
    if not job_id:
        return "not_submitted"
    result = subprocess.run(
        ["sacct", "-n", "-j", str(job_id), "--format=State", "--parsable2"],
        capture_output=True, text=True,
    )
    states = [line.split("|")[0].strip() for line in result.stdout.splitlines() if line.strip()]
    return states[0] if states else "unknown"


def plot_overlays(sweep_dir: Path, arms, records_by_run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = sweep_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    specs = [
        ("val_loss", None, "Validation total loss", "loss"),
        ("content_loss", "val_components", "Validation content loss", "CE"),
        ("activity_loss", "val_components", "Validation activity loss", "BCE"),
        ("turn_reward", "val_components", "Validation turn reward", "reward"),
        ("activity_accuracy", "val_components", "Validation activity accuracy", "fraction"),
        ("steps_per_hour", None, "Training throughput", "steps/hour"),
        ("learning_rate", None, "Learning rate", "LR"),
        ("grad_norm", None, "Gradient norm", "norm"),
    ]
    for key, outer, title, ylabel in specs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        any_values = False
        for arm in arms:
            records = records_by_run.get(arm["run_name"], [])
            xs, ys = [], []
            for record in records:
                value = record.get(key) if outer is None else record.get(outer, {}).get(key)
                if value is not None:
                    xs.append(record.get("step", 0))
                    ys.append(value)
            if xs:
                any_values = True
                ax.plot(xs, ys, marker="o" if key.startswith("val") or outer else None,
                        markersize=2.5, linewidth=1.1, label=arm["label"])
        if any_values:
            ax.set_title(title)
            ax.set_xlabel("optimizer step")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plots / f"{key}.png", dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    state = read_json(sweep_dir / "state.json", {})
    arms = state.get("arms", [])
    records_by_run = {
        arm["run_name"]: read_jsonl(_ROOT / "logs" / arm["run_name"] / "metrics.jsonl")
        for arm in arms
    }
    best_contents = []
    for records in records_by_run.values():
        values = [
            row.get("val_components", {}).get("content_loss")
            for row in validation_rows(records)
        ]
        best_contents.extend(value for value in values if value is not None)
    leader = min(best_contents) if best_contents else None

    probe_dir = sweep_dir / "probes"
    summaries = []
    now = datetime.now(timezone.utc).isoformat()
    for arm in arms:
        run_name = arm["run_name"]
        best = read_json(_ROOT / "checkpoints" / run_name / "best_checkpoint.json", {})
        probe = read_json(probe_dir / f"{run_name}.json", None)
        diagnosis = diagnose_run(
            records_by_run.get(run_name, []), probe=probe, leader_best_content=leader
        )
        summary = {
            "label": arm["label"],
            "run_name": run_name,
            "job_id": arm.get("job_id"),
            "slurm_state": slurm_status(arm.get("job_id")),
            "best_checkpoint": best,
            "diagnosis": diagnosis,
            "params": arm.get("params", {}),
        }
        summaries.append(summary)

    output = {"generated_at": now, "leader_best_content": leader, "arms": summaries}
    (sweep_dir / "latest_analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    if args.append_decisions:
        with (sweep_dir / "decision_log.jsonl").open("a") as handle:
            handle.write(json.dumps(output) + "\n")

    lines = [
        f"# Training Race Update ({now})",
        "",
        "| Arm | Run | SLURM | Step | Best content | Topology error | Decision | Flags |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for item in summaries:
        diagnosis = item["diagnosis"]
        best_content = diagnosis.get("best_content_loss")
        topology = diagnosis.get("topology_error")
        lines.append(
            f"| {item['label']} | {item['run_name']} | {item['slurm_state']} | "
            f"{diagnosis.get('latest_step', 0)} | "
            f"{best_content:.4f} | " if isinstance(best_content, (float, int)) else
            f"| {item['label']} | {item['run_name']} | {item['slurm_state']} | "
            f"{diagnosis.get('latest_step', 0)} | - | "
        )
        # Complete the most recently appended row without duplicating formatting.
        suffix = (
            f"{topology:.4f} | {diagnosis['decision']} | "
            f"{', '.join(diagnosis['flags']) or '-'} |"
            if isinstance(topology, (float, int))
            else f"- | {diagnosis['decision']} | {', '.join(diagnosis['flags']) or '-'} |"
        )
        lines[-1] += suffix
    (sweep_dir / "latest_report.md").write_text("\n".join(lines) + "\n")
    plot_overlays(sweep_dir, arms, records_by_run)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
