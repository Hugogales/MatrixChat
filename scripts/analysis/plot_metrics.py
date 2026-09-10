"""Export/plot a run's metrics.jsonl as PNG curves.

Reads ``logs/<run_id>/metrics.jsonl`` (written by ``main.py::train_on_dataset``)
and produces separate, easy-to-read PNGs for:

  - content (cross-entropy) loss           -- train + val
  - turn-taking reward                     -- train + val (higher is better)
  - activity accuracy / speak rate         -- train + val, plus the model's own
                                               predicted speak rate (compare
                                               against the ground-truth rate to
                                               check for an always-yield/always-
                                               speak collapse)
  - total loss                             -- train + val
  - activity (balanced BCE) loss           -- train + val

Usage:
    python scripts/analysis/plot_metrics.py --run_dir logs/run_000018
    python scripts/analysis/plot_metrics.py --run_dir logs/run_000018 --out_dir plots/run_000018
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=str, required=True,
                   help="Run log directory containing metrics.jsonl, e.g. logs/run_000018.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to write PNGs (default: <run_dir>/plots).")
    return p.parse_args(argv)


def load_records(run_dir: str):
    path = os.path.join(run_dir, "metrics.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"No records found in {path}")
    return records


def _series(records, key):
    """Extract (steps, values) for records that have a non-None value at `key`."""
    steps, values = [], []
    for r in records:
        v = r.get(key)
        if v is not None:
            steps.append(r["step"])
            values.append(v)
    return steps, values


def _series_nested(records, outer, key):
    steps, values = [], []
    for r in records:
        outer_val = r.get(outer)
        if outer_val is not None and outer_val.get(key) is not None:
            steps.append(r["step"])
            values.append(outer_val[key])
    return steps, values


def _smooth(values, window=15):
    """Simple trailing moving average -- train_* fields are a SINGLE noisy
    micro-batch per record (batch_size=1), so raw per-step values swing
    wildly and can't be read as a trend on their own."""
    if not values:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_pair(ax, records, train_key, val_outer, val_key, title, ylabel):
    """Plot a train series (top-level key) against a val series.

    If ``val_outer`` is ``None``, the val series is read from the top-level
    key ``val_key`` instead of a nested dict (e.g. ``val_loss``).
    """
    ts, tv = _series(records, train_key)
    if val_outer is None:
        vs, vv = _series(records, val_key)
    else:
        vs, vv = _series_nested(records, val_outer, val_key)
    if ts:
        ax.plot(ts, tv, label="train", color="tab:blue", linewidth=1.2)
    if vs:
        ax.plot(vs, vv, label="val", color="tab:orange", marker="o", markersize=3, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir or os.path.join(args.run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_records(args.run_dir)
    print(f"[info] loaded {len(records)} records from {args.run_dir}/metrics.jsonl")

    # 1. Total loss.
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_pair(ax, records, "train_loss", None, "val_loss",
               "Total loss (content_CE + lambda_activity*activity_BCE - lambda_reward*reward)", "loss")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "total_loss.png"), dpi=150)
    plt.close(fig)

    # 2. Content (cross-entropy) loss.
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_pair(ax, records, "train_content_loss", "val_components", "content_loss",
               "Content loss (cross-entropy on speaking targets)", "CE loss")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "content_loss.png"), dpi=150)
    plt.close(fig)

    # 3. Turn-taking reward (higher is better).
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_pair(ax, records, "train_turn_reward", "val_components", "turn_reward",
               "Turn-taking reward (higher is better)", "mean reward")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "turn_reward.png"), dpi=150)
    plt.close(fig)

    # 3b. Reward breakdown: p0 (silence) / p_same / p_handoff / p_overlap.
    # These four ALWAYS sum to ~1 (a partition of the model's own predicted
    # per-agent speak probabilities, not four independent measurements) -- if
    # p_overlap is trending up, p_handoff/p_same are mechanically squeezed
    # down. train_* values are a SINGLE noisy micro-batch per record, so a
    # smoothed trailing average is plotted alongside the raw points; val_*
    # values (averaged over the whole validation set each time) are far more
    # stable and plotted as-is.
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["p0_mean", "p_same_mean", "p_handoff_mean", "p_overlap_mean"]
    labels = ["p(silence)", "p(same speaker)", "p(handoff)", "p(overlap)"]
    colors = ["tab:gray", "tab:blue", "tab:green", "tab:red"]
    for name, label, color in zip(names, labels, colors):
        ts, tv = _series_nested(records, "train_turn_reward_stats", name)
        if ts:
            ax.plot(ts, tv, color=color, alpha=0.15, linewidth=0.8)
            ax.plot(ts, _smooth(tv), label=f"{label} (train, smoothed)", color=color, linewidth=1.4)
        vname = "reward_" + name
        vs, vv = _series_nested(records, "val_components", vname)
        if vs:
            ax.plot(vs, vv, label=f"{label} (val)", color=color, marker="o",
                     markersize=3, linewidth=1.0, linestyle="--")
    ax.set_title("Reward breakdown: p(silence)/p(same)/p(handoff)/p(overlap) -- sums to ~1")
    ax.set_xlabel("step")
    ax.set_ylabel("probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reward_breakdown.png"), dpi=150)
    plt.close(fig)

    # 4. Activity (balanced BCE) loss.
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_pair(ax, records, "train_activity_loss", "val_components", "activity_loss",
               "Activity loss (balanced speak/yield BCE)", "BCE loss")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "activity_loss.png"), dpi=150)
    plt.close(fig)

    # 5. Activity accuracy + speak-rate balance.
    fig, ax = plt.subplots(figsize=(7, 4))
    ts, tv = _series(records, "train_activity_accuracy")
    vs, vv = _series_nested(records, "val_components", "activity_accuracy")
    if ts:
        ax.plot(ts, tv, label="train accuracy", color="tab:green", linewidth=1.2)
    if vs:
        ax.plot(vs, vv, label="val accuracy", color="tab:red", marker="o", markersize=3, linewidth=1.2)

    gs, gv = _series(records, "train_speak_rate")
    ps, pv = _series(records, "train_predicted_speak_rate")
    if gs:
        ax.plot(gs, gv, label="ground-truth speak rate", color="tab:gray", linestyle="--", linewidth=1.0)
    if ps:
        ax.plot(ps, pv, label="model's predicted speak rate", color="tab:purple", linestyle=":", linewidth=1.2)

    ax.set_title("Activity head: accuracy + speak/yield class balance")
    ax.set_xlabel("step")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "activity_accuracy.png"), dpi=150)
    plt.close(fig)

    # 6. Kitchen-sink probe (--probe_every): the PRIMARY conversation-quality
    # signal, independent of loss -- a checkpoint can have a great loss curve
    # while every generated "conversation" is silent or degenerate (see
    # SUCCESS_STORIES.md's 07-27 correction). Only plotted if the run actually
    # used --probe_every > 0.
    probe_names = [
        "clean_handoff_rate",
        "nonoverlap_listener_rate",
        "no_listener_response_rate",
        "overlap_rate",
        "repeated_4gram_fraction",
    ]
    has_probe = any(r.get("probe") is not None for r in records)
    if has_probe:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["tab:green", "tab:blue", "tab:red", "tab:orange", "tab:purple"]
        for name, color in zip(probe_names, colors):
            ps, pv = _series_nested(records, "probe", name)
            if ps:
                ax.plot(ps, pv, label=name, color=color, marker="o", markersize=3, linewidth=1.2)
        ax.set_title("Kitchen-sink probe: conversation/handoff quality (PRIMARY signal, not loss)")
        ax.set_xlabel("step")
        ax.set_ylabel("rate")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "probe.png"), dpi=150)
        plt.close(fig)

    print(f"[info] wrote plots to {out_dir}/")
    # Structure/behavior signal FIRST -- content loss can look great while the
    # actual conversation is silent or degenerate (see SUCCESS_STORIES.md's
    # 07-27 correction). probe.png (real generated handoff/listener behavior)
    # and turn_reward.png (the structural reward, not content) are the most
    # meaningful indicators of whether the model is learning to actually
    # CONVERSE, as opposed to just memorizing content; content_loss/total_loss
    # are listed last deliberately.
    plot_names = []
    if has_probe:
        plot_names.append("probe")
    plot_names += ["reward_breakdown", "turn_reward", "activity_accuracy",
                   "activity_loss", "content_loss", "total_loss"]
    for name in plot_names:
        print(f"  {os.path.join(out_dir, name + '.png')}")

    # Print a short numeric summary of the most recent record too -- reward/
    # structure/behavior metrics first, raw loss last (see note above).
    latest = records[-1]
    print("\n[summary] most recent record (step "
          f"{latest['step']}):")
    if latest.get("probe") is not None:
        p = latest["probe"]
        print(f"  [PROBE -- actual generated conversation behavior] "
              f"clean_handoff_rate={p.get('clean_handoff_rate')}  "
              f"nonoverlap_listener_rate={p.get('nonoverlap_listener_rate')}  "
              f"no_listener_response_rate={p.get('no_listener_response_rate')}  "
              f"overlap_rate={p.get('overlap_rate')}  "
              f"repeated_4gram_fraction={p.get('repeated_4gram_fraction')}")
    print(f"  train_turn_reward={latest.get('train_turn_reward')}  "
          f"train_activity_accuracy={latest.get('train_activity_accuracy')}  "
          f"train_predicted_speak_rate={latest.get('train_predicted_speak_rate')}  "
          f"(ground-truth speak_rate={latest.get('train_speak_rate')})")
    print(f"  train_loss={latest.get('train_loss')}  "
          f"train_content_loss={latest.get('train_content_loss')}  "
          f"train_activity_loss={latest.get('train_activity_loss')}")
    if latest.get("val_components") is not None:
        print(f"  val_loss={latest.get('val_loss')}  val_components={latest.get('val_components')}")


if __name__ == "__main__":
    main()
