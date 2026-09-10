#!/usr/bin/env python3
"""IEEE-oriented Human / MatrixChat / protocol-baseline paper figures.

Each panel is untitled.  MELD is excluded from the paper sample (scripted
dialogue).  Occupancy, participation, lexical, and judge panels are violins
with mean (diamond) and median (circle) overlays.  Handoff panels are grouped
bar charts of per-example means: those events are 0/1, so violins collapse
to spikes at the endpoints.

Optional vanilla-Qwen protocol baselines: "Round-robin" is the XML-tagged
full-turn round-robin protocol (turns wrapped in `<agentN>...</agentN>`,
generation runs until the close tag or EOS) -- this replaced the older
token-level round-robin baseline as the paper's default round-robin
comparison. "Pass-the-baton" is a hidden-routing-prompt protocol. Occupancy
and handoff are Human vs MatrixChat only and never show these baselines or
the "no handoff" event: all three protocol baselines are handoff/occupancy-
degenerate by construction (exactly one resolved, zero-overlap speaker per
turn), so plotting them there is not informative. The judge panel only
includes systems that have a judge score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.analysis import _record_metric_sides
from evaluation.paired_continuation import read_jsonl

HUMAN = "#2E7D32"
MATRIX = "#1565C0"
ROUND_ROBIN = "#E65100"
BATON = "#00838F"
DATASETS = ("AMI", "Werewolf")
SOURCE_LABEL = {"ami": "AMI", "werewolf": "Werewolf"}
EXCLUDED_SOURCES = frozenset({"meld"})
TWO_WAY = ("Human", "MatrixChat")
THREE_WAY = ("Human", "MatrixChat", "Round-robin")
HUE_HUMAN = "Human"
HUE_MATRIX = "MatrixChat"
HUE_RR = "Round-robin"
HUE_BATON = "Pass-the-baton"
PALETTE = {
    HUE_HUMAN: HUMAN,
    HUE_MATRIX: MATRIX,
    HUE_RR: ROUND_ROBIN,
    HUE_BATON: BATON,
}
# "round_robin" is fed by the XML-tagged full-turn round-robin JSONL -- that
# protocol replaced the older token-level round-robin as the paper's default
# round-robin baseline everywhere except occupancy/handoff (see module
# docstring).
SIDE_TO_HUE = {
    "human": HUE_HUMAN,
    "matrixchat": HUE_MATRIX,
    "round_robin": HUE_RR,
    "pass_the_baton": HUE_BATON,
}
JUDGE_HUES = frozenset({HUE_HUMAN, HUE_MATRIX, HUE_RR, HUE_BATON})
# Occupancy and handoff never show the protocol baselines: all of them are
# degenerate by construction there (exactly one resolved, zero-overlap
# speaker every turn), so a comparison would not be informative.
TWO_WAY_PANELS = frozenset({"occupancy", "handoff"})
BAR_PANELS = frozenset({"handoff"})
HANDOFF_METRICS = frozenset(
    {
        "boundary_handoff.strict",
        "boundary_handoff.interruption",
        "boundary_handoff.dirty",
    }
)

METRICS: dict[str, dict[str, str | None]] = {
    "columns.silence_rate": {"label": "Silence rate", "direction": None},
    "columns.exactly_one_rate": {"label": "Exactly-one-speaker rate", "direction": None},
    "columns.overlap_rate": {"label": "Overlap rate", "direction": None},
    "participation.active_token_share_gap": {
        "label": "Participation gap",
        "direction": None,
    },
    "speaker_changes.rate": {"label": "Speaker-change rate", "direction": None},
    "speaking_runs.summary.mean": {"label": "Mean turn length (columns)", "direction": None},
    "speaking_runs.summary.count": {"label": "Speaking-run count", "direction": None},
    "total_active_tokens": {"label": "Active tokens", "direction": None},
    "boundary_handoff.strict": {"label": "Clean handoff", "direction": None},
    "boundary_handoff.interruption": {"label": "Interruption handoff", "direction": None},
    "boundary_handoff.dirty": {"label": "Dirty handoff", "direction": None},
    "distinct2": {"label": "Distinct-2", "direction": None},
    "repeated4gram_fraction": {"label": "Repeated 4-gram fraction", "direction": None},
    "judge.aggregate": {"label": "Judge score", "direction": None},
}

PANELS = {
    "occupancy": (
        "columns.silence_rate",
        "columns.exactly_one_rate",
        "columns.overlap_rate",
    ),
    "participation": (
        "participation.active_token_share_gap",
        "speaker_changes.rate",
        "speaking_runs.summary.mean",
    ),
    "handoff": (
        "boundary_handoff.strict",
        "boundary_handoff.interruption",
        "boundary_handoff.dirty",
    ),
    "lexical": ("distinct2", "repeated4gram_fraction"),
    "judge": ("judge.aggregate",),
}


def example_key(record: dict) -> tuple:
    """Join key across protocols: trial_id includes checkpoint so it cannot be used."""
    cut = record.get("cut") or {}
    return (
        str(record.get("source") or ""),
        int(record.get("row_index") or 0),
        float(cut.get("fraction") or 0.0),
        int(record.get("generation_seed") or 0),
    )


def join_systems(
    matrix_records: list[dict], extras: dict[str, list[dict]]
) -> list[dict]:
    """Align Human + MatrixChat + extra protocol sides on the same examples."""
    unknown = [name for name in extras if name not in SIDE_TO_HUE]
    if unknown:
        raise ValueError(f"unknown extra sides: {unknown}")
    matrix_by_key = {example_key(record): record for record in matrix_records}
    extra_by_key = {
        name: {example_key(record): record for record in records}
        for name, records in extras.items()
    }
    keys = set(matrix_by_key)
    for mapping in extra_by_key.values():
        keys &= set(mapping)
    joined: list[dict] = []
    for key in sorted(keys):
        matrix_record = matrix_by_key[key]
        if str(matrix_record.get("source") or "").lower() in EXCLUDED_SOURCES:
            continue
        human, matrix = _record_metric_sides(matrix_record)
        item: dict = {
            "source": matrix_record.get("source"),
            "human": human,
            "matrixchat": matrix,
        }
        for name, mapping in extra_by_key.items():
            _ignored, side = _record_metric_sides(mapping[key])
            item[name] = side
        joined.append(item)
    return joined


def join_matrixchat_and_round_robin(
    matrix_records: list[dict], round_robin_records: list[dict]
) -> list[dict]:
    """Align Human + MatrixChat + Round-robin metric sides on the same examples."""
    return join_systems(matrix_records, {"round_robin": round_robin_records})


def _style() -> None:
    sns.set_theme(style="ticks", font="DejaVu Serif")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _ylabel(spec: dict[str, str | None]) -> str:
    return str(spec["label"])


def _keep_record(record: dict) -> bool:
    source = str(record.get("source") or "").lower()
    return source in SOURCE_LABEL and source not in EXCLUDED_SOURCES


def _long_frame_two_way(records: list[dict], metric: str) -> pd.DataFrame:
    sides = [_record_metric_sides(record) for record in records]
    rows: list[dict] = []
    for record, (human, model) in zip(records, sides):
        if not _keep_record(record) or metric not in human or metric not in model:
            continue
        source = SOURCE_LABEL[str(record.get("source") or "").lower()]
        rows.append({"dataset": source, "system": "Human", "value": float(human[metric])})
        rows.append({"dataset": source, "system": "MatrixChat", "value": float(model[metric])})
    return pd.DataFrame(rows)


def _long_frame_joined(
    joined: list[dict], metric: str, hue_order: tuple[str, ...]
) -> pd.DataFrame:
    hue_to_side = {hue: side for side, hue in SIDE_TO_HUE.items()}
    rows: list[dict] = []
    for item in joined:
        source = SOURCE_LABEL.get(str(item.get("source") or "").lower(), None)
        if source is None:
            continue
        for system in hue_order:
            side_name = hue_to_side.get(system)
            if side_name is None:
                continue
            side = item.get(side_name) or {}
            if metric not in side:
                continue
            rows.append({"dataset": source, "system": system, "value": float(side[metric])})
    return pd.DataFrame(rows)


def _hue_offsets(hue_order: tuple[str, ...], *, split: bool) -> dict[str, float]:
    if split and len(hue_order) == 2:
        return {hue_order[0]: -0.18, hue_order[1]: 0.18}
    width = 0.8
    step = width / max(len(hue_order), 1)
    return {
        name: -width / 2.0 + step * (index + 0.5) for index, name in enumerate(hue_order)
    }


def _overlay_mean_median(axis, frame: pd.DataFrame, hue_order: tuple[str, ...], *, split: bool) -> None:
    if frame.empty:
        return
    offsets = _hue_offsets(hue_order, split=split)
    stats = (
        frame.groupby(["dataset", "system"], sort=False)["value"]
        .agg(mean="mean", median="median")
        .reset_index()
    )
    xmap = {name: index for index, name in enumerate(DATASETS)}
    for _, row in stats.iterrows():
        if row["dataset"] not in xmap or row["system"] not in offsets:
            continue
        x = xmap[row["dataset"]] + offsets[row["system"]]
        color = PALETTE[row["system"]]
        axis.scatter(
            [x],
            [row["mean"]],
            marker="D",
            s=28,
            facecolors="white",
            edgecolors=color,
            linewidths=1.1,
            zorder=6,
        )
        axis.scatter(
            [x],
            [row["median"]],
            marker="o",
            s=26,
            facecolors=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
        )


def _draw_violin(
    axis,
    frame: pd.DataFrame,
    spec: dict[str, str | None],
    *,
    hue_order: tuple[str, ...],
) -> None:
    if frame.empty:
        axis.set_axis_off()
        return
    split = len(hue_order) == 2
    sns.violinplot(
        data=frame,
        x="dataset",
        y="value",
        hue="system",
        order=list(DATASETS),
        hue_order=list(hue_order),
        split=split,
        inner=None,
        cut=0,
        density_norm="width",
        linewidth=0.7,
        saturation=0.9,
        palette={name: PALETTE[name] for name in hue_order},
        legend=False,
        ax=axis,
    )
    _overlay_mean_median(axis, frame, hue_order, split=split)
    axis.set_xlabel("")
    axis.set_ylabel(_ylabel(spec))
    axis.tick_params(axis="x", length=0)
    sns.despine(ax=axis, trim=False)


def _draw_bars(
    axis,
    frame: pd.DataFrame,
    spec: dict[str, str | None],
    *,
    hue_order: tuple[str, ...],
) -> None:
    """Grouped bars of per-example means (no error whiskers)."""
    if frame.empty:
        axis.set_axis_off()
        return
    sns.barplot(
        data=frame,
        x="dataset",
        y="value",
        hue="system",
        order=list(DATASETS),
        hue_order=list(hue_order),
        palette={name: PALETTE[name] for name in hue_order},
        errorbar=None,
        width=0.72,
        saturation=0.9,
        legend=False,
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel(_ylabel(spec))
    axis.set_ylim(0.0, 1.02)
    axis.tick_params(axis="x", length=0)
    sns.despine(ax=axis, trim=False)


def _save(figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    figure.savefig(destination.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def _legend_handles(hue_order: tuple[str, ...], *, bars: bool = False):
    handles = [
        Patch(facecolor=PALETTE[name], edgecolor="black", linewidth=0.4, label=name)
        for name in hue_order
    ]
    if bars:
        return handles
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.1,
            markersize=6,
            label="Mean",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#555555",
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=6,
            label="Median",
        )
    )
    return handles


def plot_panel(
    frame_for_metric,
    names: tuple[str, ...],
    destination: Path,
    *,
    hue_order: tuple[str, ...],
    ncols: int | None = None,
    style: str = "violin",
) -> None:
    names = tuple(name for name in names if name in METRICS)
    frames = []
    for name in names:
        frame = frame_for_metric(name)
        if frame is None or getattr(frame, "empty", True):
            continue
        frames.append((name, frame))
    if not frames:
        return
    bars = style == "bars"
    draw = _draw_bars if bars else _draw_violin
    ncols = min(ncols or len(frames), len(frames))
    nrows = int(np.ceil(len(frames) / ncols))
    panel_width = 2.7 if len(hue_order) > 2 else 2.45
    if len(hue_order) >= 4:
        panel_width = 3.15
    if bars and ncols >= 4:
        panel_width = 2.55 if len(hue_order) < 4 else 3.05
    elif bars:
        panel_width = 2.35 if len(hue_order) < 4 else 3.05
    width = panel_width * ncols
    if not bars and len(hue_order) <= 3:
        width = min(7.16, width)
    height = (2.35 if bars else 2.55) * nrows + 0.35
    if len(hue_order) >= 4:
        height += 0.2
    figure, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    flat = axes.ravel()
    for index, (name, frame) in enumerate(frames):
        draw(
            flat[index],
            frame,
            METRICS[name],
            hue_order=hue_order,
        )
    for extra in flat[len(frames) :]:
        extra.set_axis_off()
    legend_handles = _legend_handles(hue_order, bars=bars)
    legend_ncol = 4 if len(legend_handles) > 5 else min(len(legend_handles), 5)
    figure.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=legend_ncol,
        handlelength=1.4,
        fontsize=7 if len(hue_order) >= 4 else 8,
    )
    figure.tight_layout(w_pad=1.15, h_pad=1.35)
    figure.subplots_adjust(bottom=0.22 if len(hue_order) >= 4 else (0.18 if nrows == 1 else 0.12))
    _save(figure, destination)


def plot_single(frame: pd.DataFrame, name: str, destination: Path, *, hue_order: tuple[str, ...]) -> None:
    figure, axis = plt.subplots(figsize=(3.55, 2.7))
    bars = name in HANDOFF_METRICS
    draw = _draw_bars if bars else _draw_violin
    draw(axis, frame, METRICS[name], hue_order=hue_order)
    axis.legend(
        handles=_legend_handles(hue_order, bars=bars),
        frameon=False,
        loc="best",
        fontsize=7,
    )
    figure.tight_layout()
    _save(figure, destination)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(
            _ROOT
            / "logs/final_runs_20260819_handoff_redef/full_development_t1"
            / "final40_h100_c106_s176106_t1_judged/paired_continuations_judged.jsonl"
        ),
        help="MatrixChat judged JSONL (Human + MatrixChat sides).",
    )
    parser.add_argument(
        "--round-robin-input",
        default=None,
        help=(
            "Optional round-robin JSONL, joined on source/row/fraction/seed. "
            "This is the XML-tagged full-turn round-robin protocol (the "
            "paper's default 'Round-robin' baseline); the older token-level "
            "round-robin baseline is retired from paper figures."
        ),
    )
    parser.add_argument(
        "--pass-the-baton-input",
        default=None,
        help="Optional hidden pass-the-baton JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_ROOT / "paper_results_176106_20260819/figures_violins"),
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        default=None,
        help="Subset of panels to draw (default: all).",
    )
    return parser.parse_args(argv)


def _load_extra(path: str | None, side: str) -> tuple[str, list[dict]] | None:
    if not path:
        return None
    records = read_jsonl(path)
    if not records:
        raise SystemExit(f"no records in {path}")
    return side, records


def _hues_for_panel(panel: str, available: tuple[str, ...]) -> tuple[str, ...]:
    if panel == "judge":
        return tuple(name for name in available if name in JUDGE_HUES)
    return available


def main(argv=None) -> None:
    args = parse_args(argv)
    _style()
    matrix_records = read_jsonl(args.input)
    if not matrix_records:
        raise SystemExit(f"no records in {args.input}")
    extras: dict[str, list[dict]] = {}
    extra_paths: dict[str, str] = {}
    for path, side in (
        (args.round_robin_input, "round_robin"),
        (args.pass_the_baton_input, "pass_the_baton"),
    ):
        loaded = _load_extra(path, side)
        if loaded is None:
            continue
        extras[loaded[0]] = loaded[1]
        extra_paths[loaded[0]] = str(Path(path).resolve())
    joined = join_systems(matrix_records, extras) if extras else None
    if extras and not joined:
        raise SystemExit("no overlapping examples between MatrixChat and protocol JSONL")
    available = TWO_WAY + tuple(SIDE_TO_HUE[side] for side in extras)

    def frame_for(metric: str, hue: tuple[str, ...]) -> pd.DataFrame:
        if joined is not None and hue != TWO_WAY:
            return _long_frame_joined(joined, metric, hue)
        return _long_frame_two_way(matrix_records, metric)

    output = Path(args.output_dir)
    selected_panels = list(PANELS) if not args.panels else list(args.panels)
    unknown = [name for name in selected_panels if name not in PANELS]
    if unknown:
        raise SystemExit(f"unknown panels: {unknown}")
    panel_hues: dict[str, tuple[str, ...]] = {}
    for panel in selected_panels:
        names = PANELS[panel]
        if panel in TWO_WAY_PANELS or joined is None:
            panel_hue = TWO_WAY
        else:
            panel_hue = _hues_for_panel(panel, available)
        panel_hues[panel] = panel_hue
        ncols = 1 if panel == "judge" else (2 if panel == "lexical" else 3)
        plot_panel(
            lambda metric, hue=panel_hue: frame_for(metric, hue),
            names,
            output / panel,
            hue_order=panel_hue,
            ncols=ncols,
            style="bars" if panel in BAR_PANELS else "violin",
        )
    if args.panels is None:
        singles = output / "single"
        for name in METRICS:
            if name in HANDOFF_METRICS or name.startswith("columns."):
                single_hue = TWO_WAY
            elif name.startswith("judge."):
                single_hue = _hues_for_panel("judge", available if joined is not None else TWO_WAY)
            elif joined is None:
                single_hue = TWO_WAY
            else:
                single_hue = available
            plot_single(
                frame_for(name, single_hue),
                name,
                singles / name.replace(".", "_"),
                hue_order=single_hue,
            )
    n_used = (
        len(joined)
        if joined is not None
        else sum(1 for record in matrix_records if _keep_record(record))
    )
    panel_labels = {panel: list(panel_hues[panel]) for panel in selected_panels}
    manifest_path = output / "manifest.json"
    extra_manifest = {
        "round_robin_input": extra_paths.get("round_robin"),
        "pass_the_baton_input": extra_paths.get("pass_the_baton"),
    }
    if args.panels is not None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        merged = dict(manifest.get("panels") or {})
        if isinstance(merged, dict):
            merged.update(panel_labels)
            panel_labels = merged
        manifest["panels"] = panel_labels
        manifest["handoff_style"] = "grouped bars of per-example means"
        manifest.update({key: value for key, value in extra_manifest.items() if value})
    else:
        manifest = {
            "input": str(Path(args.input).resolve()),
            "n_records": n_used,
            "datasets": list(DATASETS),
            "excluded_sources": sorted(EXCLUDED_SOURCES),
            "mean_marker": "white diamond",
            "median_marker": "filled circle",
            "handoff_style": "grouped bars of per-example means",
            "unit": "one development example (not cluster-collapsed)",
            "panels": panel_labels,
            **extra_manifest,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
