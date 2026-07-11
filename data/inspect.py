"""Inspect processed MatrixChat shards (sanity-check the converter output).

Prints the manifest summary and renders a few example matrices: each agent row
as decoded tokens, with a label legend so you can eyeball that content labels sit
on the target seat, silence labels are limited, and EOS lands at turn ends.

Usage:
    python -m data.inspect --processed_dir /data/$USER/matrixchat/processed \
        --source qwen_distill --num 2 --model_path models/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.manifest import load_manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed_dir", type=str, required=True)
    p.add_argument("--source", type=str, default=None, help="Source to inspect (default: first in manifest).")
    p.add_argument("--num", type=int, default=2)
    p.add_argument("--model_path", type=str, default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--silence_token_id", type=int, default=151669)
    p.add_argument("--layout", choices=["column", "row"], default="column",
                   help="column: time flows down, each agent is a column (reads like the "
                        "conversation). row: each agent is a row.")
    p.add_argument("--max_steps", type=int, default=40, help="Max time steps (columns) to show.")
    p.add_argument("--col_width", type=int, default=18, help="Column width in column layout.")
    p.add_argument("--min_agents", type=int, default=0,
                   help="Only show examples with at least this many agents (find multi-party ones).")
    p.add_argument("--index", type=int, default=None,
                   help="Show one specific example index (ignores --num/--min_agents).")
    return p.parse_args(argv)


def _cell(tok, tid, label, silence_id, width=None):
    """Render one cell as 'token[mark]' where mark shows the label type."""
    text = tok.decode([int(tid)]).replace("\n", "\\n").strip() or "_"
    if int(tid) == silence_id:
        text = "."
    if label == -100:
        mark = " "
    elif int(label) == silence_id:
        mark = "S"   # silence supervised
    else:
        mark = "*"   # content label (next-token target)
    cell = f"{text[:10]}[{mark}]"
    if width is not None:
        cell = cell[:width].ljust(width)
    return cell


def main(argv=None):
    args = parse_args(argv)
    manifest = load_manifest(args.processed_dir)
    print("manifest sources:")
    for name, info in manifest.get("sources", {}).items():
        print(f"  {name}: id={info['source_id']} n={info['num_examples']} "
              f"weight={info['default_weight']} stats={info.get('token_stats', {})}")

    source = args.source or next(iter(manifest.get("sources", {})), None)
    if source is None:
        print("No sources in manifest.")
        return

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    ds = load_from_disk(os.path.join(args.processed_dir, source))
    print(f"\n=== {source}: {len(ds)} examples ===  legend: [*]=content label  [S]=silence label  [ ]=ignore  '.'=silence input")

    indices = [args.index] if args.index is not None else range(len(ds))
    shown = 0
    for i in indices:
        if shown >= args.num and args.index is None:
            break
        ex = ds[i]
        ii, lb = ex["input_ids"], ex["labels"]
        A, T = len(ii), ex["length"]
        if args.index is None and A < args.min_agents:
            continue
        shown += 1
        steps = min(T, args.max_steps)
        print(f"\nexample {i}: source_id={ex['source_id']} A={A} T={T} (showing {steps} steps)")

        if args.layout == "row":
            for a in range(A):
                cells = " ".join(_cell(tok, ii[a][t], lb[a][t], args.silence_token_id) for t in range(steps))
                print(f"  agent {a}: {cells}")
            continue

        # Column layout: time flows down, one column per agent.
        w = args.col_width
        header = "   t | " + " | ".join(f"agent {a}".ljust(w) for a in range(A))
        print(header)
        print("  " + "-" * (len(header) - 2))
        for t in range(steps):
            row = " | ".join(_cell(tok, ii[a][t], lb[a][t], args.silence_token_id, width=w) for a in range(A))
            print(f"  {t:>3} | {row}")


if __name__ == "__main__":
    main()
