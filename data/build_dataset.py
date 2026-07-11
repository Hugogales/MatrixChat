"""CLI orchestrator for the Stage-2 data pipeline.

Two modes:

  download : fetch raw sources into ``--raw_dir`` (run on an internet node).
  convert  : tokenize + matrix-convert raw -> Arrow shards in ``--processed_dir``
             plus a manifest (run offline on a CPU node).

Examples
--------
    python -m data.build_dataset --mode download \
        --sources molweni meld werewolf conversation_chronicles \
        --raw_dir /data/$USER/matrixchat/raw

    python -m data.build_dataset --mode convert \
        --sources molweni meld werewolf conversation_chronicles qwen_distill \
        --raw_dir /data/$USER/matrixchat/raw \
        --processed_dir /data/$USER/matrixchat/processed \
        --model_path models/Qwen3-4B-Instruct-2507 \
        --silence_token_id 151669 --silence_loss_ratio 1.0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.adapters import SOURCES, SOURCE_IDS, get_spec
from data.convert import ConvertConfig, convert_conversation
from data.manifest import update_source


def str2bool(value) -> bool:
    return str(value).strip().lower() in ("true", "t", "yes", "y", "1")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["download", "convert", "peek"], required=True)
    p.add_argument("--sources", nargs="+", default=sorted(SOURCES))
    p.add_argument("--raw_dir", type=str, required=True)
    p.add_argument("--peek_n", type=int, default=2, help="Rows to show per source in --mode peek.")
    p.add_argument("--processed_dir", type=str, default=None)
    p.add_argument("--model_path", type=str, default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--silence_token_id", type=int, default=151669)
    p.add_argument("--max_agents", type=int, default=8)
    p.add_argument("--min_turn_gap", type=int, default=0,
                   help="Min all-silence columns inserted between turns (no EOS delimiter).")
    p.add_argument("--max_turn_gap", type=int, default=3,
                   help="Max all-silence columns inserted between turns.")
    p.add_argument("--permute_agents", type=str2bool, default=True)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="Max examples per source (0 = all). Use a small value to test.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _download_one(name, raw_dir):
    spec = get_spec(name)
    if spec.hf_snapshot and spec.hf_repo:
        dest = os.path.join(raw_dir, name)
        print(f"[download] {name}: snapshot {spec.hf_repo} -> {dest}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=spec.hf_repo, repo_type="dataset", local_dir=dest)
    elif spec.hf_repo:
        print(f"[download] {name}: HF dataset {spec.hf_repo}")
        from datasets import load_dataset
        load_dataset(spec.hf_repo, cache_dir=raw_dir)
    elif spec.git_url:
        dest = os.path.join(raw_dir, name)
        if os.path.exists(dest):
            print(f"[download] {name}: already cloned at {dest}")
        else:
            print(f"[download] {name}: git clone {spec.git_url} -> {dest}")
            subprocess.run(["git", "clone", "--depth", "1", spec.git_url, dest], check=True)
    else:
        print(f"[download] {name}: no remote (generated separately, e.g. distill job). Skipping.")


def download(sources, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    failures = []
    for name in sources:
        try:
            _download_one(name, raw_dir)
        except Exception as exc:  # keep going so one bad source doesn't block the rest
            print(f"[download] {name}: FAILED ({type(exc).__name__}: {exc})")
            failures.append(name)
    if failures:
        print(f"[download] completed with failures: {failures}")
    else:
        print("[download] all sources OK")


def _build_tokenizer(model_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)

    def tokenize(text: str):
        return tok(text, add_special_tokens=False)["input_ids"]

    eos = tok.eos_token_id
    return tokenize, eos


def convert(args):
    import random

    if args.processed_dir is None:
        raise SystemExit("--processed_dir is required for --mode convert")
    tokenize, eos_id = _build_tokenizer(args.model_path)
    cfg = ConvertConfig(
        silence_token_id=args.silence_token_id,
        eos_token_id=eos_id,
        max_agents=args.max_agents,
        min_turn_gap=args.min_turn_gap,
        max_turn_gap=args.max_turn_gap,
        permute_agents=args.permute_agents,
    )

    from datasets import Dataset

    for name in args.sources:
        spec = get_spec(name)
        source_id = SOURCE_IDS[name]
        rng = random.Random(args.seed + source_id)

        def gen():
            n = 0
            for conv in spec.iter_conversations(args.raw_dir):
                conv.content_bearing = spec.content_bearing
                ex = convert_conversation(conv, tokenize, cfg, source_id, rng)
                if ex is None:
                    continue
                yield ex
                n += 1
                if args.limit and n >= args.limit:
                    break

        print(f"[convert] {name} (source_id={source_id}) ...")
        # keep_in_memory=True disables from_generator's on-disk cache, which can
        # otherwise silently return STALE examples after the adapter/converter
        # code changes. Fine for these sources; very large sources (e.g.
        # conversation_chronicles) may need sharded conversion instead.
        ds = Dataset.from_generator(gen, keep_in_memory=True)
        out_dir = os.path.join(args.processed_dir, name)
        ds.save_to_disk(out_dir)

        lengths = ds["length"] if len(ds) else []
        token_stats = {
            "num_examples": len(ds),
            "avg_length": (sum(lengths) / len(lengths)) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
        }
        update_source(
            processed_dir=args.processed_dir,
            source=name,
            source_id=source_id,
            num_examples=len(ds),
            default_weight=spec.default_weight,
            rel_path=name,
            token_stats=token_stats,
        )
        print(f"[convert] {name}: {len(ds)} examples -> {out_dir}")


def peek(args):
    """Print raw structure + a few example conversations per source (the 'view' step)."""
    import itertools

    for name in args.sources:
        spec = get_spec(name)
        print("=" * 70)
        print(f"[peek] {name}  (content_bearing={spec.content_bearing}, weight={spec.default_weight})")
        try:
            convs = list(itertools.islice(spec.iter_conversations(args.raw_dir), args.peek_n))
        except Exception as exc:
            print(f"  FAILED to read {name}: {type(exc).__name__}: {exc}")
            continue
        if not convs:
            print("  (no conversations parsed -- check the adapter field names vs the raw schema)")
            continue
        for i, conv in enumerate(convs):
            print(f"  --- conversation {i}: {len(conv.speakers)} speakers, {len(conv.turns)} turns ---")
            for turn in conv.turns[:8]:
                text = turn.text.replace("\n", " ")
                print(f"    [{conv.speakers[turn.speaker]}] {text[:80]}")
            if len(conv.turns) > 8:
                print(f"    ... (+{len(conv.turns) - 8} more turns)")


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "download":
        download(args.sources, args.raw_dir)
    elif args.mode == "peek":
        peek(args)
    else:
        convert(args)


if __name__ == "__main__":
    main()
