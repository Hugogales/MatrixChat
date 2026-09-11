"""CLI orchestrator for the Stage-2 data pipeline.

Two modes:

  download : fetch raw sources into ``--raw_dir`` (run on an internet node).
  convert  : tokenize + matrix-convert raw -> Arrow shards in ``--processed_dir``
             plus a manifest (run offline on a CPU node).

Examples
--------
    python -m data.build_dataset --mode download \
        --sources molweni meld werewolf conversation_chronicles when2speak \
        --raw_dir /data/$USER/matrixchat/raw

    python -m data.build_dataset --mode convert \
        --sources molweni meld werewolf conversation_chronicles qwen_distill when2speak \
        --raw_dir /data/$USER/matrixchat/raw \
        --processed_dir /data/$USER/matrixchat/processed \
        --model_path models/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
import uuid
import zipfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.adapters import SOURCES, SOURCE_IDS, get_spec
from data.convert import (
    CONTENT_SUPERVISION_MODES,
    ConvertConfig,
    convert_conversation_examples,
)
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
    p.add_argument("--placeholder_token_id", type=int, default=0,
                   help="Dummy input id for inactive cells (always overridden by the "
                        "model's learned inactive-cell embedding).")
    p.add_argument("--max_agents", type=int, default=8)
    p.add_argument("--min_turn_gap", type=int, default=0,
                   help="Min all-inactive columns inserted between turns (no EOS delimiter).")
    p.add_argument("--max_turn_gap", type=int, default=3,
                   help="Max all-inactive columns inserted between turns.")
    p.add_argument("--pause_threshold_seconds", type=float, default=1.0,
                   help="Timed data: ignore all-speaker gaps up to this duration.")
    p.add_argument("--pause_quantum_seconds", type=float, default=1.0,
                   help="Timed data: seconds beyond threshold per inactive column.")
    p.add_argument("--max_pause_columns", type=int, default=8)
    p.add_argument("--max_flat_len", type=int, default=1536,
                   help="Timed data: chunk examples so num_agents*time does not exceed this.")
    p.add_argument("--context_lookback_columns", type=int, default=0,
                   help="Timed data: prefix each non-first chunk with up to this many REAL "
                        "columns copied from immediately before it (genuine prior dialogue, "
                        "not synthetic filler) as unsupervised context -- so mid-conversation "
                        "chunks aren't structurally cold starts. 0 (default) = old behavior.")
    p.add_argument(
        "--content_supervision_mode",
        choices=CONTENT_SUPERVISION_MODES,
        default="target_only",
        help="Content labels: preserve target-seat-only supervision (default), or "
             "supervise every speaking row for content-bearing sources.",
    )
    p.add_argument("--permute_agents", type=str2bool, default=True)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="Max examples per source (0 = all). Use a small value to test.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _download_one(name, raw_dir):
    spec = get_spec(name)
    if spec.archive_url:
        dest = os.path.join(raw_dir, name)
        # AMI's expected extraction layout; generic archives may use any files.
        if os.path.isdir(os.path.join(dest, "words")) and os.path.isdir(
            os.path.join(dest, "dialogueActs")
        ):
            print(f"[download] {name}: already extracted at {dest}")
            return
        os.makedirs(dest, exist_ok=True)
        archive_name = os.path.basename(spec.archive_url)
        archive_path = os.path.join(dest, archive_name)
        if not os.path.isfile(archive_path):
            print(f"[download] {name}: {spec.archive_url} -> {archive_path}")
            fd, temporary = tempfile.mkstemp(prefix=f"{name}-", suffix=".zip", dir=dest)
            os.close(fd)
            try:
                urllib.request.urlretrieve(spec.archive_url, temporary)
                os.replace(temporary, archive_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            root = os.path.realpath(dest)
            for member in archive.infolist():
                target = os.path.realpath(os.path.join(dest, member.filename))
                if os.path.commonpath([root, target]) != root:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            archive.extractall(dest)
        print(f"[download] {name}: extracted to {dest}")
    elif spec.hf_snapshot and spec.hf_repo:
        dest = os.path.join(raw_dir, name)
        print(f"[download] {name}: snapshot {spec.hf_repo} -> {dest}"
              + (f" (allow_patterns={spec.hf_allow_patterns})" if spec.hf_allow_patterns else ""))
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=spec.hf_repo, repo_type="dataset", local_dir=dest,
            allow_patterns=spec.hf_allow_patterns,
        )
    elif spec.hf_repo:
        print(f"[download] {name}: HF dataset {spec.hf_repo}"
              + (f" (config={spec.hf_config})" if spec.hf_config else ""))
        from datasets import load_dataset
        load_dataset(spec.hf_repo, spec.hf_config, cache_dir=raw_dir)
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

    return tokenize


def convert(args):
    import random

    if args.processed_dir is None:
        raise SystemExit("--processed_dir is required for --mode convert")
    tokenize = _build_tokenizer(args.model_path)
    cfg = ConvertConfig(
        placeholder_token_id=args.placeholder_token_id,
        max_agents=args.max_agents,
        min_turn_gap=args.min_turn_gap,
        max_turn_gap=args.max_turn_gap,
        permute_agents=args.permute_agents,
        pause_threshold_seconds=args.pause_threshold_seconds,
        pause_quantum_seconds=args.pause_quantum_seconds,
        max_pause_columns=args.max_pause_columns,
        max_flat_len=args.max_flat_len,
        context_lookback_columns=args.context_lookback_columns,
        content_supervision_mode=args.content_supervision_mode,
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
                for ex in convert_conversation_examples(conv, tokenize, cfg, source_id, rng):
                    yield ex
                    n += 1
                    if args.limit and n >= args.limit:
                        return

        print(f"[convert] {name} (source_id={source_id}) ...")
        # A unique temporary cache is mandatory: Dataset.from_generator may
        # otherwise silently reuse stale examples after adapter/converter code
        # changes, even with keep_in_memory=True.
        with tempfile.TemporaryDirectory(prefix=f"matrixchat-{name}-") as cache_dir:
            ds = Dataset.from_generator(
                gen,
                keep_in_memory=True,
                cache_dir=cache_dir,
                fingerprint=f"{name}-{uuid.uuid4().hex}",
            )
        out_dir = os.path.join(args.processed_dir, name)
        ds.save_to_disk(out_dir)

        lengths = ds["length"] if len(ds) else []
        token_stats = {
            "num_examples": len(ds),
            "avg_length": (sum(lengths) / len(lengths)) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "overlap_columns": sum(ds["num_overlap_columns"]) if len(ds) else 0,
            "pause_columns": sum(ds["num_pause_columns"]) if len(ds) else 0,
            "speaker_changes": sum(ds["num_speaker_changes"]) if len(ds) else 0,
            "chain_rich_examples": sum(ds["is_chain_rich"]) if len(ds) else 0,
        }
        if len(ds) and "conversation_quality_score" in ds.column_names:
            scores = ds["conversation_quality_score"]
            token_stats["avg_conversation_quality_score"] = sum(scores) / len(scores)
            token_stats["high_quality_examples"] = sum(
                1 for score in scores if score >= 0.55
            )
        update_source(
            processed_dir=args.processed_dir,
            source=name,
            source_id=source_id,
            num_examples=len(ds),
            default_weight=spec.default_weight,
            rel_path=name,
            token_stats=token_stats,
            conversion_config={
                "content_supervision_mode": args.content_supervision_mode,
                "context_lookback_columns": args.context_lookback_columns,
            },
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
