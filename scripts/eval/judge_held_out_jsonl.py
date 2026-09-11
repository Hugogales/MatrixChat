#!/usr/bin/env python3
"""Post-hoc blinded LLM judge for an existing paired continuation JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.judge_adapter import (
    HELD_OUT_JUDGE_SYSTEM,
    judge_paired_continuations,
    make_client,
)
from evaluation.paired_continuation import read_jsonl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--judge-base-url", default="http://dh-dgxh100-2.hpc.msoe.edu:8000/v1"
    )
    parser.add_argument("--judge-model", default="meta/llama-4-scout-17b-16e-instruct")
    return parser.parse_args(argv)


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            judge = record.get("judge") or {}
            if isinstance(judge, dict) and (judge.get("human") or {}).get("axes"):
                done.add(str(record.get("trial_id")))
    return done


def main(argv=None) -> None:
    args = parse_args(argv)
    records = read_jsonl(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_ids(output)
    client = make_client(args.judge_base_url)
    remaining = [record for record in records if str(record.get("trial_id")) not in done]
    print(
        json.dumps(
            {
                "input_records": len(records),
                "already_judged": len(done),
                "remaining": len(remaining),
                "judge_model": args.judge_model,
                "rubric_prompt_sha256": hashlib.sha256(
                    HELD_OUT_JUDGE_SYSTEM.encode("utf-8")
                ).hexdigest(),
            },
            indent=2,
        )
    )
    with output.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(remaining, start=1):
            judged = dict(record)
            try:
                judged["judge"] = judge_paired_continuations(
                    client, args.judge_model, record
                )
            except Exception as exc:
                judged["judge"] = {"error": str(exc)}
                print(f"judge_error {index}/{len(remaining)} trial={record.get('trial_id')}: {exc}", flush=True)
            handle.write(json.dumps(judged, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(remaining):
                print(f"judged {index}/{len(remaining)}", flush=True)


if __name__ == "__main__":
    main()
